import time
import logging
import hashlib
import threading
import sys
import copy
import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))
from jsonutils import json2str

import pandas as pd
import datetime as dt

class PandasUtils(object):
    def __init__(self, db, redis_conn, discard_time=1) -> None:
        self.db = db
        self.redis_conn = redis_conn
        self.discard_time = discard_time
        self.df = pd.DataFrame()

        # start looping thread
        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.loop)
        self.th.start()

        logging.debug('Started PandasUtils ...')
    
    def check_discard_time (self, currTime):
        _startTime = currTime - dt.timedelta(seconds=self.discard_time)
        self.df = self.df.loc[(self.df['serverTime'] >= _startTime) & (self.df['serverTime'] < currTime)]

    def encode_hid (self, hid):
        shid = str(hid)
        ehid = hashlib.blake2b(key=shid.encode(), digest_size=7).hexdigest()
        if 'Unk' in shid or 'unk' in shid or shid.startswith('9999'):
            ehid = 'UNK-{}'.format(ehid[4:])
        return ehid

    def convert_hid (self, hid):
        #print ('Converting human ID of : {}'.format(hid))
        if hid is None: return None
        if 'unknown' == hid.lower():
            return 99999999
        elif 'unk' in hid.lower():
            return int(str('9999') + hid.split('_')[-1].zfill(5))
        else:
            return int(hid.zfill(9))

    def __none_in_dict (self, msg):
        for _, val in msg.items():
            if val is None: return True
        return False

    # {'timestamp': datetime.datetime(2024, 1, 25, 13, 57, 54, 217497), 'result': [{'list': [[1430.9375429963172, 2246.3622394634376, 'Unk_880', '-0.4'], [1410.4723962467556, 2111.5012885288206, 'Unk_1245', '-0.78']], 'timestamp': datetime.datetime(2024, 1, 25, 13, 57, 54, 216319)}], 'pcid': 7000}
    def insert (self, adtype, msg):
        msgList = []
        _msg = {
            'serverTime': dt.datetime.now(),
            'cam_id': adtype,
            'pcid': msg['pcid'],
        }
        for res in msg.get('result', []):
            _msg['timestamp'] = res['timestamp']
            for rl in res.get('list', []):
                #print (rl)
                if len(rl) >= 3:
                    logging.debug('Msg RL: {}'.format(rl))
                    _msg['loc_x'] = rl[0]
                    _msg['loc_y'] = rl[1]
                    _msg['human_id'] = self.convert_hid(rl[2])
                    _msg['human_comfort'] = rl[3] if len(rl) > 3 else 0
                    _msg['confident'] = rl[4] if len(rl) > 4 else -1
                    if not self.__none_in_dict(_msg):
                        msgList.append(copy.deepcopy(_msg))
        if len(msgList) > 0:
            _df = pd.DataFrame(msgList)
            self.df = pd.concat([self.df, _df], ignore_index=True)
            self.df.reset_index()
            # drop duplication
            #self.df[~self.df.duplicated(['human_id', 'timestamp'], keep=False)]
            self.df[['serverTime']] = self.df[['serverTime']].apply(pd.to_datetime)

    def time_format_conversion (self, dTime):
        _time = dTime.astype(dt.datetime)
        _time = dt.datetime.utcfromtimestamp(_time/1e9)
        return _time, _time.strftime('%Y-%m-%d %H:%M:%S')

    def check_hid_in_sql (self, hid, currTime):
        _query = "SELECT human_id FROM human_table WHERE human_id = {}".format(hid)
        _cur = self.db.query(_query)
        if len(_cur) == 0:
            logging.debug('Human ID {} not found in database'.format(hid))
            _query = 'INSERT INTO human_table (human_id, createdAt, updatedAt, is_deleted) VALUES (%s, %s, %s, %s)'
            val = (hid, currTime.strftime('%Y%m%d%H%M%S'), currTime.strftime('%Y%m%d%H%M%S'), 0)
            self.db.execute(_query, data=val, commit=True)
    
    def val_type_conversion (self, data):
        for key, val in data.items():
            if key == 'time': continue
            data[key] = int(val) if not key == 'human_comfort' else float(val)

    def loop (self):
        while True:
            if len(self.df.index) == 0: continue
            now = dt.datetime.now()
            self.check_discard_time(now)
            try:
                for hid in self.df['human_id'].unique():
                    _df = self.df[self.df['human_id'] == hid]
                    if len(_df.index) <= 0: continue
                    _df = _df[_df['confident'] == _df['confident'].max()]

                    # data container
                    _data = {
                        'cam_id': _df['cam_id'].values[0].replace('camera-', ''),
                        'loc_x': _df['loc_x'].values[0],
                        'loc_y': _df['loc_y'].values[0],
                        'time': _df['timestamp'].values[0],
                        'human_id': _df['human_id'].values[0],
                        'human_comfort': _df['human_comfort'].values[0],
                        'microsecond': _df['timestamp'].dt.microsecond.values[0],
                    }

                    # hid and type checking
                    _utcTime, _data['time'] = self.time_format_conversion(_data['time'])
                    self.check_hid_in_sql(_data['human_id'], now)
                    self.val_type_conversion(_data)

                    # insert into database
                    _query = """INSERT INTO location_table (cam_id, loc_x, loc_y, time, human_id, microsecond) VALUES (%s, %s, %s, %s, %s, %s)"""
                    _val = (_data['cam_id'], _data['loc_x'], _data['loc_y'], _data['time'], _data['human_id'], _data['microsecond'])
                    self.db.execute(_query, data=_val, commit=True)

                    # request MQTT forwarding
                    _data['time'] = int(_utcTime.timestamp())
                    #print ('***************')
                    #print (_data)
                    self.redis_conn.publish(
                        'sql.changes.listener', 
                        json2str({
                            'id': _df['cam_id'].values[0], 
                            'pcid': int(_df['pcid'].values[0]),
                            'msg': json2str(_data),
                        })
                    )
            except Exception as e:
                logging.error('Pandas frame looping error: {}'.format(e))
                pass
            if self.th_quit.is_set():
                pass
            time.sleep(self.discard_time)

if __name__ == "__main__":
    pu = PandasUtils(None, None)

    import random
    for x in range(100):
        _msg = {
            'serverTime': dt.datetime.now() - dt.timedelta(seconds=random.randint(0, 4)),
            'cam_id': str(random.randint(7100, 7103)),
            'pcid': 7000,
            'timestamp': dt.datetime.now(),
            'loc_x': random.randint(0, 1920),
            'loc_y': random.randint(0, 1920),
            'human_id': random.randint(10, 20),
            'human_comfort': random.randint(0, 10),
            'confident': random.randint(30, 60)
        }
        pu.insert(_msg['cam_id'], _msg)



