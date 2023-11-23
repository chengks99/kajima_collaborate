import time
import logging
import threading
import copy
import pandas as pd
import datetime as dt
import numpy as np
import sys
import pathlib
import redis
import hashlib
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))
from jsonutils import json2str

class PandasUtils(object):
    def __init__(self, db, redis_conn, discard_time=1) -> None:
        self.db = db
        pool = redis.ConnectionPool(
        host = '10.13.3.57',
        port = 6379,
        password = 'ew@icSG23', 
        )
        self.redis_conn = redis.Redis(connection_pool=pool)
        # self.redis_conn = redis_conn
        self.disard_time = discard_time
        self.df = None
        self.humanList = {}

        import queue
        self.dfList = queue.Queue()
        pd.set_option('display.max_columns', None)

        # start thread
        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.loop)
        self.th.start()

        self.enc_hid = False

        logging.debug('Started PandasUtils ...')
    
    # discard dataframe which longer than discard_Time
    def check_discard (self, now, discard_time=None):
        #if self.df is None: return False
        #if self.df.empty: return False
        discard_time = self.disard_time if discard_time is None else discard_time
        try:
            for hid, df in self.humanList.items():
                _startTime = now - dt.timedelta(seconds=discard_time)
                # if not self.lock: 
                #     logging.debug('Pandas locked...')
                #     return False
                _df = df[(df['serverTime'] > _startTime)]
                self.humanList[hid] = _df
        except Exception as e:
            logging.error("Panda check discard error: {}".format(e))
            pass
        #print (self.df['serverTime'].unique())

        #_pd = self.df[self.df[(self.df['serverTime'] > _startTime) & (self.df['serverTime'] < now)]]
        #self.df = _pd
        return True

    def _encode_hid (self, hid):
        enc = hashlib.blake2b(key=str(hid).encode(), digest_size=7).hexdigest()
        if 'Unk' in hid or 'unk' in hid or hid.startswith('9999'):
            enc = 'UNK-{}'.format(enc[4:])
        return enc

    # insert data into dataframe
    def insert (self, adtype, msg):
        if self.df is None: self.df = pd.DataFrame()
        if len(self.df) != 0:
            _now = dt.datetime.now()
            _startTime = _now - dt.timedelta(seconds=5)
            #print ('#####################')
            #print (_now, _startTime)
            _index = self.df[ (self.df['serverTime'] < _startTime) ].index
            self.df.drop(_index , inplace=True)
            #_df = copy.deepcopy(self.df)
            #_df = _df[_df['serverTime'] > _startTime]
        else:
            self.df = pd.DataFrame()
        #print ('#######after discard')

        # msgList = ['human_id1': pd.DataFrame, 'human_id2': pd.DataFrame ]

        msgList = []
        _msg = {}
        _msg['serverTime'] = dt.datetime.now()
        _msg['cam_id'] = adtype
        _msg['pcid'] = msg['pcid']
        for r in msg['result']:
            _msg['timestamp'] = r['timestamp']
            for l in r['list']:
                _msg['loc_x'] = l[0]
                _msg['loc_y'] = l[1]
                if self.enc_hid:
                    _msg['human_id'] = self._encode_hid(l[2])
                else:
                    _msg['human_id'] = l[2]
                _msg['human_comfort'] = l[3] if len(l) > 3 else 0
                msgList.append(copy.deepcopy(_msg))
        #print (len(msgList))
        # from operator import itemgetter

        # # Sort the msgList based on timestamp in descending order
        # msgList.sort(key=itemgetter('timestamp'), reverse=True)

        # # Create a dictionary to store the filtered messages
        # filteredMsgs = {}

        # # Iterate over the sorted msgList
        # for msg in msgList:
        #     human_id = msg['human_id']
        #     timestamp = msg['timestamp']

        #     # Check if human_id is not already present in filteredMsgs or if the timestamp is more recent
        #     if human_id not in filteredMsgs or timestamp > filteredMsgs[human_id]['timestamp']:
        #         filteredMsgs[human_id] = msg

        # # Convert the filteredMsgs dictionary back to a list
        # filteredMsgList = list(filteredMsgs.values())
        if len(msgList) > 0:
            _new_df = pd.DataFrame(msgList)
            _hid = msgList[0]['human_id']
            if _hid in self.humanList:
                _edf = self.humanList[_hid]
                self.humanList[_hid] = pd.concat([_edf, _new_df], ignore_index=True)
            else:
                self.humanList[_hid] = _new_df
            logging.debug('<{}> {} Frame Length: {}'.format(adtype, _hid, len(self.humanList[_hid].index)))

        # _new_df = pd.DataFrame(msgList)
        #print ('##### insert')
        #if len(self.df.index) == 0:
        #    self.df = _new_df
        #else:
        #    self.df = pd.concat([self.df, _new_df], ignore_index=True)
        #print (self.df)
        #logging.debug('<{}> Pandas Frame Length: {}'.format(adtype,len(self.df.index)))
        #self.dfList.put(self.df)
        # logging.debug('Update Camera result for {}: {}'.format(adtype, msg))
        # self.loop()
        
    
    # loop every discard_time to retrieve data from dataframe and insert latest unique data into SQL
    def loop (self):
        while True:
            now = dt.datetime.now()
            if not self.check_discard(now): continue
            #try:
            #    df = self.dfList.get(timeout=1)  # Timeout of 1 second
            #except:
            #    continue
            #if len(df.index) == 0: continue
            #_sdf = copy.deepcopy(df)
            try:
                for hid, df in self.humanList.items():
                    if len(df.index) == 0: continue
                    _startTime = now - dt.timedelta(seconds=self.disard_time)
                    _df = df[df['serverTime'] >= _startTime]
                    df.iloc[0:0]

                    if len(_df.index) == 0: continue
                    _edf = _df.tail(1)

                    _query = "INSERT INTO location_table (cam_id,loc_x,loc_y,time,human_id,microsecond) VALUES (%s,%s,%s,%s,%s,%s)"

                    _query = """INSERT INTO location_table (cam_id, loc_x, loc_y, time, human_id, microsecond) VALUES (%s, %s, %s, %s, %s, %s)"""
                    data = {
                        'cam_id': _edf['cam_id'].values[0].replace('camera-', ''),
                        'loc_x': _edf['loc_x'].values[0],
                        'loc_y': _edf['loc_y'].values[0],
                        'time': _edf['timestamp'].values[0],
                        'human_id': _edf['human_id'].values[0],
                        'human_comfort': _edf['human_comfort'].values[0],
                        'microsecond': _edf['timestamp'].dt.microsecond.values[0],
                    }
                    _time = data['time'].astype(dt.datetime)
                    _time = dt.datetime.utcfromtimestamp(_time/1e9)
                    # _time = pd.to_datetime(_time,unit='s')
                    # logging.debug(_time.strftime('%Y-%m-%d %H:%M:%S'))
                    data['time'] = _time.strftime('%Y-%m-%d %H:%M:%S')

                    if not self.enc_hid:
                        if "unk" in data['human_id'].lower() :
                            data['human_id'] = str(data['cam_id']) + data['human_id'].split("_")[-1].zfill(5)
                        elif "unknown" == data['human_id'].lower() :
                            data['human_id'] = str(data['cam_id']) + data['human_id'].split("_")[-1].zfill(5)
                        else :
                            data['human_id'].zfill(9)
                    
                    # logging.debug(data['human_id'])
                    for key, val in data.items():
                        if key == 'time': continue
                        if self.enc_hid:
                            if key == 'human_id': continue
                        data[key] = int(val) if not key == 'human_comfort' else float(val)
                    
                    val = (data['cam_id'], data['loc_x'], data['loc_y'], data['time'], data['human_id'], data['microsecond'])
                    if not self.db is None:
                        cur = self.db.execute(_query, data=val, commit=True)

                    # logging.debug('Update Camera result for {}: {}'.format(data['cam_id'], json2str(data)))
                    data['time'] = int(_time.timestamp())
                    # logging.debug(data['human_comfort'])
                    #print ('Try to publish {} ....'.format(data))
                    self.redis_conn.publish('sql.changes.listener', json2str({'id': _edf['cam_id'].values[0], 'pcid': int(_edf['pcid'].values[0]), 'msg': json2str(data)}))
            except Exception as e:
                logging.error("Panda frame looping error: {}".format(e))
                pass


            # if self.df is None: continue
            # if len(self.df.index) == 0: continue
            # _sdf = copy.deepcopy(self.df)
            # _startTime = now - dt.timedelta(seconds=self.disard_time)
            # _df = _sdf[_sdf['serverTime'] >= _startTime]
            # _sdf.iloc[0:0]

            # if len(_df.index) == 0: continue
            # for u in _df['human_id'].unique().tolist():
            #     _hdf = _df[_df['human_id'] == u]
            #     #print (u)
            #     #print (_hdf)
            #     if _hdf.empty: continue
            #     _edf = _hdf.tail(1)
            #     # logging.debug(_edf)
                
            #     #!FIXME: insert into database
            #     #_query = "SELECT * FROM cam_table_demo WHERE cam_id = {}".format(camID)
            #     #cur = self.db.execute(_query)
            #     _query = "INSERT INTO location_table (cam_id,loc_x,loc_y,time,human_id,microsecond) VALUES (%s,%s,%s,%s,%s,%s)"

            #     _query = """INSERT INTO location_table (cam_id, loc_x, loc_y, time, human_id, microsecond) VALUES (%s, %s, %s, %s, %s, %s)"""
            #     data = {
            #         'cam_id': _edf['cam_id'].values[0].replace('camera-', ''),
            #         'loc_x': _edf['loc_x'].values[0],
            #         'loc_y': _edf['loc_y'].values[0],
            #         'time': _edf['timestamp'].values[0],
            #         'human_id': _edf['human_id'].values[0],
            #         'human_comfort': _edf['human_comfort'].values[0],
            #         'microsecond': _edf['timestamp'].dt.microsecond.values[0],
            #     }
            #     _time = data['time'].astype(dt.datetime)
            #     _time = dt.datetime.utcfromtimestamp(_time/1e9)
            #     # _time = pd.to_datetime(_time,unit='s')
            #     # logging.debug(_time.strftime('%Y-%m-%d %H:%M:%S'))
            #     data['time'] = _time.strftime('%Y-%m-%d %H:%M:%S')

            #     #FIXME: TESTING CODE
            #     ## Split the human ID from the unk
            #     if "unknown" == data['human_id'].lower() :
            #         data['human_id'] = '99999999'
            #     elif "unk" in data['human_id'].lower() :
            #         data['human_id'] = '9999' + data['human_id'].split("_")[-1].zfill(4)
            #     else :
            #         data['human_id'].zfill(8)
            #     # if data['human_id'].lower() is 'unknown':
            #     #      data['human_id'] = '999989'
            #     # else:
            #     #     data['human_id'] = '99999{}'.format( data['human_id'][-1])
            #     #FIXME: END TEST
            #     # logging.debug(data)
            #     for key, val in data.items():
            #         if key == 'time': continue
            #         data[key] = int(val) if not key == 'human_comfort' else float(val)

            #     val = (data['cam_id'], data['loc_x'], data['loc_y'], data['time'], data['human_id'], data['microsecond'])
            #     if not self.db is None:
            #         cur = self.db.execute(_query, data=val, commit=True)

            #     # logging.debug('Update Camera result for {}: {}'.format(data['cam_id'], json2str(data)))
            #     data['time'] = int(_time.timestamp())
            #     # logging.debug(data['human_comfort'])
            #     #print ('Try to publish {} ....'.format(data))
            #     self.redis_conn.publish('sql.changes.listener', json2str({'id': _edf['cam_id'].values[0], 'pcid': int(_edf['pcid'].values[0]), 'msg': json2str(data)}))
            if self.th_quit.is_set():
                pass
            time.sleep(self.disard_time)