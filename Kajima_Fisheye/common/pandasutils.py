import time
import logging
import threading
import copy
import pandas as pd
import datetime as dt

import sys
import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))
from jsonutils import json2str

class PandasUtils(object):
    def __init__(self, db, redis_conn, discard_time=1) -> None:
        self.db = db
        self.redis_conn = redis_conn
        self.disard_time = discard_time
        self.df = None

        pd.set_option('display.max_columns', None)

        # start thread
        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.loop)
        self.th.start()

        logging.debug('Started PandasUtils ...')
    
    # discard dataframe which longer than discard_Time
    def check_discard (self, now, discard_time=None):
        if self.df is None: return False
        if self.df.empty: return False
        discard_time = self.disard_time if discard_time is None else discard_time

        _startTime = now - dt.timedelta(seconds=discard_time)
        if not self.lock: 
            logging.debug('Pandas locked...')
            return False
        self.df = self.df[(self.df['serverTime'] > _startTime)]
        #print (self.df['serverTime'].unique())

        #_pd = self.df[self.df[(self.df['serverTime'] > _startTime) & (self.df['serverTime'] < now)]]
        #self.df = _pd
        return True

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
                _msg['human_id'] = l[2]
                msgList.append(copy.deepcopy(_msg))
        #print (len(msgList))
        _new_df = pd.DataFrame(msgList)
        #print ('##### insert')
        if len(self.df.index) == 0:
            self.df = _new_df
        else:
            self.df = pd.concat([self.df, _new_df], ignore_index=True)
        #print (self.df)
        logging.debug('Pandas Frame Length: {}'.format(len(self.df.index)))
    
    # loop every discard_time to retrieve data from dataframe and insert latest unique data into SQL
    def loop (self):
        while True:
            now = dt.datetime.now()
            #if not self.check_discard(now): continue
            if self.df is None: continue
            if len(self.df.index) == 0: continue
            _sdf = copy.deepcopy(self.df)
            _startTime = now - dt.timedelta(seconds=self.disard_time)
            _df = _sdf[_sdf['serverTime'] >= _startTime]
            _sdf.iloc[0:0]

            if len(_df.index) == 0: continue
            for u in _df['human_id'].unique().tolist():
                _hdf = _df[_df['human_id'] == u]
                #print ('#################')
                #print (u)
                #print (_hdf)
                if _hdf.empty: continue
                _edf = _hdf.tail(1)
                #print (_edf)
                
                #!FIXME: insert into database
                #_query = "SELECT * FROM cam_table_demo WHERE cam_id = {}".format(camID)
                #cur = self.db.execute(_query)
                _query = "INSERT INTO location_table (cam_id,loc_x,loc_y,time,human_id,microsecond) VALUES (%s,%s,%s,%s,%s,%s)"

                _query = """INSERT INTO location_table (cam_id, loc_x, loc_y, time, human_id, microsecond) VALUES (%s, %s, %s, %s, %s, %s)"""

                data = {
                    'cam_id': _edf['cam_id'].values[0].replace('camera-', ''),
                    'loc_x': _edf['loc_x'].values[0],
                    'loc_y': _edf['loc_y'].values[0],
                    'time': _edf['timestamp'].values[0],
                    'human_id': _edf['human_id'].values[0],
                    'microsecond': _edf['timestamp'].dt.microsecond.values[0],
                }
                _time = data['time'].astype(dt.datetime)
                _time = pd.to_datetime(_time)
                data['time'] = _time.strftime('%Y-%m-%d %H:%M:%S')

                #FIXME: TESTING CODE
                if data['human_id'].lower() == 'unknown':
                     data['human_id'] = '999989'
                else:
                    data['human_id'] = '99999{}'.format( data['human_id'][-1])
                #FIXME: END TEST
                
                for key, val in data.items():
                    if key == 'time': continue
                    data[key] = int(val)

                val = (data['cam_id'], data['loc_x'], data['loc_y'], data['time'], data['human_id'], data['microsecond'])
                if not self.db is None:
                    cur = self.db.execute(_query, data=val, commit=True)
                
                #print ('Try to publish {} ....'.format(data))
                self.redis_conn.publish('sql.changes.listener', json2str({'id': _edf['cam_id'].values[0], 'pcid': int(_edf['pcid'].values[0]), 'msg': json2str(data)}))
            if self.th_quit.is_set():
                break
            time.sleep(self.disard_time)