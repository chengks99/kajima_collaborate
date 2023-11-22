#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import threading
import fnmatch
import pathlib
import time
import hashlib
import datetime as dt

from plugin_module import PluginModule

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from jsonutils import json2str
from sqlwrapper import SQLDatabase

class MD5SumCheck(PluginModule):
    component_name = 'md5-sumcheck'
    subscribe_channels = ['*.*.hashreport']

    def __init__(self, redis_conn, db, args, **kw):
        self.db = self.init_db(args)
        self.prevRun, self.nextRun = None, None
        self.redis = redis_conn

        PluginModule.__init__(self, redis_conn, db, **kw)
        if not self.standalone:
            self.start_listen_bus()
            self.save_info()
        
        self.start_counter()
    
    def init_db (self, args):
        logging.debug('Connecting to SQL {} ...'.format(args.dbhost))
        db = SQLDatabase(args)
        if db.connection():
            logging.debug('SQL {} connected ...'.format(args.dbhost))
            return db
        else:
            logging.error('Unable to connect to SQL server')
            exit(1)
    
    def start_counter (self):
        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.run)
        self.th.start()
    
    def __run (self):
        req = {'dir': ['all']}
        self.redis.publish('md5.check.request', json2str(req))
        self.prevRun = dt.datetime.now()
        self.nextRun = self.prevRun + dt.timedelta(days=7)
        logging.debug('MD5 request sent with type {}. Next activation is {}'.format(json2str(req), self.nextRun.strftime("%Y-%m-%d %H:%M:%S")))

    def run (self):
        if self.prevRun is None:
            self.__run()            
        
        while True:
            _curr = dt.datetime.now()
            if _curr > self.nextRun:
                self.__run()
            if self.th_quit.is_set():
                break
            time.sleep(60*60*24)

    '''
        md5.*.report should contains:
        {
            'sumcheck': [
                {'path', 'hashString'}
                ]
            },
            'timestamp': datetime object
    '''
    def process_redis_msg (self, ch, msg):
        if fnmatch.fnmatch(ch, '*.*.hashreport'):
            self.process_md5_sum('.'.join(ch.split('.')[1:-1]), msg)
    
    '''
        database should contain hash, timestamp, node id, path
    '''
    def process_md5_report (self, adtype, msg):
        _query = "SELECT * FROM md5_sum WHERE node_id = {}".format(adtype)
        cur = self.db.query(_query)
        logging.debug('checking md5 hash for node id'.format(adtype))
        mismatchList = []
        if len(cur) == 1:
            for c in cur:
                for p in msg.get('sumcheck', []):
                    if p['path'] == c[2]:
                        if p['hashstring'] == c[3]:
                            logging.debug('{} hash string of {} matched ...'.format(adtype, p['path']))
                        else:
                            logging.error('{} hash string {} MISSMATCH !!!'.format(adtype, p['path']))
                            mismatchList.append(p['path'])
        else:
            for p in msg.get('sumcheck', []):
                _query = 'INSERT INTO md5_sum (timestamp, node_id, file_path, hashstring) VALUES (%s, %s, %s, %s)'
                val = (msg['timestamp'].strftime('%Y%m%d%H%M%S'), adtype, p['path'], p['hashString'])
                self.db.execute(_query, data=val, commit=False)
            self.db.commit()
        
        if len(mismatchList) != 0:
            self.redis.publish('md5.{}.error'.format(adtype), json2str({'unmatch': mismatchList}))
            self.redis.delete('*{}*'.format(adtype))

def load_processing_module (*a, **kw):
    return MD5SumCheck(*a, **kw)

if __name__ == "__main__":
    raise Exception('This module must start with backend server')