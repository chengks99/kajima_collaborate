import hashlib
import pathlib
import logging
import os
import sys

import datetime as dt

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from jsonutils import json2str

class CheckMD5(object):
    def __init__(self, redis_conn, component_prefix) -> None:
        self.redis = redis_conn
        self.component_prefix = component_prefix

    def _get_dir_list (self):
        dirList = []
        if isinstance(self.msg, str):
            if self.msg == 'all':
                for dir in os.scandir(str(scriptpath.parent)):
                    if dir.is_dir():
                        dirList.append(dir.path)
            else:
                _dir = scriptpath.parent / self.msg
                if _dir.is_dir():
                    dirList.append(str(_dir))

        if isinstance(self.msg, list):
            for l in self.msg:
                _dir = scriptpath.parent / l
                if _dir.is_dir():
                    dirList.append(str(_dir))
        
        return dirList
    
    def _get_file_list (self, dirList):
        fileList = []
        for dl in dirList:
            for (dirpath, dirnames, filenames) in os.walk(dl):
                for f in filenames:
                    if f.endswith(('.py', '.json')):
                        fileList.append(os.path.join(dirpath, f))
        logging.debug('Prepare hash string for {}'.format(', '.join(f for f in fileList)))
        return fileList
    
    def _get_sum_check (self, fileList):
        _sumList = []
        for f in fileList:
            _sumList.append({
                'path': f,
                'hashString': hashlib.md5(open(f,'rb').read()).hexdigest()
            })
        return _sumList

    def report (self, msg={'dir': 'all'}):
        self.msg = msg.get('dir', 'all')
        dirList = self._get_dir_list()
        if len(dirList) != 0:
            fileList = self._get_file_list(dirList)
            sumList = self._get_sum_check(fileList)

            if len(sumList) != 0:
                msg = {'sumcheck': sumList, 'timestamp': dt.datetime.now()}
                self.redis.publish('{}.hashreport'.format(self.component_prefix), json2str(msg))
    
    # msg = {'unmatch': []}
    def sum_check_error (self, msg):
        for f in msg.get('unmatch', []):
            logging.error('Fail hash checking: {}'.format(f))
        exit(1)

    
