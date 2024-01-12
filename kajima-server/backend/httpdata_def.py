#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time
import sys
import json
import threading
import requests
from requests.auth import HTTPBasicAuth

import datetime as dt
#from plugin_module import PluginModule
from pathlib import Path
scriptpath = Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from jsonutils import json2str

class HTTPDataIntegration(object):
    def __init__ (self, args, **kw):
        self.details = {
            'url': kw.get('url', "https://thegear-staging.iot-trialpack.com"),
            'usr': kw.get('username', "IoTPanaAPIY"),
            'pwd': kw.get('password', "TGR!oTPan@23"),
            'interval': kw.get('interval', 15)
        }
        self.deviceList = {}
        logging.debug('Start Data Integration Module')
        self._start_threading()
    
    def _start_threading (self):
        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.run)
        self.th.start()

    def _time_conversion (self, key, tm):
        if tm is None:
            now = dt.datetime.now()
            if key == 'dateFrom':
                preTime = now - dt.timedelta(days=1)
                _tm = preTime.strftime("%Y-%m-%dT%H:%M:%S.%f")
                _tm = '{}Z'.format(_tm[:-3])
            else:
                _tm = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
                _tm = '{}Z'.format(_tm[:-3])
            return _tm
        else:
            _tm = tm.strftime("%Y-%m-%dT%H:%M:%S.%f")
            _tm = '{}Z'.format(_tm[:-3])
            return _tm

    def _filter_data (self, data, req):
        for k, v in req.items(): 
            data[k] = v
        if 'dateFrom' in data:
            data['dateFrom'] = self._time_conversion('dateFrom', data['dateFrom'])
        if 'dateTo' in data:
            data['dateTo'] = self._time_conversion('dateTo', data['dateTo'])
        new_data = {
            key: value for key, value in data.items() if value is not None
        }
        return new_data

    def _form_url_string (self, data):
        _str = ''
        for k, v in data.items():
            _str += '{}={}&'.format(k, v)
        return _str[:-1]

    def _requests (self, url):
        try:
            r = requests.get(url, auth=HTTPBasicAuth(self.details['usr'], self.details['pwd']))
            r.raise_for_status()
            return (json.loads(r.text))
        except requests.exceptions.HTTPError as e:
            logging.error(e.response.text)
            return None

    def _get_iaq_devices (self, **kw):
        _data = {
            'query': "deviceType eq 'iaq'",
            'pageSize': 2000,
            'withTotalPages': 'true'
        }
        _req_data = self._filter_data(_data, kw)
        _url = "{}/inventory/managedObjects?{}".format(self.details['url'], self._form_url_string(_req_data))
        r = self._requests(_url)
        if not r is None:
            _obj = r.get('managedObjects', [])
            self.deviceList = {}
            for o in _obj:
                self.deviceList[o['id']] = {}
            logging.debug('IAQ Device: {} found'.format(len(_obj)))
        else:
            raise ValueError('Unable to retrieve iaq devices')
    
    def _get_measurements (self, **kw):
        _data = {
            'pageSize': 1,
            'source': 78601,
            'revert': 'true',
            'valueFragmentType': None,
            'vlaueFragmentSeries': None,
            'dateFrom': None ,
            'dateTo': None,
        }
        _req_data = self._filter_data(_data, kw)
        _url = "{}/measurement/measurements?{}".format(self.details['url'], self._form_url_string(_req_data))
        r = self._requests(_url)
        if not r is None:
            _mea = r.get('measurements', [])
            for m in _mea:
                _id = m.get('source', {})
                _id = _id.get('id', -1)
                _time = m.get('time', dt.datetime.now())
                if int(_id) > 0:
                    _data = m.get('ktg_iaq', {})
                    if 'T' in _data: self.deviceList[_id]['T'] = _data['T']
                    if 'H' in _data: self.deviceList[_id]['H'] = _data['H']
                    self.deviceList[_id]['timestamp'] = _time
        else:
            raise ValueError('Unable to retrieve measurement for {}'.format(_req_data.get('source', 'error')))

    def run (self):
        while True:
            self._get_iaq_devices()
            for dl in self.deviceList:
                self._get_measurements(source=dl)
            time.sleep(self.details.get('interval', 15) * 60)
            if self.th_quit.is_set():
                break

    def close (self):
        self.th_quit.set()

def load_processing_module (*a, **kw):
    return HTTPDataIntegration(*a, *kw)

if __name__ == "__main__":
    print ('start http data')
    _http = HTTPDataIntegration(1)
    #raise Exception("This module must start with backend server")