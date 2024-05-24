#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
backend server for Kajima Project

Main jobs are:
1. listen to all GPU's status/event
2. distribute status and update database
'''

import logging
import sys
import io
import pathlib
import datetime as dt
import fnmatch
import configparser
import numpy as np
import requests
import json
import base64
import hashlib
from json import JSONEncoder
from io import BytesIO
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from sqlwrapper import SQLDatabase
import argsutils as au
from jsonutils import json2str,str2json
from plugin_module import PluginModule
from pandasutils_new import PandasUtils
from humanAPI import HumanAPI

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class BackendServer (PluginModule):
    component_name = 'BES'
    subscribe_channels = [
                        'camera.*.status', 'camera.*.config', 'camera.*.detail-config', 
                        'camera.*.query', 'camera.*.result',
                        'audio.*.status', 'audio.*.config', 'audio.*.detail-config', 
                        'audio.*.query', 'audio.*.result',
                        'util.*.query', '*.body.notify', 'face.recognition.vector'
                        ]

    def __init__(self, args, **kw) -> None:
        self.redis_conn = au.connect_redis_with_args(args)
        self.args = args
        self.cfg = {}
        self.cam = {}
        self.aud = {}
        self.plugins = {}
        self.body_details = {
            'fvList' : []
        }

        self.lastreport = {}
        
        self.housekeep_period = kw.pop('housekeep_period', 150)
        self.plugin_modules = []

    def __str__ (self):
        return '<BES>'
    
    # get plugin module information and return
    def get_info (self):
        ''' return a dict containing description of this module '''
        r = PluginModule.get_info(self)
        r.update({
            'plugin-modules': [m.component_name for m in self.plugin_modules],
        })
        return r
    
    # read configuration file and split configuration to cfg and plugins
    # for plugin details in config.ini, it should start section by [plugin-(PLUGINNAME)]
    def load_system_configuration (self, file_path):
        cfgf = scriptpath.parent / file_path
        if cfgf.is_file():
            config = configparser.ConfigParser()
            config.read(cfgf)
            for section in config.sections():
                _params = None
                if 'plugin' in section:
                    if not section in self.plugins: self.plugins[section] = {}
                    _params = self.plugins[section]
                else:
                    if not section in self.cfg: self.cfg[section] = {}
                    _params = self.cfg[section]
                
                for key in config[section]:
                    if 'port' in key:
                        _params[key] = int(config[section][key])
                    elif key == 'enabled':
                        if fnmatch.fnmatch(config[section][key], '*rue'):
                            _params[key] = True
                        else:
                            _params[key] = False
                    else:
                        _params[key] = config[section][key]
        else:
            logging.error('Unable to locate config file at {}'.format(str(cfgf)))
            self.close()
    
    # load camera configuration
    def load_camera_configuration (self, file_path):
        cfgf = scriptpath.parent / file_path
        if cfgf.is_file():
            config = configparser.ConfigParser()
            config.read(cfgf)
            for section in config.sections():
                logging.debug('Load configuration for {}'.format(section))
                if not section in self.cam: self.cam[section] = {}
                for key in config[section]:
                    if key == 'id' or key == 'pcid' or key == 'devid':
                        self.cam[section][key] = int(config[section][key])
                    else:
                        self.cam[section][key] = str(config[section][key])
            
                if 'campath' in self.cam[section]:
                    _fpath = scriptpath.parent / self.cam[section]['campath']
                    if _fpath.is_file():
                        with open(str(_fpath)) as _cfgf:
                            self.cam[section]['config'] = json.load(_cfgf)

                self.redis_conn.set("{}.config".format(section.replace('-', '.')), json2str(self.cam[section]))
        else:
            logging.error('Unable to read camera configuration file')
            exit(1)
    
    # load audio configuration
    def load_audio_configuration (self, file_path):
        cfgf = scriptpath.parent / file_path
        if cfgf.is_file():
            config = configparser.ConfigParser()
            config.read(cfgf)
            for section in config.sections():
                logging.debug('Load configuration for {}'.format(section))
                if not section in self.aud: self.aud[section] = {}
                for key in config[section]:
                    if key == 'id' or key == 'pcid':
                        self.aud[section][key] = int(config[section][key])
                    else:
                        self.aud[section][key] = str(config[section][key])
                
                if 'audpath' in self.aud[section]:
                    _fpath = scriptpath.parent / self.aud[section]['audpath']
                    if _fpath.is_file():
                        with open(str(_fpath)) as _cfgf:
                            self.aud[section]['config'] = json.load(_cfgf)
                self.redis_conn.set('audio.{}.config'.format(section), json2str(self.aud[section]))
        else:
            logging.error('Unable to read audio configuration file')
            exit(1)

    # init sql connection
    def init_sql (self, skip=False):
        if skip:
            logging.warning('Debug mode skip SQL connection')
            self.db = None
            self.audio_db = None
            self.init_db = None
        else:
            logging.debug('Connecting to SQL {} ...'.format(self.cfg['sql']['dbhost']))
            self.db = SQLDatabase(au.to_namespace(self.cfg))
            if self.db.connection():
                logging.debug('SQL {} connected ...'.format(self.cfg['sql']['dbhost']))
            else:
                logging.error('Unable to connect to SQL server')
                exit(1)
            
            self.audio_db = SQLDatabase(au.to_namespace(self.cfg))
            if self.audio_db.connection():
                logging.debug('SQL for Audio {} connected ...'.format(self.cfg['sql']['dbhost']))
            else:
                logging.error('Unable to connect to SQL server')
                exit(1)

            self.init_db = SQLDatabase(au.to_namespace(self.cfg))
            if self.init_db.connection():
                logging.debug('SQL for Initialization {} connected ...'.format(self.cfg['sql']['dbhost']))
            else:
                logging.error('Unable to connect to SQL server')
                exit(1)

            self.utility = SQLDatabase(au.to_namespace(self.cfg))
            if self.utility.connection():
                logging.debug('SQL for Utility {} connected ...'.format(self.cfg['sql']['dbhost']))
            else:
                logging.error('Unable to connect to SQL server')
                exit(1)
            
            

    # load plugin module
    def load_plugin_modules (self, **extra_kw):
        import importlib.util
        module_name = lambda m: 'procmod_' + m.replace('-', '')
        self.plugin_modules = []
        main_path = scriptpath.parent / 'backend'
        for key, value in self.plugins.items():
            if value.get('enabled', False):
                _path = value.get('path', None)
                if _path is None:
                    logging.debug('Plugin module {} no path found'.format(key))
                    continue
                fpath = main_path / _path
                if not fpath.is_file():
                    logging.error('Plugin file not found: {}'.format(str(fpath)))
                spec = importlib.util.spec_from_file_location(module_name(key), str(fpath))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.plugin_modules.append(
                    module.load_processing_module(
                        self.redis_conn, self.db, au.to_namespace(self.cfg), **extra_kw
                    )
                )
                logging.info('processing module {} loaded'.format(key))

    def load_face_features (self):
        _query = "SELECT * FROM face_db"
        _details = []
        if not self.init_db is None:
            cur = self.init_db.query(_query)
            for c in cur:
                _details.append({
                    'features': self.loading_bytes(c[1]),
                    'name': c[5],
                    'gender' : 0,
                    'race' : 0,
                    'age' :0
                })
            self.redis_conn.set('person.face.features', json2str({'fvList': _details}))
            self.redis_conn.publish('person.face.updates', json2str({'fvList': _details}))
        logging.debug("Loaded face features size: {}".format(len(_details)))

    def load_body_features (self):
        _query = "SELECT * FROM body_table"
        _details = []
        if not self.init_db is None:
            cur = self.init_db.query(_query)
            for c in cur:
                _details.append({
                    'name': c[1],
                    'features': self.loading_bytes(c[2]),
                })
            self.redis_conn.set('person.body.features', json2str({'fvList': _details}))
            self.redis_conn.publish('person.body.updates', json2str({'fvList': _details}))
        logging.debug('Loaded body features size: {}'.format(len(_details)))
       

    # start backend server
    def start (self, **extra_kw):
        self.load_system_configuration(self.args.cfg)
        logging.info(self.args.cam)
        self.load_camera_configuration(self.args.cam)
        self.load_audio_configuration(self.args.aud)
        self.load_human_api()
        
        _skip = extra_kw.get('skip_sql', False)
        self.init_sql(skip=_skip)
        self.load_face_features()
        self.load_body_features()        

        PluginModule.__init__(self,
            redis_conn=self.redis_conn, db=self.db
        )
        self.load_plugin_modules(**extra_kw)
        # start pandas object
        self.pandas = PandasUtils(
            db=self.db,
            redis_conn=self.redis_conn,
        )
        self.start_listen_bus()
        self.start_thread('housekeep', self.housekeep)
        self.save_info()
        self.start_thread('reset_body', self.reset_body)


    
    def load_human_api (self):
        args = au.to_namespace(self.cfg)
        dets = {
            'ip': args.humip,
            'port': int(args.humport),
            'username': args.humusr,
            'password': args.humpwd,
        }
        self.humAPI = HumanAPI(dets)
    
    def reset_body(self):
        # Timer
            # Get the current time
        import datetime
        import time
        while not self.is_quit():
            current_time = datetime.datetime.now()
            current_hour = current_time.hour
            current_minute = current_time.minute
            current_second = current_time.second
            if current_hour == 19 and current_minute == 58:
                logging.debug('Reseting')
                _query = '''DELETE FROM body_table;'''
                # Convert the input list or array to a NumPy array     
                if self.init_db is not None:
                    self.init_db.execute(_query, commit=False)
                    logging.debug('Deleted body table')  
                self.redis_conn.delete('person.body.features')
                self.body_details = {
                    'fvList' : []
                }
            else:
                #logging.debug(current_time)
                pass
            time.sleep(60)

        
        


    # redis message listener
    def process_redis_msg (self, ch, msg):
        # print (dt.datetime.now(), ch, msg['timestamp'])

        # if not fnmatch.fnmatch(ch, 'camera.*.result'):
        # if ch  == 'camera.7110.result':
        #     logging.debug('{}: redis-msg received from {}: {}'.format(self, ch, msg['timestamp']))
        #     logging.debug('{}: redis-msg received from {}: {}'.format(self, ch, msg['result']))
        # if ch not in self.lastreport.keys():
        #     self.lastreport[ch] = msg['timestamp'].timestamp()
        # else:
        #     print(msg['timestamp'].timestamp()-self.lastreport[ch])
        #     if msg['timestamp'].timestamp() - self.lastreport[ch] > 5:
        #         self.lastreport[ch] = msg['timestamp'].timestamp()
        if fnmatch.fnmatch(ch, 'camera.*.status'):
            self.update_camera_status('.'.join(ch.split('.')[1:-1]), msg) 
        elif fnmatch.fnmatch(ch, 'camera.*.result'):
            self.update_camera_result('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, 'camera.*.query'):
            self.response_camera_query('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, 'audio.*.status'):
            self.update_audio_status('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, 'audio.*.query'):
            self.response_audio_query('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, 'audio.*.result'):
            self.update_audio_result('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, 'face.*.extraction'):
            self.update_face_extraction(msg)
        # elif fnmatch.fnmatch(ch, 'dome.*.query'):
        #     self.response_dome_query('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, '*.body.notify'):
            self.update_body_features(msg)
        elif fnmatch.fnmatch(ch, 'util.*.query'):
            self.update_detection('.'.join(ch.split('.')[1:-1]), msg)
        elif fnmatch.fnmatch(ch, 'face.recognition.*'):
            self.update_face_feature(msg)
            # else:
            #     pass


    # updating camera status
    def update_camera_status (self, adtype, status):
        # logging.debug('Update {} status: {}'.format(adtype, status))  
        _query = '''INSERT INTO device_status (device_id, type, last_message, timestamp) VALUES (%s,%s,%s,%s) 
        ON DUPLICATE KEY UPDATE last_message = VALUES(last_message), timestamp = VALUES(timestamp);'''
        # Convert the input list or array to a NumPy array     
        data = {
            'device_id' : status['camera'],
            'last_message' : "camera.{}.status".format(status['camera']),
            'type' : status['type'],
            'timestamp' : status['timestamp']
        }
        val = (data['device_id'],data['type'], data['last_message'],data['timestamp'])
        if self.init_db is not None:
            self.init_db.execute(_query, data=val, commit=False)
            logging.debug('Update {} status: {}'.format(adtype, status))  
        self.init_db.commit()

    def update_audio_status (self, adtype, status):
        # logging.debug('Update {} status: {}'.format(adtype, status))  
        _query = '''INSERT INTO device_status (device_id, type, last_message, timestamp) VALUES (%s,%s,%s,%s) 
        ON DUPLICATE KEY UPDATE last_message = VALUES(last_message), timestamp = VALUES(timestamp);'''
        # Convert the input list or array to a NumPy array     
        data = {
            'device_id' : status['camera'],
            'last_message' : "audio.{}.status".format(status['camera']),
            'type' : status['type'],
            'timestamp' : status['timestamp']
        }
        val = (data['device_id'],data['type'], data['last_message'],data['timestamp'])
        if self.init_db is not None:
            self.init_db.execute(_query, data=val, commit=False)
            logging.debug('Update {} status: {}'.format(adtype, status))  
        self.init_db.commit()
        
    # updating camera result
    def update_camera_result (self, adtype, result):
        logging.debug('Update {} result @ {}, length: {}'.format(adtype, result['timestamp'], len(result['result'])))
        if not 'pcid' in result:
            result['pcid'] = self.cam[adtype]['pcid']
        # logging.debug(result)

        self.pandas.insert(adtype, result)
    
    # response to camera sql query
    def response_camera_query (self, adtype, query):
        logging.debug('Response {} query: {}'.format(adtype, query))
        if query.get('msgType', None) == 'init':
            try:
                self._process_camera_init_request(adtype, camType='camera')
            except Exception as e:
                logging.debug(e)    

    # update audio result into database
    def update_audio_result (self, adtype, result):
        if not 'pcid' in result:
            result['pcid'] = self.aud[adtype]['pcid']

        if not self.audio_db is None:
            _query = 'INSERT INTO audio_location_table (degree, datetime, loudness, mic_id) VALUES (%s, %s, %s, %s)'
            val = (result['est_posi'], result['timestamp'].strftime('%Y%m%d%H%M%S'), result['audio_level'], result['mic_id'])
            self.audio_db.execute(_query, data=val, commit=True)
            _query = 'INSERT INTO emotion_table (mic_id, emotion, createdAt, updatedAt) VALUES (%s, %s, %s, %s)'
            val = (result['mic_id'], result['status'], result['timestamp'].strftime('%Y%m%d%H%M%S'), dt.datetime.now().strftime('%Y%m%d%H%M%S'))
            self.audio_db.execute(_query, data=val, commit=True)

            self.audio_db.commit()

        logging.debug('Update audio result for {} at {}'.format(adtype, result['timestamp']))
        data = {
                'status' : result['status'],
                'audio_level': result['audio_level'],
                'est_posi' : result['est_posi'],
                'mic_id' : result['mic_id'],
                'pc_id' : result['pcid'],
                'time' :  result['timestamp'].timestamp(),
                # 'microsecond' : result['timestamp'].microsecond
            }
        # logging.debug(data)
        self.redis_conn.publish('sql.changes.listener', json2str({'id': result['mic_id'], 'pcid': int(result['pcid']), 'msg': json2str(data)}))
    
    # processing update utility 
    def update_detection(self,adtype,result):
        if not 'pcid' in result:
            result['pcid'] = self.aud[adtype]['pcid']
        
        if not self.utility is None:
            _query = 'INSERT INTO utility_table (cam_id, datetime, people_count) VALUES (%s, %s, %s)'
            val = (result['cam_id'], result['timestamp'].strftime('%Y%m%d%H%M%S'), result['people_count'])
            self.utility.execute(_query,data = val, commit = True)
            logging.debug("Update detection for {}: count : {}, timestamp : {}".format(result['cam_id'],result['people_count'],result['timestamp'].strftime('%Y%m%d%H%M%S')))
            self.utility.commit()
            data = {
                'util_id': result['cam_id'],
                'util_rate': result['people_count'],
                'time': result['timestamp'].timestamp(),
            }
            self.redis_conn.publish('sql.changes.listener', json2str({'id': result['cam_id'], 'pcid': int(result['pcid']), 'msg': json2str(data)}))
        
    def response_audio_query (self, adtype, query):
        logging.debug('Response {} query: {}'.format(adtype, query))
        if query.get('msgType', None) == 'init':
            self._process_audio_init_request(adtype, query.get('address', ''))

    # def response_dome_query (self, adtype, query):
    #     logging.debug('Response {} query: {}'.format(adtype, query))
    #     if query.get('msgType', None) == 'init':
    #         self._process_camera_init_request(adtype, camType='dome')
    
    def loading_bytes (self, query):
        current_byte = io.BytesIO(query)
        arrDict = {'array': np.load(current_byte, allow_pickle=True)}
        encodedNumpyData = json.dumps(arrDict, cls=NumpyArrayEncoder)
        decodedArrays = json.loads(encodedNumpyData)
        return decodedArrays['array']
    
    

    def get_base_file (self, path,camID = None):
        #! Hardcoded:
        if 'lookup' in path:
            # path = str(scriptpath.parent / 'storage/lookup/cam18_lookup.json')
            if int(camID) > 7100:
                path = str('/home/ewaic/Documents/cam_data/cam_{}/lookup/lookup.json').format(str(int(camID)-7100))
            else:
                path = str('/home/ewaic/Documents/cam_data/dome_{}/lookup/lookup.json').format(str(int(camID)-7000))

        else:
            # path = str(scriptpath.parent / 'storage/floorplan/1639024639_1637229147_1636619360_prdcsg-floorplan.jpg' )
            path = str('/home/ewaic/Desktop/GEAR_Building.png')

        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
        return None

    # sql query process for camera initialization
    def _process_camera_init_request (self, adtype, camType='camera'):
        # camID = adtype.split('-')[1]
        camID = adtype
        resp = {'cam': {}, 'floor': {}, 'mic': {}, 'type': 'init'}
        basefile = {}
        _query = "SELECT * FROM cam_table_demo WHERE cam_id = {}".format(camID)
        if not self.init_db is None:
            cur = self.init_db.query(_query)
            logging.debug('cam_table_demo query done for camera-{}'.format(camID))
            # logging.debug(cur)
            if len(cur) == 1:
                _re = resp['cam']
                for c in cur:
                    _re['cam_name'] = c[1]
                    _re['type'] = c[2]
                    _re['cam_int'] = self.loading_bytes(c[3]) if c[3] is not None else None
                    _re['hom_mat_sit'] = self.loading_bytes(c[4]) if c[4] is not None else None
                    _re['hom_mat_stand'] = self.loading_bytes(c[5]) if c[5] is not None else None
                    _re['ud_vector'] = self.loading_bytes(c[6]) if c[6] is not None else None
                    _re['img_path'] = c[7]
                    _re['source'] = 'rtsp://admin:Welcome123@{}/MediaInput/h264/'.format(c[8])
                    #_re['lookup'] = c[9]
                    _re['lookup'] = self.get_base_file(c[9],camID)
                    #basefile['cam'] = {'lookup': self.get_base_file(c[9])}
                    # _re['topleft'] = [int(i) for i in c[10].split(",")]
            
        _query = "SELECT floor_id FROM cam2floor_table WHERE cam_id = {}".format(camID)
        if not self.init_db is None:
            cur = self.init_db.query(_query)
            logging.debug('cam2floor query done for camera-{}'.format(camID))
            if len(cur) == 1:
                _re = resp['floor']
                for c in cur:
                    _re['floor_id'] = c[0]

                _query = "SELECT * FROM floorplan_table WHERE floor_id = {}".format(_re['floor_id'])
                _cur = self.init_db.query(_query)
                logging.debug('floorplan query done for camera-{}'.format(camID))
                if len(_cur) == 1:
                    for c in _cur:
                        _re['name'] = c[1]
                        # _re['scale_x'] = c[2]
                        # _re['scale_y'] = c[3]
                        _re['image'] = self.get_base_file(c[2])
                        #_re['image'] = c[4]
                        #basefile['floor'] = {'image': self.get_base_file(c[9])}
                        # if len(c) > 4:
                        #     _re['x_dir'] = self.get_base_file(c[4])
                        #     #_re['x_dir'] = c[4]
                        #     #basefile['floor'] = {'x_dir': basefile['floor'])}
                        #     _re['y_dir'] = c[5]
                        #     _re['origin'] = c[6]
                        # else:
                        #     _re['origin'] = [0, 0]
        
        _query = "SELECT mic_id FROM mic2cam_table WHERE cam_id = {}".format(camID)
        if not self.init_db is None:
            cur = self.init_db.query(_query)
            logging.debug('mic2cam query done for camera-{}'.format(camID))
            if len(cur) == 1:
                _re = resp['mic']
                for c in cur:
                    _re['mic_id'] = c[0]

                _query = "SELECT * FROM mic_table WHERE mic_id = {}".format(_re['mic_id'])
                _cur = self.init_db.query(_query)
                if len(_cur) == 1:
                    for c in _cur:
                        _re['name'] = c[1]
                        _re['type'] = c[2]
                        _re['source'] = c[3]
                        _re['saved_path'] = c[4]
        # logging.debug()
        #self.redis_conn.publish('camera.{}.basefile'.format(adtype), json2str(basefile))
        self.redis_conn.set("{}.{}.detail-config".format(camType, adtype), json2str(resp))
        self.redis_conn.publish('{}.{}.sqlquery'.format(camType, adtype), json2str(resp))
        logging.debug('{}.{}.sqlquery'.format(camType, adtype))

    def _process_audio_init_request (self, adtype, addr):
        resp = {'mic': {}, 'type': 'init'}
        _query = "SELECT mic_id FROM mic_table WHERE source={}".format(addr)
        if not self.init_db is None:
            cur = self.init_db.query(_query)
            if len(cur) == 1:
                _re = resp['mic']
                for c in cur:
                    _re['mic_id'] = c[0]
        
        self.redis_conn.set('audio.{}.detail-config'.format(adtype), json2str(resp))
        self.redis_conn.publish('audio.{}.detail-config'.format(adtype), json2str(resp))

    def __get_encrypted_id (self):
        _query = "SELECT * FROM face_db"
        cur = self.init_db.query(_query)
        eIDList = [c[5] for c in cur]
        if len(eIDList) == 0:
            return 777000
        else:
            return eIDList[-1] + 1
        
    def req_http (self, url, data, header=None):
        try:
            if header is None:
                r = requests.post(url, data=data)
            else:
                r =requests.post(url, headers=header, data=data)
            r.raise_for_status()
            return (json.loads(r.text))
        except requests.exceptions.HTTPError as e:
            logging.error(e.response.text)
            return None
        except requests.exceptions.ConnectionError as e:
            logging.error('Failed to establish a new connection')
            return None

    #  update face features into database
    def update_face_feature (self, msg):
        if msg.get('length', -1) != len(msg.get('fvList', [])):
            logging.error('Invalid length in fvList')
        
        for fv in msg.get('fvList', []):
            _query = "SELECT * FROM face_db WHERE face_id = {}".format(fv['name'])
            cur = self.init_db.query(_query)
            if len(cur) == 0:
                encryptedID = self.__get_encrypted_id()
                _query = 'INSERT INTO face_db (name, features, race, age, gender, eID) VALUES (%s, %s, %s, %s, %s, %s)'
                val = (fv['name'], fv['features'], fv['race'], fv['age'], fv['gender'], encryptedID)
                self.humAPI.add_human_data({'gender': fv['gender'], 'ageGroup': fv['age'], 'humanID': encryptedID, 'race': fv['race']})
            else:
                encryptedID = cur[0]['eID']
                _query = """UPDATE face_db SET features = %s age = %s WHERE eID = %s"""
                val = (fv['features'], fv['age'], encryptedID)
                self.humAPI.update_human_data({'gender': fv['gender'], 'ageGroup': fv['age'], 'humanID': encryptedID, 'race': fv['race']})
            self.init_db.execute(_query, data=val, commit=False)

            '''
            if len(cur) == 0:
                _query = 'INSERT INTO face_db (face_id, face_features, createdAt, human_id) VALUES (%s, %s, %s, %s)'
                val = (fv['face_id'], fv['feature'], msg['timestamp'].strftime('%Y%m%d%H%M%S'), fv['human_id'])
                self.init_db.execute(_query, data=val, commit=False)
            else:
                _query = """UPDATE face_db SET last_update = %s face_features = %s WHERE face_id = %s"""
                val = (msg['timestamp'].strftime('%Y%m%d%H%M%S'), fv['feature'], fv['face_id'])
                self.init_db.execute(_query, data=val, commit=False)
            '''
        self.init_db.commit()
        self.load_face_features()
    
    # should contains ['name', 'features', 'person_details']
    def update_body_features (self, msg):
        # if self.body_details is None:
        #     self.body_details = self.redis_conn.get('person.body.features')
        #     logging.debug("???????????????")
        # if not self.body_details is None:
        if msg is not None:
            self.body_details['fvList'].append(msg)
            self.redis_conn.publish('person.body.updates', json2str({'fvList': self.body_details['fvList']}))
            self.redis_conn.set('person.body.features', json2str({'fvList': self.body_details['fvList']}))

            _query = """INSERT INTO body_table (human_id, body_features, time) VALUES (%s, %s, %s)"""
            # Convert the input list or array to a NumPy array
            np_array = np.array(msg['features'])
            np_bytes = BytesIO()
            np.save(np_bytes, np_array, allow_pickle=True)
            np_array =  np_bytes.getvalue()      
            data = {
                'human_id' : msg['name'],
                'body_features' : np_array,
                'time' : dt.datetime.now()
            }
            val = (data['human_id'], data['body_features'],data['time'])
            if self.init_db is not None:
                self.init_db.execute(_query, data=val, commit=False)
                logging.debug("Updated body features for {}".format(data['human_id']))
            self.init_db.commit()
    
    def update_face_extraction (self, msg):
        _Ffv =[]
        if not msg is None:
            _faceFV = msg.get('fvList', [])
            for f in _faceFV:
                hid = hashlib.blake2b(key=str(f['name']).encode(), digest_size=7).hexdigest()
                _query = """SELECT * FROM human_table WHERE human_id = {}""".format(hid)
                cur = self.init_db.query(_query)
                _now = dt.datetime.now()
                if len(cur) == 0:
                    _query = """INSERT INTO human_table (human_id, createdAt, updatedAt, is_deleted, gender, race, age, face_pth) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"""
                    val = (hid, _now.strftime('%Y%m%d%H%M%S'), _now.strftime('%Y%m%d%H%M%S'), 0, f['gender'], f['race'], f['age_grp'], f['fv'])
                    self.init_db.execute(_query, data=val, commit=True)
                else:
                    _query = """UPDATE human_table SET face_pth = %s age = %s race = %s gender=%s updatedAt = %s WHERE human_id = %s"""
                    val = (f['fv'], f['age'], f['race'], f['gender'], _now.strftime('%Y%m%d%H%M%S'))
                    self.init_db.execute(_query, data=val, commit=False)
                _Ffv.append({
                    "Device_ID": "7000",
                    "Data_name": "human_age_gender_race",
                    "Data_type": "string",
                    "Value": '{}_{}_{}_{}'.format(hid, f['age_grp'], f['gender'], f['race'])
                    }
                )
            if len(_Ffv) > 1:
                self.redis_conn.publish('sql.humanChanges.listener', json2str({'fvList': _Ffv, 'pcid': 7000, 'timeStamp': _now}))

    # closing
    def close (self):
        PluginModule.close(self)

    # housekeeping to clean buffer
    def housekeep (self):
        ''' housekeeping thread '''
        while not self.is_quit(self.housekeep_period):
            for mod in self.plugin_modules:
                mod.housekeep()
            PluginModule.housekeep(self) 

if __name__ == "__main__":
    parser = au.init_parser('Kajima Universal Server', redis={}, sql={})
    au.add_arg(parser, '--cfg', h='specify config file {D}', d='config.ini')
    au.add_arg(parser, '--cam', h='specify camera configuration file {D}', d='vis_config.ini')
    au.add_arg(parser, '--aud', h='specify audio configuration file {D}', d='aud_config.ini')
    args = au.parse_args(parser)

    svr = BackendServer(args=args)
    svr.start(skip_sql=False)

    try:
        while not svr.is_quit(10):
            pass
    except KeyboardInterrupt:
        logging.info('Ctrl-C received -- terminating ...')
        svr.close()