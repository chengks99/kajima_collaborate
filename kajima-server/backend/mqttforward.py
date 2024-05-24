#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import ssl
import json
import hashlib
import base64

import datetime as dt

try:
    from paho.mqtt import client as mqtt_client
except:
    print ('Error import library')
    pass

from plugin_module import PluginModule
from pathlib import Path
scriptpath = Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from jsonutils import json2str
from sqlwrapper import SQLDatabase

DEBUG = True
if DEBUG:
    import copy

class MQTTBroker(object):
    def __init__ (self, args) -> None:
        print (args)
        self.broker = args.broker
        self.port = args.port
        self.topic = args.topic
        if not self.__get_cert_path(args):
            logging.error('Unable to get all certificate files')
            exit(1)
        self.is_connected = False
        logging.debug('Connecting to MQTT broker {}:{} ...'.format(self.broker, self.port))
        self._connection()
        self._start()

    def __get_cert_path (self, args):
        _certDir = args.cert
        self.cert = {
            'cert': scriptpath.parent / _certDir / args.pemcert,
            'key': scriptpath.parent / _certDir /  args.pemkey,
            'ca': scriptpath.parent / _certDir /  args.pemca,
        }

        for k, v in self.cert.items():
            if v.is_file():
                self.cert[k] = str(v)
            else:
                return False
        return True

    def __get_context (self):
        try:
            # ssl_context = ssl.create_default_context()
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
            # ssl_context.check_hostname = False
            # ssl_context.set_alpn_protocols(['http/1.1'])
            ssl_context.load_verify_locations(cafile=self.cert['ca'])
            ssl_context.load_cert_chain(certfile=self.cert['cert'], keyfile=self.cert['key'])
            logging.debug('SSL Context loaded')
            return ssl_context
        except Exception as e:
            logging.error('SSL Context failed: {}'.format(e))
            exit(1)

    def _connection (self):
        self.client = mqtt_client.Client('MQTTForwarding')
        USE_CONTEXT = True
        if USE_CONTEXT:
            self.client.tls_set_context(context=self.__get_context())
            self.client.tls_insecure_set(True)
        else:     
            logging.debug('Use MQTT TLS_set')
            self.client.tls_set(ca_certs=self.cert['ca'], certfile=self.cert['cert'], keyfile=self.cert['key'], tls_version=ssl.PROTOCOL_TLS)
            self.client.tls_insecure_set(False)
        logging.debug(ssl.OPENSSL_VERSION)
        self.client.on_connect = self._on_mqtt_connect
        self.client.connect(self.broker, self.port)
        logging.debug('Connected to MQTT Broker {}:{}'.format(self.broker, self.port))
    
    def _on_mqtt_connect (self, client, userdata, flags, rc):
        if rc == 0:
            #logging.debug('Connected to MQTT Broker {}:{}'.format(self.broker, self.port))
            self.is_connected = True
        else:
            logging.error('Reconnect to MQTT Broker {}:{}'.format(self.broker, self.port))
            self.is_connected = False
            self._connection()
    
    def _start (self):
        self.client.loop_start()
    
    def connection (self):
        return self.is_connected
        
    def get_msg_old (self, pcID, dataList, dataTime, tag):
        currTime = int(dt.datetime.timestamp(dt.datetime.now()))
        if dataTime is None:
            logging.error('Unable to retrieve {} data timestamp'.format(tag))
            return None, currTime
        
        if len(dataList) == 0:
            logging.debug('No {} data to report'.format(tag))
            return None, currTime
        
        dataTime = int(dt.datetime.timestamp(dataTime))
        msg = json.dumps(
            {
                "PC_ID": pcID,
                "Device_data": dataList,
                "Timestamp": dataTime
            }
        )

        logging.debug('Prepared {} message: ID: {}, DataLen: {}, timestamp: {}'.format(tag, pcID, len(dataList), dataTime))
        return msg, currTime
    
    def get_msg (self, dataList):
        pass

    def publish (self, msg):
        # msg['Timestamp'] = int(dt.datetime.timestamp(msg['Timestamp']))
        self.client.publish(self.topic, json.dumps(msg), qos=0, retain=True)
        logging.info('MQTT message sent : {}'.format(msg))

    def close (self):
        self.client.disconnect()
        self.client.loop_stop()

class MQTTForwarding(PluginModule):
    component_name = 'mqtt-forward'
    subscribe_channels = ['sql.changes.listener', 'sql.humanChanges.listener']

    def __init__(self, redis_conn, db, args, **kw):
        self.db = self.init_db(args)
        # self.redis = redis_conn
        import redis
        pool = redis.ConnectionPool(
        host = '10.13.3.57',
        port = 6379,
        password = 'ew@icSG23', 
        )
        self.redis_conn = redis.Redis(connection_pool=pool)
        self.timing = dt.datetime.now()
        self.dbDict = {}
        self.micData = {'time': dt.datetime.now(), 'dataDict': {}}

        self.topic = args.topic
        self.MQTT = True
        if self.MQTT:
            self.tt = MQTTBroker(args=args)

        PluginModule.__init__(self, redis_conn, db, **kw)
        if not self.standalone:
            self.start_listen_bus()
            self.save_info()

    def init_db (self, args):
        logging.debug('Connecting to SQL {} ...'.format(args.dbhost))
        db = SQLDatabase(args)
        if db.connection():
            logging.debug('SQL {} connected ...'.format(args.dbhost))
            return db
        else:
            logging.error('Unable to connect to SQL server')
            exit(1)

    def _encoding (self, msg):
        msg_bytes = msg.encode('ascii')
        b64_bytes = base64.b64encode(msg_bytes)
        return b64_bytes.decode('ascii')

    def _decoding (self, b64_str):
        base64_bytes = b64_str.encode('ascii')
        msg_bytes = base64.b85decode(base64_bytes)
        return msg_bytes.decode('ascii')

    def _publish (self, res, mqtt_msg, interval=30):
        if self.MQTT:
            self.tt.publish(mqtt_msg)
        _query = """UPDATE mqtt_device SET last_update =  %s WHERE device_id = %s"""
        _val = (res['time'], res['cam_id'])
        self.dbDict['camera-{}'.format(res['cam_id'])] = {'query': _query, 'value': _val}
        #print (_query, _val)
        _now = dt.datetime.now()
        _timediff = _now - self.timing
        if _timediff.total_seconds() > interval:
            for key, val in self.dbDict.items():
                logging.debug('Update Database for {}: {}'.format(key, val['value']))
                cur = self.db.execute(val['query'], data=val['value'], commit=False)
            self.db.commit()
            self.timing = _now

    # example msg:  {'id': 'camera-18', 'msg': '{"cam_id": 18, "loc_x": 97, "loc_y": 881, "time": "2022-12-15 12:30:23", "human_id": 999990, "microsecond": 193836}'}
    def process_redis_msg_old (self, ch, msg):
        if ch in self.subscribe_channels:
            # logging.debug("{}: redis-msg received from '{}': {}".format(self, ch, msg))
            res = msg.get('msg', None)
            #print ('*********')
            #print (res)
            if not res is None:
                _res = json.loads(res)
                if 'cam_id' in _res:
                    # _res['comfort'] = 0
                    # if DEBUG:
                    #     bCAM = copy.deepcopy(_res['cam_id'])
                    #     _res['cam_id'] = 7103
                    enc = hashlib.blake2b(key=str(_res['human_id']).encode(), digest_size=7).hexdigest()
                    if 'human_comfort' in _res:
                        _name = "human_x_y_comfort"
                        _val = '{}_{}_{}_{}'.format(enc, _res['loc_x'], _res['loc_y'], _res['human_comfort'])
                    else:
                        _name = "human_x_y_comfort"
                        _val = '{}_{}_{}_{}'.format(enc, _res['loc_x'], _res['loc_y'], 0)

                    mqtt_msg = {
                        "PC_ID": '{0:06d}'.format(msg['pcid']),
                        "Device_data": [
                            {
                                "Device_ID": '{0:06d}'.format(_res['cam_id']),
                                "Data_name": _name,
                                "Data_type": "string",
                                "Value": _val,
                            }
                        ],
                        "Timestamp":int(_res['time']),
                    }
                    self.tt.publish(mqtt_msg)

                if 'mic_id' in _res:
                    volume_level = {
                        "PC_ID": '{0:06d}'.format(msg['pcid']),
                        "Device_data": [
                            {
                                "Device_ID": '{0:06d}'.format(_res['mic_id']),
                                "Data_name": 'volume level',
                                "Data_type": "integer",
                                "Value": int(_res['audio_level']),
                            }
                        ],
                        "Timestamp":int(_res['time']),
                    }

                    self.tt.publish(volume_level)

                    emotion = {
                        "PC_ID": '{0:06d}'.format(msg['pcid']),
                        "Device_data": [
                            {
                                "Device_ID": '{0:06d}'.format(_res['mic_id']),
                                "Data_name": 'emotion type',
                                "Data_type": "integer",
                                "Value": int(_res['status']),
                            }
                        ],
                        "Timestamp":int(_res['time']),
                    }
                    self.tt.publish(emotion)

                # if DEBUG:
                #     _res['cam_id'] = bCAM
                # self._publish(_res, mqtt_msg)
    
    def process_redis_msg (self, ch, msg):
        if ch in self.subscribe_channels:
            if ch == 'sql.changes.listener':
                self.updates_result(msg)
            if ch == 'sql.humanChanges.listener':
                self.updates_human(msg)
    
    def updates_human (self, msg):
        _hm = {
            "PC_ID": msg.get('pcid', '7000'),
            "Human_data": msg.get('fvList', []),
            "TimeStamp": int(msg.get('timestamp', dt.datetime.now()).timestamp()),
        }
        self.tt.publish(_hm)
    
    def updates_result (self, msg):
        #logging.debug("{}: redis-msg received from '{}': {}".format(self, ch, msg))
        res = msg.get('msg', None)
        #print ('*********')
        #print (res)
        if not res is None:
            _res = json.loads(res)
            if 'util_id' in _res:
                print ('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                print (_res['time'])
                mqtt_msg = {
                    "PC_ID": '{0:06d}'.format(msg['pcid']),
                    "Device_data": [
                        {
                            "Device_ID": '{0:06d}'.format(_res['util_id']),
                            "Device_name": 'deviceID_rate',
                            "Data_type": "string",
                            "Value": '{}_{}'.format(_res['util_id'], _res['util_rate']),
                        }
                    ],
                    "Timestamp": int(_res['time']),
                }
                self.tt.publish(mqtt_msg)
            
            if 'cam_id' in _res:
                # _res['comfort'] = 0
                # if DEBUG:
                #     bCAM = copy.deepcopy(_res['cam_id'])
                #     _res['cam_id'] = 7103
                shid = str(_res['human_id'])
                enc = hashlib.blake2b(key=shid.encode(), digest_size=7).hexdigest()
                if 'Unk' in shid or 'unk' in shid or shid.startswith('9999'):
                    enc = 'UNK-{}'.format(enc[4:])
                #print ('************ hid: {}, encode: {}'.format(shid, enc))
                if 'human_comfort' in _res:
                    _name = "human_x_y_comfort"
                    _val = '{}_{}_{}_{}'.format(enc, _res['loc_x'], _res['loc_y'], _res['human_comfort'])
                else:
                    _name = "human_x_y_comfort"
                    _val = '{}_{}_{}_{}'.format(enc, _res['loc_x'], _res['loc_y'], 0)
                #print ('*************** {}'.format(_val))
                mqtt_msg = {
                    "PC_ID": '{0:06d}'.format(msg['pcid']),
                    "Device_data": [
                        {
                            "Device_ID": '{0:06d}'.format(_res['cam_id']),
                            "Data_name": _name,
                            "Data_type": "string",
                            "Value": _val,
                        }
                    ],
                    "Timestamp":int(_res['time']),
                }
                self.tt.publish(mqtt_msg)

            if 'mic_id' in _res:
                # accumulate data to send every 5s, 
                # need to inform PRDCV change of format to 
                _mic = {
                    "PC_ID": '{0:06d}'.format(msg['pcid']),
                    "Device_data": [
                        {
                            "Device_ID": '{0:06d}'.format(_res['mic_id']),
                            "Data_name": 'volume level',
                            "Data_type": "integer",
                            "Value": int(_res['audio_level']),
                        },
                        {
                            "Device_ID": '{0:06d}'.format(_res['mic_id']),
                            "Data_name": 'emotion type',
                            "Data_type": "integer",
                            "Value": int(_res['status']),
                        }
                    ],
                    "Timestamp":int(_res['time']),
                }
                if _mic["PC_ID"] in self.micData['dataDict']:
                    self.micData['dataDict'][_mic['PC_ID']]["Device_data"].extend(_mic['Device_data'])
                    self.micData['dataDict'][_mic['PC_ID']]['Timestamp'] = _mic['Timestamp']
                else:
                    self.micData['dataDict'][_mic['PC_ID']] = _mic

                # send message every 5s
                _now = dt.datetime.now()
                _timediff = _now - self.micData.get('time', _now)
                if _timediff.total_seconds() > 5:
                    for key in self.micData['dataDict']:
                        self.tt.publish(self.micData['dataDict'][key])
                    #self.tt.publish(self.micData['dataDict'])
                    self.micData['time'] = _now
                    self.micData['dataDict'] = {}
            # if DEBUG:
            #     _res['cam_id'] = bCAM
            # self._publish(_res, mqtt_msg)
            '''
            MQTT_MSG = {
                "PC_ID": '007000',                                                                                  ### PC ID
                "Device_data": [                                                                                    ### consolidate all microphone data within same PC_ID into
                    {"Device_ID": '007201', "Data_name": 'volume level', "Data_type": "integer", "Value": 56},      ###     a single list and send out every 5s
                    {"Device_ID": '007202', "Data_name": 'volume level', "Data_type": "integer", "Value": 80},
                    {"Device_ID": '007201', "Data_name": 'emotion type', "Data_type": "integer", "Value": 0},
                    {"Device_ID": '007202', "Data_name": 'emotion type', "Data_type": "integer", "Value": 2},
                ],
                "Timestamp": 1697734421,                                                                            ### record lastest microphne data's timestamp
            }
            '''

    def close (self):
        if self.MQTT:
            self.tt.close()
        PluginModule.close(self)

def load_processing_module (*a, **kw):
    return MQTTForwarding(*a, **kw)

if __name__ == "__main__":
    raise Exception('This module must start with backend server')

    
