#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import pathlib
import time
import ssl
import json
import threading

import datetime as dt

try:
    from paho.mqtt import client as mqtt_client
except:
    print ('Error import library')
    pass

class MQTTBroker(object):
    def __init__ (self, args) -> None:
        self.broker = args.broker
        self.port = args.port
        self.topic = args.topic
        self.is_connected = False
        logging.debug('Connecting to MQTT broker {}:{} ...'.format(self.broker, self.port))
        self._connection()
        self._start()

    def _connection (self):
        self.client = mqtt_client.Client('MQTTForwarding')
        self.context = ssl.SSLContext(ssl.PROTOCOL_TLS)
        self.context.check_hostname = False
        self.client.tls_set_context(self.context)
        self.client.on_connect = self._on_mqtt_connect
        self.client.connect(self.broker, self.port)
    
    def _on_mqtt_connect (self, client, userdata, flags, rc):
        if rc == 0:
            logging.debug('Connected to MQTT Broker {}:{}'.format(self.broker, self.port))
            self.is_connected = True
        else:
            logging.error('Reconnect to MQTT Broker {}:{}'.format(self.broker, self.port))
            self.is_connected = False
            self._connection()
    
    def _start (self):
        self.client.loop_start()
    
    def connection (self):
        return self.is_connected
        
    def get_msg (self, pcID, dataList, dataTime, tag):
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

    def publish (self, msg):
        self.client.publish(self.topic, msg, qos=0, retain=True)

class MQTTForwardingBridge(object):
    def __init__ (self, db, args, **kw) -> None:
        self.hostAddr = {'ip': args.broker, 'port': args.port}
        self.topic = args.topic
        scriptpah = pathlib.Path(__file__).parent.resolve()
        self.caPath = scriptpah.parent / args.cert

        self.tt = MQTTBroker(args=args)
        self.db = db
        import redis
        pool = redis.ConnectionPool(
        host = '10.13.3.57',
        port = 6379,
        password = 'ew@icSG23', 
        )
        self.redis_conn = redis.Redis(connection_pool=pool)
        # self.redis_conn = args.redis_conn
    
    def start (self):
        if not self.db.connection():
            logging.error('SQL Database not connected')
            exit(1)
        if not self.tt.connection():
            logging.error('MQTT Broker not connected')
            exit(1)
        
        # init looping and data for MQTT & SQL
        _query = "UPDATE mqtt_device SET last_update =  CAST( '" + str(dt.datetime.now()) + "' AS DATETIME)"
        self.db.execute(_query, commit=True)

        self.th_quit = threading.Event()
        self.th = threading.Thread(target=self.loop)
        self.th.start()
    
    def _publish (self, msg, cTime, id):
        logging.debug('Publish to MQTT: {}'.format(msg))
        self.tt.publish(msg)
        _query = "UPDATE mqtt_device SET last_update =  CAST( '" + str(dt.datetime.fromtimestamp(cTime)) + "' AS DATETIME) WHERE device_id ="+ str(id)
        logging.debug('Update SQL after publish: {}'.format(_query))
        #self.db.execute(_query, commit=True)

    def loop (self):
        while self.db.connection():
            _query = 'select * from mqtt_device where active = 1'
            cur = self.db.query(_query)  

            for c in cur:
                now = dt.datetime.now()
                diff = int((now - c[4]).total_seconds())
                if diff >= c[5]:
                    if c[2] == 0:
                        self._camera(c)
                    elif c[2] == 1:
                        self._audio(c)
            
            if self.th_quit.is_set(): break
    
    def stop (self):
        self.th_quit.set()

    def is_quit (self):
        if self.tt.connection() and self.db.connection():
            return False
        return True
    
    def _camera (self, data):
        #_query = "select cam_id,loc_x,loc_y,time,human_id from location_table where cam_id=" + str(data[3]) +" and time >= CAST( '" + str(data[4]) + "' AS DATETIME)"
        _query = "select cam_id,loc_x,loc_y,time,human_id from location_table where cam_id=" + str(data[3]) +" and time >= CAST( '2022-08-11 17:06:38' AS DATETIME)"
        cur = self.db.query(_query)
        logging.debug('Camera query: {}'.format(_query))

        dataList = []
        dTimestamp = None
        logging.debug('Camera query with ID {}, Cast: {}. Result length: {}'.format(data[3], data[4], len(cur)))

        if len(cur) > 0:
            for c in cur:
                dataList.append({
                    "Device_ID": ""+str(c[0])+"",
                    "Data_name": "human_x_y_comfort",
                    "Data_type": "string",
                    "Value": "{}_{}_{}".format(str(c[4]), str(c[1]), str(c[2]))
                })
                dTimestamp = c[3]
            
            msg, currTime = self.tt.get_msg(str(data[0]), dataList, dTimestamp, 'camera')
            self._publish(msg, currTime, data[3])

    def _audio (self, data):
        _query = "select * from emotion_table , audio_location_table where emotion_table.mic_id=" + str(data[3]) + " and emotion_table.mic_id=audio_location_table.mic_id and datetime >= CAST( '" + str(data[4]) + "' AS DATETIME)"
        cur = self.db.query(_query)
        logging.debug('Audio query: {}'.format(_query))

        dataList = []
        dTimestamp = None
        logging.debug('Audio query with ID {}, Cast: {}. Result length: {}'.format(data[3], data[4], len(cur)))

        if len(cur) > 0:
            for c in cur:
                dataList.append({
                    "Device_ID": str(c[1]),
                    "Data_name": "angle_Vol_emo",
                    "Data_type": "float",
                    "Value": "{}_{}_{}".format(str(c[6]), str(c[8]), str(c[2]))
                })
                dTimestamp = c[4]
            
            msg, currTime = self.tt.get_msg(str(data[0]), dataList, dTimestamp, 'audio')
            self._publish(msg, currTime, data[3]) 

def load_processing_module (*a, **kw):
    return MQTTForwardingBridge(*a, **kw)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MQTT library...')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('-b', '--broker', type=str, help='MQTT broker IP address (default: 54.84.97.142)', default='10.13.3.251')
    parser.add_argument('-p', '--port', type=int, help='port number for MQTT server (default: 8883)', default=8883)
    parser.add_argument('-du', '--dbuser', type=str, help='SQL username (default: root)', default='root')
    parser.add_argument('-dp', '--dbpwd', type=str, help='SQL password (default: Welcome123)', default='ew@icSG23')
    parser.add_argument('-dh', '--dbhost', type=str, help='SQL host address (default: localhost)', default='localhost')
    parser.add_argument('-dt', '--dbtbl', type=str, help='SQL table name (default: mockup_db)', default='kajima_db')
    parser.add_argument('-ca', '--cert', type=str, help='CA certificate directory (default: ./cert)', default='cert')
    parser.add_argument('-tp', '--topic', type=str, help='Topic for MQTT server (default: GEAR/L3/Panasonic/Device/007000)', default='GEAR/L3/Panasonic/Device/007000')
    parser.add_argument('-l', '--logfile', type=str, help='Specify log file (Default: None)', default=None)

    args = parser.parse_args(sys.argv[1:])

    logfile = 'none' if args.logfile is None else args.logfile
    if logfile != 'none':
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s %(module)s] %(levelname)s: %(message)s',
            filename=logfile,
            filemode='w'
        )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if args.debug else logging.INFO)
    console.setFormatter(logging.Formatter('%(module)-12s: %(levelname)-8s: %(message)s'))
    logging.getLogger().addHandler(console)
    logging.debug(u"args: {}".format(args))

    scriptpath = pathlib.Path(__file__).parent.resolve()
    sys.path.append(str(scriptpath.parent / 'common'))

    from sqlwrapper import SQLDatabase

    sql = SQLDatabase(args=args)
    _bridge = MQTTForwardingBridge(sql, args)
    _bridge.start()
    try:
        while not _bridge.is_quit():
            pass
    except KeyboardInterrupt:
        logging.info("Ctrl-C received -- terminating ...")
        _bridge.stop()

    
