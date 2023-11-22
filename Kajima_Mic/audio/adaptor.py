#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Base adaptor class definition
To be imported by all adaptors.

Two adaptors class are defined here.
Adaptor() is normal adaptor which do not have REST interface.
WebAdaptor() is enhanced to be used with REST server (such as flask)
'''

import redis
import logging
import sys
import pathlib
import time
import datetime as dt
import threading
from queue import Queue, Empty
#logging.basicConfig()

scriptpath = pathlib.Path(__file__).parent.resolve()
if (scriptpath.parent / 'common').exists():
    sys.path.append(str(scriptpath.parent / 'common'))
elif (scriptpath / 'common').exists():
    sys.path.append(str(scriptpath / 'common'))
from sispcomp import SISPComponentBase
from jsonutils import json2str, str2json
from argsutils import connect_redis_with_args
from miscutils import get_my_ip, get_all_ip

class Adaptor(SISPComponentBase):
    '''
    base apdator class
    '''
    component_type = 'audio'
    info_frequency = 10
    subscribe_channels = []
    def __init__ (self, args, **kw):
        self.adaptor_type = args.type
        self.component_name = args.id
        self.site = args.site
        self.location = args.location 
        self.pcid = args.pcid
        self.status_period = dt.timedelta(seconds=args.status_period)
        self.last_publish = dt.datetime.now() - self.status_period
        SISPComponentBase.__init__(self, args, **kw)
        
    def start_listen_bus (self):
        ''' start to listen to event-bus.  
        We do not put this in the constructor in case child classes needs to
        perform other initialization 
        '''
        SISPComponentBase.start_listen_bus(self)
        self.start_thread('periodic', self.periodic_publish)

    def periodic_publish(self):
        ''' periodically publish our status so that IME knows we are still alive '''
        ch = "{}.status".format(self.component_prefix)
        info_n = self.info_frequency    # we also periodically save our info in redis
        while True:
            # check whether we should save our info in redis.  We save it after every 10 status publish
            info_n += 1
            if info_n >= self.info_frequency:
                try:
                    self.save_info()
                    self.broadcast_redis_change(change='info')
                except:
                    logging.error('Failed to save info and broadcast changes')
                    pass
                info_n = 0
            # publish our status -- some one else might published it for us already recently
            if self.last_publish + self.status_period <= dt.datetime.now():
                self.publish_status(ch)
            # trigger other periodic tasks
            if hasattr(self, 'periodic_task'):
                self.periodic_task()
            if self.is_quit((self.last_publish+self.status_period-dt.datetime.now()).total_seconds()):
                break
        logging.debug("{}: status-publish thread terminated".format(self))

    def publish_status (self, ch=None, status=None):
        ''' all adaptor need to publish its status '''
        if status is None:
            status = self.get_status()
        if not ch: ch = "{}.status".format(self.component_prefix)
        self.last_publish = dt.datetime.now()
        try:
            #print ('**************************')
            #print (status)
            if not status is None:
                self.redis_conn.publish(ch, json2str(status))
        except:
            logging.error('Failed to publish message {}'.format(ch))
            pass
        logging.debug("{}: ch='{}' <= msg='{}'".format(self, ch, status))

    def get_status (self):
        ''' this is a standard get_status() method.  Child adaptors should override this method '''
        return {
            "camera": self.component_name,
            "type": self.adaptor_type,
            "timestamp": dt.datetime.now(),
            "status": "normal",
            "site": self.site,
            "location": self.location,
            "pc-id": self.pcid,
        }
    
    def _format_list_ (self, fmt):
        ret = SISPComponentBase._format_list_(self, fmt)
        if 'l' in fmt or fmt == 'all':
            ret.append('loc={}'.format(self.location))
        if 's' in fmt or fmt == 'all':
            ret.append('site={}'.format(self.site))
        return ret

    def get_info (self):
        ''' return a dict containing information of this adaptor '''
        ret = SISPComponentBase.get_info(self)
        ret.update({
            'camera': self.component_name,
            'camera-type': self.adaptor_type,
            'site': self.site,
            'location': self.location,
            "status-period": self.status_period.total_seconds(),
        })
        return ret

    def get_redismsg_by_channel (self, ch):
        import json
        cfg = self.redis_conn.get(ch)
        return None if cfg is None else json.loads(cfg) 

    def load_config (self, args):
        ''' load configuration either from redis or from configfile '''
        import ast
        cfg = self.redis_conn.get("{}.config".format(self.component_prefix))
        if not cfg:
            if pathlib.Path(args.config).is_file():
                logging.debug("{}: loading from config file '{}'".format(self, args.config))
                with open(args.config, 'rt') as f:
                    cfg = ast.literal_eval(f.read())
        else:
            import pprint
            cfg = ast.literal_eval(cfg)
            logging.debug("{}: saving redis config to file '{}'".format(self, args.config))
            with open(args.config, 'wt') as f:
                f.write(pprint.pformat(cfg))
        return cfg
    
    def publish_redis_msg (self, ch, msg):
        self.redis_conn.publish(ch, json2str(msg))

    def process_redis_msg (self, ch, msg):
        ''' process a message returned from redis event bus '''
        logging.debug("{}: redis-msg received from '{}': {}".format(self, ch, msg))


class WebAdaptor(Adaptor):
    '''
    base apdator class with web interface.
    Note that implementatuion still need to do their own flask configuration.
    This class simply adds some common functionality (mainly http-key for authentication)
    to support a web interface.
    '''
    def __init__ (self, args, **kw):
        # set up redis connection and other parameters
        self.http_port = args.http_port
        self.http_keys = None
        self.http_keys_expiry = 0
        Adaptor.__init__(self, args, **kw)

    def get_info (self):
        ''' return a dict containing information of this adaptor '''
        r = Adaptor.get_info(self)
        if self.http_keys is None:
            self.get_http_keys()
        r['http-port'] = self.http_port
        r['http-key'] = self.http_keys[0]
        return r

    def get_gateway_info (self):
        ''' retrieve the api-gateway info saved in redis '''
        info = self.redis_conn.get("api-gateway.info")
        return str2json(info) if info else {}

    def get_http_keys (self):
        ''' retrieve the http key (either from api-gateway or self-generated) '''
        info = self.get_gateway_info()
        self.http_keys = []
        if self.component_name in info:
            self.http_keys.append(info[self.component_name]['http-key'])
        if 'api-gateway' in info:
            self.http_keys.append(info['api-gateway']['http-key'])
        if not self.http_keys:
            import uuid
            self.http_keys = [ uuid.uuid4().hex ]
        self.http_keys_expiry = 10

    def is_allowed_http (self, key=None, flask=None, req={}):
        ''' check a http key and return True if it is allowed '''
        if not self.http_keys:
            self.get_http_keys()
        if key is None:
            key = req.get('key', None)
            if flask:
                key = flask.request.args.get('key', key)
        if key not in self.http_keys:
            logging.debug("{}: invalid-key: {} != {}".format(self, key, self.http_keys))
            return False
        return True

    def periodic_task (self):
        ''' periodically refreshes http-keys '''
        self.http_keys_expiry -= 1
        if self.http_keys_expiry < 0:
            self.get_http_keys()


from argsutils import add_arg, add_redis_args

def add_common_adaptor_args(parser, **kw):
    ''' convenient routine to add standard argparse arguments for adaptors '''
    add_redis_args(parser)
    g = parser.add_argument_group("adaptor configuration parameters")
    add_arg(g, "--type", h="the type of adaptor {D}", d=kw.get('type', 'adaptor'), m='TYPE')
    add_arg(g, "--id", h="the ID of the adaptor  {D}", d=kw.get('id', 'adaptor-1'), m='ID')
    add_arg(g, "--pcid", h="the ID of PC {D}", d=kw.get('pcid', 7000), m='PCID')
    add_arg(g, "--location", h="location of the adaptor  {D}", d=kw.get('location', 'Kajima'), m='LOC')
    add_arg(g, "--site", h="site where the adaptor is deployed  {D}", d=kw.get('site', 'Kajima-Building'), m='SITE')
    add_arg(g, "--status-period", h="interval (seconds) to send status  {D}", t=int, d=kw.get('status_period', 300), m='SEC')
    add_arg(g, "--config", t=str, h="configuration file for adaptor  {D}", m='FILE', d=str(scriptpath / 'config/{}.json'.format(kw.get('id', 'camera').lower())))
    add_arg(g, "--testmode", a=True, h="run in test mode (some module not supported) -- default: False")
    add_arg(g, "--sock", h="Socket Number for audio application", d=kw.get('sock', 8000), m='SOCK')
    return g
