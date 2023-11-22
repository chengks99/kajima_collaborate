#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Base SISP Components
'''

import logging
import sys
import pathlib
import datetime as dt
import threading
from queue import Queue, Empty

scriptpath = pathlib.Path(__file__).parent.resolve()
if (scriptpath.parent / 'common').exists():
    sys.path.append(str(scriptpath.parent / 'common'))
elif (scriptpath / 'common').exists():
    sys.path.append(str(scriptpath / 'common'))
from jsonutils import json2str, str2json
from argsutils import connect_redis_with_args
from miscutils import get_all_ip, get_my_ip

class SISPComponentBase (object):
    '''  Base SISP Component.  
        All other major components in SISP will be inherited
        from this.  It implements basic functionality
        such as redis connection, thread management
    '''
    component_type = 'base'     # type of this component
    component_name = 'base'     # identifier of this component
    subscribe_channels = []     # what redis channel this component will subscribe to
    ### NOTE: 'component_type' and 'component_name' will become the namespace to use in Redis
    ### for example, this component's configuration parameter will be read from the redis varaible
    ### '<type>.<name>.config', and the component will save its information in the variable
    ### '<type>.<name>.info'
    ### Child instances can override this by defining 'component_prefix' before calling super().__init__()

    def __init__ (self, args=None, **kw):
        #
        if not hasattr(self, 'component_prefix'):
            self.component_prefix = '{}.{}'.format(self.component_type, self.component_name)
        # get our ip address
        self.my_ip = get_my_ip()
        # redis connection handle
        self.redis_conn, self.pubsub = kw.pop('redis_conn', None), None
        if not self.redis_conn and args:
            self.redis_conn = connect_redis_with_args(args)
        # threading support
        self._quit, self._threads = Queue(), {}
        self._quit_ch = [ '{}.quit'.format(self.component_prefix) ]

    def start_thread (self, name, target, **kw):
        ''' start a thread with the given name and target function '''
        if name in self._threads:
            logging.error("{}: start_thread() with same name '{}'!".format(self, name))
        self._threads[name] = threading.Thread(target=target, **kw)
        self._threads[name].start()

    def start_listen_bus (self):
        ''' start to listen to event-bus.  
        We do not put this in the constructor in case child classes needs to
        perform other initialization 
        '''
        self.start_thread('event-bus', self.listen_event_bus)

    def __str__ (self):
        ''' return a string description of this component '''
        return "<{}>".format(self.component_name)

    def _format_list_ (self, fmt):
        ''' handle format specifiers used when with '{:xyz}'  '''
        ret = []
        if 't' in fmt or fmt == 'all':
            ret.append('type={}'.format(self.component_type))
        if 'n' in fmt or fmt == 'all':
            ret.append('name={}'.format(self.component_name))
        if 'T' in fmt or fmt == 'all':
            ret.append('threads={}'.format(','.join(x for x in self._threads)))
        return ret

    def __format__ (self, fmt=''):
        ''' return formatted description of this instance '''
        if not fmt:
            return self.__str__()
        ret = self._format_list_(fmt)
        if '+' in fmt or fmt == 'all':
            return '<{}>:\n\t{}\n'.format(self.__str__(), ';\n\t'.join(ret))
        return '<{}>'.format('; '.join(ret))

    def get_info (self):
        ''' return a dict containing information of this component 
            The dict is typically stored in redis as a set variable (see store_info())
            Derived class should override this to return a more complete info dict
        '''
        return {
            'component': self.component_name,
            "ip-address": get_all_ip(),
            "threads": [ x for x in self._threads if self._threads[x].is_alive() ],
            'update-time': dt.datetime.now(),
            "listening": self.subscribe_channels,
        }

    def save_info (self):
        ''' save our information to redis
            This will use get_info() to obtain the dict to be stored in redis 
        '''
        self.redis_conn.set("{}.info".format(self.component_prefix), 
            json2str(self.get_info())
        )

    def broadcast_redis_change (self, ch=None, **details):
        ''' inform others there is some changes to the redis variables '''
        if not ch: ch = self.component_prefix
        if 'source' not in details: details['source'] = self.component_name
        self.redis_conn.publish("redis.change.{}".format(ch), json2str(details))
 
        
    def listen_event_bus (self):
        import redis
        ''' thread for listening to subscribed Redis channels '''
        logging.debug("{}: listening to event bus [{}] ...".format(self, self.subscribe_channels))
        self.pubsub = self.redis_conn.pubsub()
        self.pubsub.psubscribe(*self.subscribe_channels, *self._quit_ch)
        while not self.is_quit():
            try:
                # logging.debug(self.pubsub.listen(), self.subscribe_channels)
                # import time
                # starttime = time.time()
                for msg in self.pubsub.listen():
                    # import fnmatch
                    # if fnmatch.fnmatch( msg['channel'], 'camera.*.result'):  
                    #     try:
                    #         logging.info(len(str2json(msg['data'])['result']))
                    #         logging.info(str2json(msg['data'])['result'][0])
                    #         logging.info(msg['channel'])
                    #         print("#######{}#######")
                    #     except:
                    #         continue
                    # logging.info(msg)
                    if msg['channel'] in self._quit_ch and msg['data'] == 'QUIT':
                        logging.debug("received 'QUIT' from {}".format(msg['channel']))
                    elif isinstance(msg['data'], str) and msg['data'].startswith('{') and msg['data'].endswith('}'):
                        xmsg = str2json(msg['data'])
                        # logging.debug(msg['channel'], xmsg)
                        self.process_redis_msg(ch = msg['channel'], msg = xmsg)
                        # print(time.time()-starttime)
                        # print("#################")
                    break   # check for quit()
            except redis.exceptions.ConnectionError as e:
                if not self.redis_conn.ping():
                    logging.error('Unable to connect Redis Server')
                logging.debug(msg['channel'])
                # logging.debug("Last message received: {}".format(json2str(msg)))
                logging.error(e)
                # exit(1)
                pass
        logging.debug("{}: stop listening to event bus".format(self))
        exit()

    def process_redis_msg (self, ch, msg):
        ''' process a message returned from redis event bus (virtual) 
            msg will be converted from str to dict using json2str()
        '''
        logging.debug("{}: redis-msg received from '{}': {}".format(self, ch, msg))

    def is_quit (self, timeout=-1):
        ''' check if we are quiting because close() is called
            timeout: how long to wait.  if <=0, non-blocking check 
        '''
        try:
            if self._quit.get(timeout=timeout if timeout>0 else 0.001) == "QUIT":
                self._quit.put("QUIT")  # put the 'QUIT' so that other threads will get it as well
                return True
            return False
        except Empty:
            return False

    def close (self):
        ''' close the component (i.e. destroy) 
            This will send 'QUIT' to all listening threads, and cause is_quit() method to return True. 
        '''
        # signal all thread to terminate
        self._quit.put("QUIT")
        self.redis_conn.publish(self._quit_ch[0], 'QUIT')
        # wait for all threads to complete
        for tn, thr in self._threads.items():
            if thr.is_alive():
                logging.debug("{}: waiting for thread:{} to terminate ...".format(self, tn))
                thr.join(1)
                if thr.is_alive():
                    logging.debug("{}: thread:{} not terminating".format(self, tn))
        self._threads = {}
        # close the redis connection
        self.redis_conn.delete("{}.info".format(self.component_prefix))       # remove the info entry from Redis
        self.redis_conn.close()

