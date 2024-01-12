#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
plugin_module.py
Base class for plugin module
'''
import logging
import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()
from sispcomp import SISPComponentBase
from jsonutils import json2str

class PluginModule(SISPComponentBase):
    ''' base class for plugin module '''
    component_type = 'module'
    component_name = 'base-module'
    subscribe_channels = ['camera.*.status']

    def __init__(self, redis_conn, db, **kw):
        self.site = kw.get('site', 'Kajima')
        self.mongodb = db
        self.standalone = kw.get('standalone', False)
        SISPComponentBase.__init__(self, redis_conn=redis_conn, **kw)
        self._quit_ch = ['BSP.quit']
    
    def get_info (self):
        ''' return a dict containing description of this module '''
        ret = SISPComponentBase.get_info(self)
        ret.update({
            'module-name': self.component_name,
            'site': self.site
        })
        return ret
    
    def housekeep (self):
        ''' housekeeping '''
        self.save_info()
        
    def broadcast_db_change (self, coll, **details):
        details['source'] = self.component_name
        details['collection'] = coll
        self.redis_conn.publish("mongodb.change.{}".format(coll), json2str(details))
    
    def publish_msg (self, id, msg):
        self.redis_conn.publish(id, json2str(msg))