#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
jsonutils.py
Some JSON encoding/decoding utilities
basically provide 2 functions:
    json2str() and str2json()
which replaces json.dumps() and json.loads()
with ability to handle datetime and bson.ObjectId fields

We have two sets of different implementations:
First one uses JSONEncoder/JSONDecoder
Second one use convert_jsondict() to convert dictionary

Currently, the second simplementation is used.
To change, see the documentation around line 130.

See the bottom testing for examples 
'''

import json
import datetime as dt
from json.encoder import JSONEncoder
from dateutil import parser as dtparser
from bson import ObjectId


class JSONDatetimeEncoder(json.JSONEncoder):
    FORMAT = "%Y-%m-%dT%H:%M:%S.%f"
    def __init__ (self, *args, **kwargs):
        self.FORMAT = kwargs.pop('datetime_format', self.FORMAT)
        JSONEncoder.__init__(self, *args, **kwargs)
    def default(self, obj):
        if isinstance(obj, dt.datetime):
            if self.FORMAT == '$dt':
                return { '$dt': obj.timestamp() }
            return obj.strftime(self.FORMAT)
        if isinstance(obj, ObjectId):
            return '$@:' + str(obj)
        return json.JSONEncoder.default(self, obj)

class JSONDatetimeDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        self.dt_field = kwargs.pop('datetime_field', 'timestamp')
        if not isinstance(self.dt_field, (list, tuple, set)):
            self.dt_field = [ self.dt_field ]
        json.JSONDecoder.__init__(self, object_pairs_hook=self.object_pairs_hook, *args, **kwargs)

    def object_pairs_hook(self, pairs):
        ret = {}
        for k,v in pairs:
            try:
                if k in self.dt_field and isinstance(v, str) and '{' not in v and '}' not in v:
                    # it is a datetime field name, and value is a string
                    ret[k] = dtparser.parse(v)
                    continue
                elif isinstance(v, str) and v.startswith('$@:'):
                    ret[k] = ObjectId(v[3:])
                    continue
                elif isinstance(v, dict) and '$dt' in v:
                    if isinstance(v['$dt'], str):
                        ret[k] = dtparser.parse(v['$dt'])
                    else:
                        ret[k] = dt.datetime.fromtimestamp(v['$dt'])
                    continue
            except:
                pass
            ret[k] = v
        return ret

def convert_jsondict (d, fieldmap={}, typemap={}, allmap=[]):
    ''' convenient routine that parse through a dict
        any key that appears in fieldmap will be subjected to the callable
        any value of type that appears in typemap will be subjected to the callable
        all key-value pair will be subjected to the callables in allmap
    '''
    if isinstance(d, dict):
        ret = {}
        for k,v in d.items():
            if k in fieldmap:
                k, v = fieldmap[k](k, v)
                if k is None:
                    continue
            for x in allmap:
                k,v = x(k,v)
            ret[k] = convert_jsondict(v, fieldmap, typemap, allmap)
        return ret
    if isinstance(d, (list, tuple)):
        return [ convert_jsondict(x, fieldmap, typemap, allmap) for x in d ]
    if type(d) in typemap:
        return typemap[type(d)](d)
    return d

# what we almost always use
def dt2json(x): 
    return { '$dt': x.timestamp() }

def id2json(x): 
    return '$@:' + str(x)

def convert_d2js(d): 
    return convert_jsondict(d, typemap = { dt.datetime: dt2json, ObjectId: id2json })

def json2dt (v):
    if isinstance(v, dict) and '$dt' in v:
        try:
            if isinstance(v['$dt'], str):
                return dtparser.parse(v['$dt'])
            elif isinstance(v['$dt'], (int,float)):
                return dt.datetime.fromtimestamp(v['$dt'])
        except:
            pass
    return v

def json2dt_kv (k,v):
    return k, json2dt(v)

def json2id (v):
    if isinstance(v, str) and v.startswith('$@:'):
        try:
            return ObjectId(v[3:])
        except:
            return v
    return v

def json2id_kv (k,v):
    return k, json2id(v)

def convert_js2d(d): 
    return convert_jsondict(d, allmap = [ json2dt_kv, json2id_kv ])

### UNCOMMENT THE BELOW TWO LINES TO USE FIRST IMPLEMENTATION
#json2str = lambda *a,**kw: json.dumps(*a, cls=JSONDatetimeEncoder, **kw)
#str2json = lambda *a,**kw: json.loads(*a, cls=JSONDatetimeDecoder, **kw)

### COMMENT THE 14 LINES BELOW TO USE FIRST IMPLEMENTATION
import logging
def json2str(d, **kw):
    try:
        return json.dumps(convert_d2js(d), **kw)
    except:
        logging.exception("d={}".format(d))
        return {}

def str2json(s, **kw):
    try:
        return convert_js2d(json.loads(s, **kw))
    except:
        logging.exception("d={}".format(d))
        return {}

# ----------------------------------------
# convert a dictionary to one that is more amiable to printing
# by shortening long strings and skipping elements in array
# ----------------------------------------
def print_json(d, **kw):
    '''
    d is the dictionary
    accepted keywords are:
        'strmax': minimum length of string that will be shortened (default: 50)
        'listmax': minimum length of list that will be shortened (default: 7)
        'showlen': whether to show the original length of the shorten string/list (default: True)
        'sep': separator between dictionary fields (default: ', ')
    '''
    _slen = kw.get('strmax', 50)
    _alen = kw.get('listmax', 7)
    _showlen = kw.get('showlen', True)
    _sep = kw.get('sep', ', ')
    if isinstance(d, dict):
        r = []
        for k in d:
            if isinstance(d[k], (dict, list, str)):
                r.append(u"'" + k + u"': " + print_json(d[k], **kw))
            else:
                r.append(u"'" + k + u"':{}".format(d[k]))
        return u"{" + _sep.join(r) + u"}"
    if isinstance(d, list):
        if len(d) > _alen:
            s = u"[" + u', '.join(print_json(x, **kw) for x in d[:_alen-3])
            s += u' ... {a}]'.format(a=print_json(d[-1], **kw))
            if _showlen:
                s += '<{b}>'.format(b=len(d))
            return s
        return u"[{}]".format(_sep.join(print_json(x, **kw) for x in d))
    if isinstance(d, str):
        if len(d) > _slen:
            s = u"'" + d[:_slen-15] + u'...' + d[-5:]
            s += u"'<{}>".format(len(d)) if _showlen else "'"
            return s.replace('\n', '\\n')
        return u"'{}'".format(d)
    return str(d)


if __name__ == "__main__":
    a = {
        'timestamp': dt.datetime.now(),
        'db-id': ObjectId(),
        'field-1': {
            "name": "hello", "age": 24,
            "birthdate": dt.datetime(2000,1,1,12,12,12),
            "timestamp": "hello world",
            "list": [ 100, dt.datetime(2020,1,1), ObjectId() ],
        }
    }
    print("a = {}\n".format(a))
    sa = json2str(a)
    print("sa = json.dumps(a) = '{}'".format(sa))
    aa = str2json(sa)
    print("json.loads(sa) = {}".format(aa))

    print("\n\n")

    a2js = convert_d2js(a)
    print("convert_js2d(a) = {}\n".format(a2js))
    print("convert_d2js(a2js) = {}\n".format(convert_js2d(a2js)))