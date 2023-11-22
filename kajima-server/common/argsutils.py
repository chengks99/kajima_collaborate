#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
argparse utilities
'''

import logging
import re

def add_arg(x, *arg, **kw):
    ''' convenient routine to provide shorthands to kwargs of argparse.add_arguments().
    For example, use 'h' for 'help', and the help text can contain '{D}' that will be
    replaced by '-- Default: <default>'.
    Shorthands:
        h: help, t: type,
        a: action -- can use a=True as shorthand to action='store_true', a=False for action='store_false'
        d: default, m: metavar, c: choices, n: nargs, r: required
    '''
    defvals = kw.pop('_defvals', {})
    valname = arg[-1].replace('--','')
    for a in [('h', 'help'), ('t', 'type'), ('d', 'default'), ('m', 'metavar'), 
            ('a', 'action'), ('c','choices'), ('n', 'nargs'), ('r', 'required')]:
        if a[0] in kw: kw[a[1]] = kw.pop(a[0])
    if 'type' not in kw and 'choices' not in kw and 'action' not in kw:
        kw['type'] = type(kw['default']) if 'default' in kw else str
    elif isinstance(kw.get('action', None), bool):
        if 'default' not in kw and '{D}' in kw.get('help', ''):
            kw['help'] = kw['help'].format(D=' -- Default: {}'.format(not kw['action']))
        kw['action'] = 'store_true' if kw['action'] else 'store_false'
    if '{D}' in kw.get('help', ''):
        defv = kw.get('default', defvals.get(valname, None))
        kw['help'] = kw['help'].format(D=' -- Default: {}'.format(defv))
    x.add_argument(*arg, **kw)

def add_redis_args(parser, groupname="Redis configuration parameters", host='localhost', port=6379, passwd=None, db=None):
    ''' add in arguments related to redis '''
    g = parser.add_argument_group(groupname)
    add_arg(g, "--redis-host", t=str, h="hostname/IP of the redis server {D}", d=host, m='HOST')
    add_arg(g, "--redis-port", t=int, h="port number of the redis server {D}", d=port, m='PORT')
    add_arg(g, "--redis-no-decode", a=True, h="do not decode responses for redis -- default: False")
    add_arg(g, "--redis-passwd", t=str, h="password for redis authentication {D}", d=passwd, m='PASSWD')
    add_arg(g, "--redis-db", t=str, h="databse name of redis {D}", d=db, m='DB')
    return g

def connect_redis_with_args(args, return_pool=False):
    ''' connect to redis bus based on the parsed input args '''
    logging.debug("Connecting to redis {}:{} ...".format(args.redis_host, args.redis_port))
    import redis
    pool = redis.ConnectionPool(
        host = args.redis_host,
        port = args.redis_port,
        db = args.redis_db, 
        password = args.redis_passwd, 
        decode_responses = not args.redis_no_decode)
    conn = redis.Redis(connection_pool=pool)
    logging.debug('Redis {}:{} connected ... '.format(args.redis_host, args.redis_port))
    if return_pool:
        return pool, conn
    return conn

def add_mongo_args(parser, groupname="MongoDB configuration parameters", host='localhost', port=27017, user=None, passwd=None, db='mockup_db'):
    ''' add in arguments related to mongo '''
    g = parser.add_argument_group(groupname)
    add_arg(g, "--mongo-host", t=str, h="hostname/IP of the mongo server {D}", d=host, m='HOST')
    add_arg(g, "--mongo-port", t=int, h="port number of the mongo server {D}", d=port, m='PORT')
    add_arg(g, "--mongo-user", t=str, h="username for mongo authentication {D}", d=user, m='USER')
    add_arg(g, "--mongo-passwd", t=str, h="password of the mongo authentication {D}", d=passwd, m='PASSWD')
    add_arg(g, "--mongo-db", t=str, h="databse name to use for Mongo {D}", d=db, m='DB')
    return g

def connect_mongodb_with_args(args, return_client=False):
    ''' convenient routine to connect to mongoDB using args '''
    import pymongo
    uri = 'mongodb://'
    if args.mongo_user:
        uri += '{u}:{p}@' if args.mongo_passwd else '{u}@'
    uri = (uri + '{h}').format(u=args.mongo_user, p=args.mongo_passwd, h=args.mongo_host)
    if args.mongo_port != 27017:
        # non-default port
        uri += ':{}'.format(args.mongo_port)
    logging.debug("connecting to database {}.{} ...".format(uri, args.mongo_db))
    mongoclient = pymongo.MongoClient(uri)
    if return_client:
        return mongoclient, mongoclient[args.mongo_db]
    return mongoclient[args.mongo_db]

def add_sql_args(parser, groupname="SQL configuration parameters", host='localhost', port=8883, user='root', passwd='ew@icSG23', db='kajima_db'):
    g = parser.add_argument_group(groupname)
    add_arg(g, "--sql-host", t=str, h="hostname/IP of SQL server {D}", d=host, m='HOST')
    add_arg(g, "--sql-port", t=int, h="port number of SQL server {D}", d=port, m="PORT")
    add_arg(g, "--sql-user", t=str, h="username for SQL authentication {D}", d=user, m='USER')
    add_arg(g, "--sql-passwd", t=str, h="password of the SQL authentication {D}", d=passwd, m='PASSWD')
    add_arg(g, "--sql-db", t=str, h="databse name to use for SQL {D}", d=db, m='DB')
    return g

def init_parser (description='Smart Intgrated Solution Platform', mongo=None, redis=None, sql=None):
    ''' initializes argument parser with common set of options
        if mongo is not None, add_mongo_args() will be called to add mongo related options
        if redis is not None, add_redis_args() will be called to add redis related options
    '''
    import argparse
    parser = argparse.ArgumentParser(description=description)
    add_arg(parser, '-d', '--debug', h='Turn on debugging output on console', a=True)
    add_arg(parser, '--logfile', t=str, h='Save debugging output to specified logfile', d=None)
    if isinstance(mongo, dict):
        add_mongo_args(parser, **mongo)
    elif mongo is not None:
        add_mongo_args(parser)
    if isinstance(sql, dict):
        add_sql_args(parser, **sql)
    else:
        add_sql_args(parser)
    if isinstance(redis, dict):
        add_redis_args(parser, **redis)
    elif redis is not None:
        add_redis_args(parser)
    return parser

def parse_args(parser, args=None, conn_redis=False, conn_mongo=False,
            log_format='[%(asctime)s %(module)s %(levelname)s]: %(message)s',
):
    ''' call parse_args() and set up logging related functions based on the returned args
        if conn_redis is True, will call connect_redis_with_args() and return as the 2nd value
        if conn_mongo is True, will call connect_mongodb_with_args() and return as the 2nd/3rd value
    '''
    import sys
    if not args: args = sys.argv[1:]
    args = parser.parse_args(args)
    # set up logging
    if hasattr(args, 'logfile') and args.logfile not in [None, 'none']:
        logging.basicConfig(
            level=logging.DEBUG, 
            format=log_format,
            filename=args.logfile,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG if args.debug else logging.INFO)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console)
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO, 
            format=log_format,
        )

    logging.debug(u"args: {}".format(args))
    # if connection to redis/mongo is required
    if conn_redis:
        r = connect_redis_with_args(args)
        if conn_mongo:
            m = connect_mongodb_with_args(args)
            return args, r, m
        return args, r
    if conn_mongo:
        return args, connect_mongodb_with_args(args)
    return args

def to_namespace(dict):
    from argparse import Namespace
    if len(dict.keys()) == 1:
        return Namespace(**dict)
    else:
        _dict = {}
        for key, contain in dict.items():
            for field, value in contain.items():
                if not field in _dict:
                    _dict[field] = value
                else:
                    logging.error('Duplicate key when converting dict to namespace {} == {}-{}'.format(field, key, field))
        return Namespace(**_dict)