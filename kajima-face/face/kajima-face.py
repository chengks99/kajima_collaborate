#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
face feature extraction script

Main jobs are:
1. call face feature extractions module
2. get feature array from extraction module
3. pack and publish into redis for backend server to insert into database
'''

import logging
import sys
import json
import pathlib
import fnmatch
import datetime as dt

import pandas as pd

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from plugin_module import PluginModule

import argsutils as au
from jsonutils import json2str
from hashcheckutils import CheckMD5

sys.path.append(str(scriptpath.parent / 'engine'))
from face_engine import FaceEngine

class FaceFeatureExtraction (PluginModule):
    component_name = 'FEM'

    def __init__(self, args, **kw) -> None:
        self.subscribe_channels = ['md5.check.request', 'md5.{}.error'.format(self.args.id)]
        self.redis_conn = au.connect_redis_with_args(args)
        self.args = args
        self.db = None

        self.housekeep_period = kw.pop('housekeep_perios', 150)

        PluginModule.__init__(self,
            redis_conn=self.redis_conn, db=self.db
        )
        self.start_listen_bus()
        self.start_thread('housekeep', self.housekeep)
        self.save_info()
    
    def __str (self):
        return '<FEM>'
    
    def get_info (self):
        ''' return a dict containing description of this module '''
        r = PluginModule.get_info(self)
        return r
    
    '''
        redis channel: face.recognition.vector
        redis msg format: {
            'length': INT, 
            'fvList': [
                {'face_id': INT, 'fv': LIST, 'human_id': INT},
                    .....
            ]
            'timestamp': datatime object
    '''
    def start (self, **extra_kw):
        self.cm5 = CheckMD5(self.redis_conn, self.component_prefix)
        self.cm5.report()
        fvDict = {}

        # checking files availability
        cfgm = scriptpath.parent / self.args.config_model
        csvf = scriptpath.parent / self.args.csv_path

        if not cfgm.exists():
            logging.error('Unable to locate config model file at {}'.format(str(cfgm)))
            exit(1)
        
        if not csvf.exists():
            logging.error('Unable to locate name list and image path at {}'.format(str(csvf)))
            exit(1)
        
        cfg = self.load_config(cfgm)
        if cfg is None: exit(1)

        self.fe = FaceEngine(
            fd_model_path=cfg['fd_model_path'],
            fr_model_path=cfg['fr_model_path'],
            fd_threshold=cfg['fd_threshold'],
            fd_input_resize=cfg['fd_input_resize'],
            max_detected_faces=cfg['max_detected_faces'],
            fr_threshold=cfg['fr_threshold'],
            device=cfg['device'],
            rgb=cfg['rgb'],
            emb_layer_name=cfg['emb_layer_name']
        )

        fvDict['fvList'] = self.fv_extraction(csvf)

        fvDict['timestamp'] = dt.datetime.now()
        self.redis_conn.publish('face.recognition.vector', json2str(fvDict))        

    def load_config (self, cfgm):
        try:
            cfg = json.load(open(str(cfgm, 'r')))
            for k in ['fd_input_resize', 'device', 'max_detected_faces', 'rgb']:
                if k in cfg:
                    cfg[k] = int(cfg[k])
            return cfg
        except Exception as e:
            logging.error('Error read config model file: {}'.format(e))
            return None

    def fv_extraction (self, csvf):
        fvList = []
        df = pd.read_csv(str(csvf, names=['name', 'image_path']))
        summary = {
            'total': len(df.index),
            'missed': 0,
            'failed': 0,
            'loaded': 0,
        }
        for i, row in df.iterrows():
            _imgf = scriptpath.parent / row['image_path']
            if not _imgf.exists():
                logging.error('Unable to locate {} face image at {}'.format(row['name'], str(_imgf)))
                summary['missed'] += 1
                continue

            _img = self.fe.read_image(str(_imgf))
            if _img is None:
                logging.error('Unable to load {} face image at {}'.format(row['name'], str(_imgf)))
                summary['failed'] += 1
                continue

            _out = self.fe.PersonalFeatures(self.fe.color_conversion(_img), max_faces=1)
            if _out is None:
                summary['failed'] += 1
                logging.error('Unable to extract {} face feature...'.format(row['name']))
                continue

            feature, race, age_grp, gender = _out
            fvList.append({
                'name': row['name'],
                'fv': feature,
                'race': race,
                'age_grp': age_grp,
                'gender': gender
            })
            summary['loaded'] += 1
            logging.debug('{} feature loaded...'.format(row['name']))
        
        logging.debug('Loading summary:')
        for c in summary.items():
            if c == 'total': continue
            logging.debug('{}: {}/{} ({:.2f}%)'.format(c, summary[c], summary['total'], (summary[c]/summary['total']*100)))

        return fvList

    def process_redis_msg (self, ch, msg):
        if ch in self.subscribe_channels:
            logging.debug('{} received redis message {}: {}'.format(self, ch, msg))
            if ch == 'md5.check.request':
                self.cm5.report()
            if fnmatch.fnmatch(ch, 'md5.*.error'):
                self.cm5.sum_check_error(msg)

if __name__ == "__main__":
    parser = au.init_parser('Face Feature Extraction Module', redis={}, sql={})
    au.add_arg(parser, '--config_model', h='specify model configuration file {D}', d='config/config_model.json')
    au.add_arg(parser, '--csv_path', h='specify file containing names and paths to image', d='test_images/namelist.csv')
    args = au.parse_args(parser)

    fem = FaceFeatureExtraction(args=args)
    fem.start()

    try:
        while not fem.is_quit(10):
            pass
    except KeyboardInterrupt:
        logging.info('Ctrl-C received -- terminating ...')
        fem.close()