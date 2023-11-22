import sys
import os
import json
import logging
import pathlib
import fnmatch
import datetime as dt
import numexpr as ne
ne.set_num_threads(8)

from adaptor import Adaptor
from camera import CamHolder, FloorHolder, MicHolder, DetectionEngine, CameraStream

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath / 'common'))
import argsutils as au
from jsonutils import json2str
from hashcheckutils import CheckMD5
#from localization import localize

class Dome(Adaptor):
    def __init__(self, args, **kw):
        self.args = args
        self.cfg = {}
        self.subscribe_channels = [
            'camera.{}.update'.format(self.args.id),
            'camera.{}.sqlquery'.format(self.args.id),
            'md5.check.request', 
            'md5.{}.error'.format(self.args.id)
        ]
        self.pcid = self.args.pcid
        Adaptor.__init__(self, args, **kw)
        self.load_config()
        logging.debug(self.cfg)
        self.start_listen_bus()
        self.run()
    
    def load_config (self):
        logging.debug('Load configuration from Redis for {}'.format(self.args.id))
        cfg = Adaptor.get_redismsg_by_channel(self, '{}.config'.format(self.component_prefix))
        if cfg is None:
            logging.debug('No Configuration for {} found in Redis, use config file {} instead'.format(self.args.id, self.args.config))
            if not os.path.isfile(self.args.config):
                logging.error('Unable to retrieve configuration file ...')
                exit(1)
            with open(self.args.config) as cfgf:
                self.cfg = json.load(cfgf)
        else:
            logging.debug('Configuration in redis loaded ...')
            if 'pcid' in cfg: self.pcid = cfg['pcid']
            if 'config' in cfg:
                self.cfg = cfg['config']
            else:
                for key, val in cfg.items():
                    if key == 'id': continue
                    cfgPath = str(scriptpath.parent / val)
                    if not os.path.isfile(cfgPath):
                        logging.error('Unable to retrieve {} config from path {}'.format(key, val))
                        exit(1)
                    with open(val) as cfgf:
                        self.cfg.update(json.load(cfgf))
        logging.debug('{} configuration: {}'.format(self.args.id, self.cfg))

    def run (self):
        self.cm5 = CheckMD5(self.redis_conn, self.component_prefix)
        self.cm5.report()
        msg = Adaptor.get_redismsg_by_channel(self, '{}.detail-config'.format(self.component_prefix))
        if msg is None:
            logging.debug('Detail config not found, make request for detail config to backend server')
            self.publish_redis_msg('{}.query'.format(self.component_prefix), {'msgType': 'init'})
        else:
            logging.debug('Found detail config in redis bus')
            self.process_sql_result(msg)
            self.run_init()

    def process_redis_msg(self, ch, msg):
        if ch in self.subscribe_channels:
            if 'sqlquery' in ch:
                logging.debug('[{}]: channel: {}'.format(self.args.id, ch))
                self.process_sql_result(msg)
                self.run_init()
            if ch == 'person.face.updates':
                logging.debug('[{}]: channel: {}'.format(self.args.id, ch))
                self.cur_engine.person_face_updates(msg)
            if ch == 'person.body.updates':
                logging.debug('[{}]: channel: {}'.format(self.args.id, ch))
                self.cur_engine.person_body_updates(msg)
            if ch == 'md5.check.request':
                self.cm5.report()
            if fnmatch.fnmatch(ch, 'md5.*.error'):
                self.cm5.sum_check_error(msg)

    def process_sql_result (self, msg):
        if msg.get('type', '') == 'init':
            self.cam = CamHolder(self.args.id, msg.get('cam', None))
            self.floor = FloorHolder(self.args.id, msg.get('floor', None))
            self.mic = MicHolder(self.args.id, msg.get('mic', None))
    
    def run_init (self):
        # should contain {'fvList': [{'eID', 'features}, ...]}
        face_details = Adaptor.get_redismsg_by_channel(self, 'person.face.features')
        if face_details is None:
            logging.error('Empty face featurs')
            exit(1)

        # should contain {'fvList': [{'name', 'features', 'person_details'}, ...] }
        body_details = Adaptor.get_redismsg_by_channel(self, 'person.body.features')
        if body_details is None:
            logging.warning('Empty body featurs')
            body_details = {'fvList': []}
        self.process_engine(face_details, body_details)

    def process_engine (self, face_details, body_details):
        # self.cur_engine = DetectionEngine(self.cfg, face_details, body_details, self.get_redis_conn())
        self.cur_engine = DetectionEngine(self.cfg, face_details, body_details, self.redis_conn)
        logging.debug('Detection Engine module initialized...')

    def get_status (self):
        r = Adaptor.get_status(self)
        r['status'] = 'Normal'
        return r
    
    def get_info (self):
        r = Adaptor.get_info(self)
        return r
    
    def close (self):
        self.th_quit.set()

if __name__ == "__main__":
    import argsutils as au
    from adaptor import add_common_adaptor_args

    parser = au.init_parser('Kajima Dome Application')
    add_common_adaptor_args(
        parser,
        type='dome',
        location='Office-1',
        id='dome-18',
        pcid=7000
    )
    args = au.parse_args(parser)
    logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s %(module)s %(levelname)s]: %(message)s')   
    dome = Dome(args=args)
    logging.debug(args=args)
    logging.debug("{:all}".format(dome))
    try:
        while not dome.is_quit(1):
            pass
    except KeyboardInterrupt:
        dome.close()