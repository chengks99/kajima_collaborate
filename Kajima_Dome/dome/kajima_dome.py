import sys
import os
import json
import logging
import pathlib
import fnmatch
import datetime as dt
import numexpr as ne
ne.set_num_threads(8)
import time
import threading
from adaptor import Adaptor
# from camera import CamHolder, FloorHolder, MicHolder


scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath / 'common'))
from jsonutils import json2str
from hashcheckutils import CheckMD5
#from localization import localize

#-------------------- Extract the majority number from utility rate --------------

def majority_element(nums):
    count = 0
    candidate = None

    for num in nums:
        if count == 0:
            candidate = num
            count = 1
        elif num == candidate:
            count += 1
        else:
            count -= 1

    return candidate

class Dome(Adaptor):
    def __init__(self, args, **kw):
        self.args = args
        self.utility = self.args.utility
        self.view = self.args.view
        self.cfg = {}
        self.subscribe_channels = [
            'camera.{}.update'.format(self.args.id),
            'camera.{}.sqlquery'.format(self.args.id),
            'md5.check.request', 
            'md5.{}.error'.format(self.args.id),
            'person.body.updates',
            'person.face.updates'
        ]
        self.pcid = self.args.pcid
        Adaptor.__init__(self, args, **kw)
        self.th_quit = threading.Event()
        self.load_config()
        self.start_listen_bus()
        self.run()
    
    def load_config (self):
        logging.debug('Load configuration from Redis for {}'.format(self.args.id))
        # cfg = Adaptor.get_redismsg_by_channel(self, '{}.config'.format(self.component_prefix))
        cfg = None
        if cfg is None:
            logging.debug('No Configuration for {} found in Redis, use config file {} instead'.format(self.args.id, 'dome-{}'.format(self.args.id)))
            if not os.path.isfile('./config/dome-{}.json'.format(self.args.id)):
                logging.error('Unable to retrieve configuration file ...')
                exit(1)
            with open('./config/dome-{}.json'.format(self.args.id)) as cfgf:
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
        # msg = None
        if msg is None:
            logging.debug('Detail config not found, make request for detail config to backend server')
            self.publish_redis_msg('{}.query'.format(self.component_prefix), {'msgType': 'init'})
        else:
            logging.debug('Found detail config in redis bus')
            self.process_sql_result(msg)
            self.run_init()

    def process_redis_msg(self, ch, msg):
        if ch in self.subscribe_channels:
            logging.debug(msg)
            if 'sqlquery' in ch:
                # logging.debug('[{}]: channel: {}'.format(self.args.id, ch))
                self.process_sql_result(msg)
                self.run_init()
            if ch == 'person.face.updates':
                # print('[{}]: channel: {}'.format(self.args.id, ch))
                print("Received Face Updates")
                try :
                    self.cur_engine.person_face_updates(msg)
                except:
                    pass
            if ch == 'person.body.updates':
                # print('[{}]: channel: {}'.format(self.args.id, ch))
                print("Received Body Updates")
                try:
                    self.cur_engine.person_body_updates(msg)
                except:
                    pass
            if ch == 'md5.check.request':
                self.cm5.report()
            if fnmatch.fnmatch(ch, 'md5.*.error'):
                self.cm5.sum_check_error(msg)

    def process_sql_result (self, msg):
        if msg.get('type', '') == 'init':
            from camera import CamHolder, FloorHolder, MicHolder
            self.cam = CamHolder(self.args.id, msg.get('cam', None))
            self.floor = FloorHolder(self.args.id, msg.get('floor', None))
            self.mic = MicHolder(self.args.id, msg.get('mic', None))
    
    def run_init (self):
        # should contain {'fvList': [{'eID', 'features}, ...]}
        face_details = Adaptor.get_redismsg_by_channel(self, 'person.face.features')
        if face_details is None:
            print('Empty face features')
            face_details = None
            # exit(1)
        # logging.info("The face are : {}".format(face_details))
        # exit()
        # should contain {'fvList': [{'name', 'features', 'person_details'}, ...] }
        body_details = Adaptor.get_redismsg_by_channel(self,'person.body.features')
        if body_details is None:
            print('Empty body features')
            # body_details = {'fvList': []}
            body_details = None
        self.process_engine(face_details, body_details)

    def process_engine (self, face_details, body_details):
        logging.debug('Detection Engine module initialized...')
        logging.info("Running Detection Engine")
        if self.view:
            import cv2
            import numpy as np        
        from camera import DetectionEngine

        self.cur_engine = DetectionEngine(self.cfg, face_details, body_details, self.get_redis_conn())
        last_report = time.time()
        self.utility_list = []
        while not self.is_quit():
            output, images = self.cur_engine.run()

            if self.view:
                # images = np.hstack((images[0], images[1]))
                # images = cv2.resize(images, (3840, 1080))
                cv2.namedWindow("Video feed", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Video feed", 1280, 720)
                cv2.imshow("Video feed", images[0])
  
            if output is not None:
                roi_dts = output[0]

            self.utility_list.append(max(len(roi_dts[1]),len(roi_dts[0])) if output is not None else 0)
            
            if self.utility and time.time() - last_report >= 60:
                ppl_count =  majority_element(self.utility_list) if self.utility_list is not None else 0
                msg = {
                "cam_id": self.args.id,
                "timestamp": {"$dt": dt.datetime.now().timestamp()},
                "pcid": 7000,
                "people_count": ppl_count
                }   
                self.redis_conn.publish("util.{}.query".format(self.args.id),json2str(msg))
                last_report = time.time()
                self.utility_list = [] 
            else:
                # logging.info("This dome will not process people_counting")   
                pass   

            if self.view  and cv2.waitKey(1) & 0xFF == ord('q'):
            # video_shower.stop()
            # change waitKey() for different refresh interval; press 'q' key to quit
                cv2.destroyAllWindows()
                self.view = False
    
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
    au.add_arg(parser, '-u','--utility', h='Running for ultility rate  {D}', a = True)
    au.add_arg(parser, '-v','--view', h='Viewing for debug mode  {D}', a = True)

    add_common_adaptor_args(
        parser,
        type='dome',
        location='Office-1',
        id='dome-7001',
        pcid=7000
    )
    args = au.parse_args(parser)  
    dome = Dome(args=args)
    logging.debug("{:all}".format(dome))
    try:
        while not dome.is_quit(1):
            pass
    except KeyboardInterrupt:
        dome.close() 
    