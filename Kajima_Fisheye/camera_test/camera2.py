import numpy as np
import cv2
import time
import sys
import pathlib
import copy
import glob
import os
import io
import base64
import logging
import numpy as np

from queue import Queue
from threading import Thread, Lock

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'engine'))
from person_engine_yolox_kp_fisheye_PMV import PersonEngine
#from person_engine_rapid_kp_fisheye import PersonEngine

class CamHolder(object):
    def __init__(self, id, msg) -> None:
        self.id = id
        self.process_msg(msg)
    
    def process_msg (self, msg):
        if msg is None: 
            logging.error('Error retrieve camera demo info from database')
            exit(1)
        
        self.cam_name = msg.get('cam_name', '')
        self.type = msg.get('type', '')
        '''
        self.cam_int = self.loading_bytes(msg.get('cam_int', 0))
        self.hom_mat_sit = self.loading_bytes(msg.get('hom_mat_sit', 0))
        self.hom_mat_stand = self.loading_bytes(msg.get('hom_mat_stand', 0))
        self.ud_vector = self.loading_bytes(msg.get('ud_vector', 0))
        '''
        self.cam_int = np.asarray(msg.get('cam_int', []))
        self.hom_mat_sit = np.asarray(msg.get('hom_mat_sit', []))
        self.hom_mat_stand = np.asarray(msg.get('hom_mat_stand', []))
        self.ud_vector = np.asarray(msg.get('ud_vector', []))
        self.img_path = msg.get('img_path', '')
        self.source = msg.get('source', '')
        self.lookup = msg.get('lookup', '')
        self.topleft = msg.get('topleft', [])

    def loading_bytes (self, query):
        current_byte = io.BytesIO(query)
        return np.load(current_byte, allow_pickle=True)
    
    def undistort_image (self, img, coeff=0.5):
        H,W,_ = img.shape
        dim = (H, W)
        focal_coef = coeff

        new_cam_int = copy.deepcopy(self.cam_int)
        new_cam_int[0,0] , new_cam_int[1,1] = new_cam_int[0,0] * focal_coef  , new_cam_int[1,1] * focal_coef
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.cam_int, self.ud_vector, np.eye(3), new_cam_int, dim, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_img

    def undistort(self, path, coeff) :
        images = []
        cur_path = [path] if path.split(".")[-1] == ".png" else glob.glob(os.path.join(path,"*.png"))

        for i in sorted(cur_path) :
            cur_img = cv2.imread(i)
            cur_image = self.undistort_image(cur_img,coeff)
            images.append(cur_image)
            
        return images

    def capture_frame(self,frame_num) :
        saved_frame = []
        number_frame = frame_num

        capture = cv2.VideoCapture(self.source)
        for i in range(frame_num) :
            ret, frame = capture.read()
            saved_frame.append(frame) 

        return saved_frame

class FloorHolder(object):
    def __init__(self, id, msg) -> None:
        self.id = id
        self.process_msg(msg)
    
    def process_msg (self, msg):
        if msg is None: 
            logging.error('Error retrieve camera floorplan info from database')
            exit(1)
        
        self.name = msg.get('name', '')
        self.scale_x = msg.get('scale_x', -1)
        self.scale_y = msg.get('scale_y', -1)
        self.image = self.load_floorplan(msg.get('image', None))
        self.x_dir = msg.get('x_dir', '')
        self.y_dir = msg.get('y_dir', '')
        self.origin = msg.get('origin', [])

    def load_floorplan (self, source):
        if source is None:
            logging.error('Floorplan source is None')
            exit(1)
        img = base64.b64decode(source); 
        npimg = np.fromstring(img, dtype=np.uint8); 
        return cv2.imdecode(npimg, 1)
    
    def show_floorplan (self):
        cv2.imshow(self.name, self.image)
        cv2.waitkey(0)
        cv2.destroyAllWindows()

class MicHolder(object):
    def __init__(self, id, msg) -> None:
        self.id = id
        self.process_msg(msg)
    
    def process_msg (self, msg):
        if msg is None: 
            logging.error('Error retrieve camera microphone info from database')
            exit(1)
        
        self.name = msg.get('name', '')
        self.type = msg.get('type', '')
        self.source = msg.get('source', '')
        self.saved_path = msg.get('saved_path', '')

class DetectionEngine(object):
    def __init__(self, cfg,body_details) -> None:
        self.flag = False
        self.body_details = body_details
        self.init_pe(cfg)
        self.input_q = Queue(1)
        self.detect_q = Queue(1)
        self.skeleton_q = Queue(1)
        self.feature_q = Queue(1)
        self.output_q = Queue(1)

        self.detect_t = Thread(
            target=self.person_engine.detection_engine.worker,
            args=(self.input_q, self.detect_q))
        self.detect_t.daemon = True
        self.detect_t.start()
        logging.debug('Thread "detect_t" started ...')

        self.skeleton_t = Thread(
            target=self.person_engine.skeleton_engine.worker,
            args=(self.detect_q, self.skeleton_q))
        self.skeleton_t.daemon = True
        self.skeleton_t.start()
        logging.debug('Thread "skeleton_t" started ...')

        self.feature_t = Thread(
            target=self.person_engine.feature_engine.worker,
            args=(self.skeleton_q, self.feature_q))
        self.feature_t.daemon = True
        self.feature_t.start()
        logging.debug('Thread "feature_t" started ...')

        self.person_t = Thread(
            target=self.person_engine.worker,
            args=(self.feature_q, self.output_q))
        self.person_t.daemon = True
        self.person_t.start()
        logging.debug('Thread "detect_t" started ...')
    
    def init_pe (self, cfg):
        # checking file location
        fList = ['pd_model', 'pr_model', 'kp_model', 'kp_cfg', 'pmv_model', 'cloth_model']
        fPath = {}
        for f in fList:
            fp = scriptpath.parent / cfg['model_path'].get(f, f)
            if f != 'pr_model':
                if not fp.is_file():
                    logging.error('Unable to locate {} at {} ...'.format(f, fp))
                    exit(1)
            fPath[f] = str(fp)
        
        _fp = scriptpath.parent / cfg['body_model_info']['body_database']
        fPath['body_database'] = str(_fp)

        self.person_engine = PersonEngine(
            pd_model=fPath['pd_model'], 
            pr_model=fPath['pr_model'], 
            kp_model=fPath['kp_model'], 
            kp_cfg=fPath['kp_cfg'], 
            cloth_model=fPath['cloth_model'], 
            pmv_model=fPath['pmv_model'],
            radiant_temp=cfg['pmv_model_info'].get('radiant_temp'),
            room_temp=cfg['pmv_model_info'].get('room_temp'),
            rel_humidity=cfg['pmv_model_info'].get('rel_humidity'),
            air_speed=cfg['pmv_model_info'].get('air_speed'),
            body_db_file= fPath.get('body_database',""),
            pd_det_threshold=cfg['body_model_info'].get('pd_det_threshold'),
            pd_nms_threshold=cfg['body_model_info'].get('pd_nms_threshold'),
            pd_input_resize=int(cfg['body_model_info'].get('pd_input_resize')),
            max_detected_persons=int(cfg['body_model_info'].get('max_detected_person')),
            min_person_width=int(cfg['body_model_info'].get('min_person_width')),
            pr_threshold=cfg['body_model_info'].get('pr_threshold'),
            device=int(cfg['body_model_info'].get('device')),
            body_details = self.body_details
        )

        logging.debug('Person Engine loaded successfully')
        self.flag = True
    
    def set_env_var (self, temp, humid):
        if self.person_engine.flag:
            self.person_engine.set_env_var(temp, humid)
 
    def person_body_updates(self,msg):
        self.body_details = msg
        self.person_engine.person_body_updates(self.body_details)
        self.body_details = self.person_engine.body_details

    def run (self, image):
        # image = cv2.flip(image, 0)
        # image = cv2.flip(image, 1)

        self.input_q.put(image)
        self.output = None
        if self.output_q.empty():
            # self.output = None
            pass
        else:
            self.output = self.output_q.get()
        return self.output

class CameraStream(object):
    def __init__(self, cam_width=640, cam_height=480, cam_fps=30, src=0, start_frame=0):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, cam_width if not cam_width is None else 640)
        self.stream.set(4, cam_height if not cam_height is None else 480)
        self.stream.set(5, cam_fps if not cam_fps is None else 30)

        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            logging.debug("Camera Stream already started!!")
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        image = self.frame.copy()
        self.read_lock.release()
        return self.grabbed, image
        
    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stream.release()
