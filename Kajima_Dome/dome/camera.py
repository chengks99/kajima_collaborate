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
# from person_engine_yolox_kp_fisheye_PMV import PersonEngine
from person_engine_yolox_kp_fd_PMV import PersonEngine

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
        logging.debug(msg)
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
    def __init__(self, cfg, face_details, body_details, redis_conn) -> None:
        self.cfg_cam = cfg.pop('camera')
        self.cam_count = cfg.pop('count')
        self.cfg_person = cfg
        self.face_details = face_details
        self.redis_conn = redis_conn
        self.body_details = body_details

        self.init_cam_stream(self.cfg_cam)
        self.init_person_engine(self.cfg_person, self.cfg_cam)


    def init_cam_stream (self, cfg, start_frame=0):
        self.camera_width = cfg.get('camera_width', 1280)
        self.camera_height = cfg.get('camera_height', 960)
        self.camera_fps = cfg.get('fps', 30)
        self.camera_source1 = cfg.get('camera_source1', '')
        self.camera_source2 = cfg.get('camera_source2', '')
        self.start_frame = 0

        if self.camera_source1 == '' or self.camera_source2 == '':
            logging.error('Camera source is empty string ...')
            exit(1)

        self.stream1 = CameraStream(self.camera_width, self.camera_height, self.camera_fps, self.camera_source1, self.start_frame).start()
        self.stream2 = CameraStream(self.camera_width, self.camera_height, self.camera_fps, self.camera_source2, self.start_frame).start()
        logging.info("Initiated stream for {} ,{} ".format(self.camera_source1, self.camera_source2))
    
    def person_face_updates (self, msg):
        self.face_details = msg
        # print(self.face_details)
        self.person_engine.person_face_updates(self.face_details)
    
    def person_body_updates (self, msg):
        self.body_details = msg
        self.person_engine.person_body_updates(self.body_details)

        self.body_details = self.person_engine.body_details

        # print("The body db is : {}".format(self.body_details))


    def init_person_engine (self, cfg_person, cfg_cam):
        fList = ['pd_model', 'pr_model', 'kp_model', 'kp_cfg', 'pmv_model', 'cloth_model']
        for f in fList:
            fp = scriptpath.parent / cfg_person['model_path'].get(f, f)
            if f != 'pr_model':
                if not fp.is_file():
                    logging.error('Unable to locate {} at {} ...'.format(f, fp))
                    exit(1)
            cfg_person['model_path'][f] = str(fp)
        
        _fp = scriptpath.parent / cfg_person['body_model_info']['body_database']
        cfg_person['body_model_info']['body_database'] = str(_fp)
        self.person_engine = PersonEngine(cfg_person, cfg_cam, self.face_details, self.body_details, self.redis_conn)

    def run (self):
        # count = 0
        import time
        start_time = time.time()
        res = {'stream1': {'success': False, 'image': None}, 'stream2': {'success': False, 'image': None}}
        res['stream1']['success'], res['stream1']['image'] = self.stream1.read()
        res['stream2']['success'], res['stream2']['image'] = self.stream2.read()
        # logging.info(res)
        images = []
        for k, v in res.items():
            if not v['success']:
                logging.error('Unable to read stream from {}'.format(k))
                return False
            images.append(cv2.resize(v['image'],(1920, 1080)))
        # logging.info(images)
        self.output = self.person_engine.YoloxPersonBoxTrackMatchFeatureBatch(images)
         
        # cv2.imwrite("./image1/{}.png".format(start_time),images[0])
        # cv2.imwrite("./image2/{}.png".format(start_time),images[1])
        if self.output is not None:
            logging.info(self.output[0])
        # time2 = time.time()
        # logging.info("Processing time : {}".format(time2-start_time))
        # logging.info("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        return self.output , images

class CameraStream(object):
    def __init__(self, cam_width=640, cam_height=480, cam_fps=30, src=0, start_frame=0):
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
