import os
import numpy as np
import cv2
import mysql.connector
import glob
# from cv2 import aruco
from io import BytesIO
import glob

import pathlib
import sys

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))

from adaptor import Adaptor

class Camera2(Adaptor):

    def __init__ (self, args, **kw) -> None:
        Adaptor.__init__(self, args, **kw)
        self.start_listen_bus()
        self.save_info()
    
    def load_config (self):
        self.refresh = 10
        for id in self.uHoo:
            self.uHoo[id] = Adaptor.load_config_by_id(self, id)
            self.uHoo[id]['access'] = {'key': None, 'timestamp': None}
            _ref = self.uHoo[id].get('refresh', 5)
            if _ref > self.refresh:
                self.refresh = _ref
    
    def process_redis_msg (self, ch, msg):
        if ch in self.subscribe_channels:
            self.status_lock = dt.datetime.now()
            logging.debug('[{}]: channel: {}, msg: {}'.format(COMID, ch, msg))
            _new = msg.get('new', {})
            _exist = msg.get('existing', {})
            if _new.get('last-status', 'off') != _exist.get('last-status', 'off'):
                self.set_power(_new.get('last-status', 'off'))
            for key, val in _new.get('last-msg', {}).items():
                if key in _exist.get('last-msg', {}):
                    if val != _exist['last-msg'][key]:
                        if key == 'setpoint-temp': self.set_temperature(val)
                        if key == 'user-mode': self.set_mode(val)
                        if key == 'fan-speed': self.set_fan_speed(val)
            self.get_status()



# Camera class has information that including its configuration and also its functionality such as its conversion, parameters and models.
class Camera(object) :

    def __init__(self,cam_id,db) :
        
        self.mycursor = db.cursor()
        self.id = cam_id

        sql_command = "SELECT * FROM cam_table_demo WHERE cam_id = {}".format(self.id)
        self.mycursor.execute(sql_command)
        sql_query = self.mycursor.fetchall()
        


        if len(sql_query) == 1 :

            for x in sql_query :

                self.cam_name = x[1]
                self.type = x[2]
                self.cam_int = self.loading_bytes(x[3])
                self.hom_mat_sit = self.loading_bytes(x[4])
                self.hom_mat_stand = self.loading_bytes(x[5])
                self.ud_vector = self.loading_bytes(x[6])
                self.img_path = x[7]
                ip_source = str(x[8])
                self.source = 'rtsp://Admin:Welcome123@' + ip_source + '/MediaInput/h264/'
                self.lookup = x[9]
                self.topleft = [int(i) for i in x[10].split(",")]
                # self.bottomright = [int(i) for i in x[16].split(",")]
        
        # Retrieve the floorplan information from the camera id
        sql_command = "SELECT floor_id FROM cam2floor_table WHERE cam_id = {}".format(self.id)
        self.mycursor.execute(sql_command)
        sql_query = self.mycursor.fetchall()
        
        if len(sql_query) == 1 :
            for x in sql_query :
                self.floor_id =  x[0]
        print(self.floor_id)
        # Retrieve the microphone information from the camera id
        sql_command = "SELECT mic_id FROM mic2cam_table WHERE cam_id = {}".format(self.id)
        self.mycursor.execute(sql_command)
        sql_query = self.mycursor.fetchall()
        
        # if len(sql_query) == 1 :
        #     for x in sql_query :
        #         self.mic_id =  x[0]
        # print(self.mic_id)
    def loading_bytes(self,query) :
        current_byte = BytesIO(query)
        loaded_numpy = np.load(current_byte,allow_pickle = True)

        return loaded_numpy

    def undistort_image(self,img,coeff = 0.5) :

        H,W,_ = img.shape

        dim = (H, W)

        focal_coef = coeff

        new_cam_int = copy.deepcopy(self.cam_int)

        new_cam_int[0,0] , new_cam_int[1,1] = new_cam_int[0,0] * focal_coef  , new_cam_int[1,1] * focal_coef
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.cam_int, self.ud_vector, np.eye(3), new_cam_int, dim, cv2.CV_16SC2)

        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_img

    def undistort(self,path,coeff) :
        
        images = []
        
        if path.split(".")[-1] == ".png" :
            cur_path = [path]

        else :
            cur_path = glob.glob(os.path.join(path,"*.png"))

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
    





        

