import os
from re import T
# import base64

import cv2
import numpy as np
from skimage import transform
# from sklearn.preprocessing import normalize

from yolox_detector import YoloxPredictor
from embedding import UbodyEmbedding, FullbodyEmbedding
from person_tracker_PMV import FishEyeTracker

import visualization as vsl

import sys
import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'pose'))
from detect import HumanPoseDetection

#16fp models
# from Pose.detect_16fp import HumanPoseDetection
# from Engines.embedding_16fp import UbodyEmbedding

sys.path.append(str(scriptpath.parent / 'common'))
from database import Database

from collections import OrderedDict

from PIL import Image
import time
import math
import pandas as pd
from termcolor import colored
from queue import Queue

import torch
# import torchvision.transforms as transforms

# from mmcv import Config
sys.path.append(str(scriptpath.parent / 'thermalcomfort'))
from clothatt_prediction import ClothPredictor

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # super().__init__()
        self.fc1 = nn.Linear(34, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PersonEngine(object):
    def __init__(self, pd_model, pr_model, kp_model, kp_cfg, 
                 cloth_model, pmv_model, radiant_temp, room_temp, rel_humidity, air_speed,
                 body_db_file,
                 body_details,
                 pd_det_threshold=0.5, pd_nms_threshold=0.3, pd_input_resize=0, 
                 max_detected_persons=0, min_person_width=50, pr_threshold=0.3, device=-1):
        
        self.body_details = body_details
        # self.rgb = rgb
        self.min_person_width = min_person_width
        self.frame_num = 0
        self.LMK_VISIBILITY_THRESHOLD = 0

        self.pr_threshold = pr_threshold

        self.detection_engine = DetectionEngine(pd_model, pd_det_threshold, pd_nms_threshold,
                                pd_input_resize, max_detected_persons, min_person_width, device) 
        logging.debug('Detection Engine loaded ...')

        self.skeleton_engine = SkeletonEngine(kp_model, kp_cfg, device)
        logging.debug('Skeleton Engine loaded ...')

        self.feature_engine = FeatureEngine(pr_model, self.LMK_VISIBILITY_THRESHOLD, device)
        logging.debug('Feature Engine loaded ...')

        env_variables =  {'tr': radiant_temp, 'tdb': room_temp, 
                        'to': room_temp, 'rh': rel_humidity, 'v': air_speed}

        #print (pr_threshold, body_db_file, pmv_model, env_variables)
        self.BodyTracker = FishEyeTracker(
            pr_threshold, 
            body_db_file, 
            pmv_model, 
            env_variables,
            body_details=self.body_details,
        )
        logging.debug('Body Tracker loaded ...')

        self.cloth_predictor = ClothPredictor(cloth_model, device)
        logging.debug('Cloth Predictor loaded')

        #load action recognition model
        model_location = scriptpath.parent / 'storage/models/yolox/action_jk.pth'
        #model_locatioon = "./models/action_jk.pth"
        if not model_location.is_file():
            logging.error('Unable to locate YOLOX model location at {}'.format(str(model_location)))
            exit(1)

        if device<0:
            self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cuda:{}".format(device))
        
        self.actionNet = Net()
        self.actionNet.load_state_dict(torch.load(str(model_location)))
        # self.actionNet = torch.load(model_locatioon)

        self.actionNet = self.actionNet.to(self.device) 
        self.actionNet.eval()       

        # self.queue_det = Queue(1)
        # self.queue_skl = Queue(1)
        # self.queue_feat = Queue(1)
        # self.Yolox_det = Thread(target=self.YoloxPersonDetect.worker, args=(self.queue_det))
        # self.Yolox_det.daemon = True
        # self.Yolox_det.start()
    def person_body_updates (self, body_details):
        self.body_details = body_details
        self.BodyTracker.body_updates(body_details)
        
    def YoloxPersonBoxTrackMatchFeature(self, bgrimg, new_dts, lmks, lmk_confs, body_crops, body_features):
        img_dim = bgrimg.shape[:2]
        # rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        # dts = self.detection_engine.YoloxPersonDetect(bgrimg)

        self.frame_num += 1

        # if len(dts) == 0:
        #     return None    

        # #skeleton detection
        # lmks, lmk_confs, new_dts = self.skeleton_engine.SkeletonDetect(bgrimg, dts)

        # #feature extraction
        # body_crops, body_features = self.feature_engine.BodyFeatureExtraction(bgrimg, new_dts, lmks, lmk_confs)

        bboxs = new_dts[:,:4]
        #print("Detected {} boxes".format(bboxs.shape))

        #clothattribute
        try:
            cloth_attributes = self.cloth_predictor.model_inference(bgrimg, lmks, lmk_confs)
        except Exception as e:
            print("CLOTH ATT Detection failed, replace to alternative default")
            cloth_attributes = []
            for lmk, lmk_conf in zip(lmks, lmk_confs):
                cloth_attributes.append({'top': ['short_Sleeve_Top', 0.99], 'bot': ['trousers', 0.99], 'full': ['vest_Dress', 0.99], 'iclo': 0.38})
        print("CLOTH ATT ", cloth_attributes)

        #estimate action 
        actions = self.__action_recognition(lmks, img_dim)

        #update tracking table
        t1 = time.time()
        trackids, subids, pmvs = self.BodyTracker.update(bboxs, lmks, lmk_confs, body_features, img_dim, cloth_attributes, actions)
        t2 = time.time() - t1
        #print("tracking time ", t2)

        # visualization.draw_dt_on_np(bgrimg, new_dts)
        vsl.draw_lmks(bgrimg, lmks, lmk_confs, (255,255,255), self.LMK_VISIBILITY_THRESHOLD)
        bgrimg = vsl.draw_yolox(bgrimg, new_dts)

        return new_dts, body_crops, subids, trackids, lmks, bgrimg, pmvs, actions

    def YoloxPersonBoxTrackMatchFeature_no_thread(self, bgrimg):
        img_dim = bgrimg.shape[:2]
        # rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)

        dts = self.detection_engine.YoloxPersonDetect(bgrimg)

        self.frame_num += 1

        if dts is None:
            return None 

        #skeleton detection
        lmks, lmk_confs, new_dts = self.skeleton_engine.SkeletonDetect(bgrimg, dts)

        #feature extraction
        body_crops, body_features = self.feature_engine.BodyFeatureExtraction(bgrimg, new_dts, lmks, lmk_confs)
        bboxs = new_dts[:,:4]

        #clothattribute
        try:
            cloth_attributes = self.cloth_predictor.model_inference(bgrimg, lmks, lmk_confs)
        except Exception as e:
            print("CLOTH ATT Detection failed, replace to alternative default")
            cloth_attributes = {'top': ['short_Sleeve_Top', 0.99], 'bot': ['trousers', 0.99], 'full': ['vest_Dress', 0.99], 'iclo': 0.38} 
        print("CLOTH ATT ", cloth_attributes)      
        #estimate action 
        actions = self.__action_recognition(lmks, img_dim)
    
        #update tracking table
        t1 = time.time()
        trackids, subids, pmvs = self.BodyTracker.update(bboxs, lmks, lmk_confs, body_features, img_dim, cloth_attributes, actions)
        t2 = time.time() - t1
        #logging.debug("tracking time: {} "format(t2))
    
        # print("TRACK PMV ", pmvs)

        # visualization.draw_dt_on_np(bgrimg, dts)
        vsl.draw_lmks(bgrimg, lmks, lmk_confs, (255,255,255), self.LMK_VISIBILITY_THRESHOLD)
        bgrimg = vsl.draw_yolox(bgrimg, dts)

        return new_dts, body_crops, subids, trackids, lmks, bgrimg, pmvs, actions
        # return dts

    def __convert_to_polar(self, pointx, pointy, img_size) :

        # radius = [2992/2,2992/2]
        radius = [img_size[0]/2, img_size[1]/2]

        theta = math.atan(math.sqrt(pointx**2 + pointy**2)/radius[0])
        psi = math.atan(pointy/pointx)

        new_point = [psi,theta]

        return psi, theta

    def __action_recognition(self, lmks, img_size):
        #estimate action: 0(standing), 1(sitting)

        #convert lmks to polar coordiantes
        polar_lmks = []
        for i in range(len(lmks)):
            lmk = lmks[i]
            polar_lmk = []
            for joint in lmk:
                psi,theta = self.__convert_to_polar(joint[0], joint[1], img_size)
                polar_lmk.append(psi)
                polar_lmk.append(theta)
            
            polar_lmks.append(polar_lmk)

        with torch.no_grad():
            inputs = torch.FloatTensor(polar_lmks)
            inputs = inputs.cuda(self.device)
            outputs = self.actionNet(inputs)

            results = torch.argmax(outputs, dim=1)
            actions = results.cpu().numpy()
            #print("action ", actions)

        # print(new_out)
        # results = list(self.flatten(results))

        return actions

    def worker(self, input_q, output_q):
        # detect_q = Queue()
        # skeleton_q = Queue()
        # feature_q = Queue()
        # detect_q.put(self.detection_engine.worker,

        while True:
            output = input_q.get()
            # bgrimg, dts = output
            bgrimg, dts, lmks, lmk_confs, body_crops, body_features = output
            # output_q.put()
            output_q.put(self.YoloxPersonBoxTrackMatchFeature(bgrimg, dts, lmks, lmk_confs, body_crops, body_features))
            logging.debug('Person Engine track Match Feature')

    def SaveBodyDB(self):
        self.BodyFaceTracker.save_database(self.body_db_file)

class DetectionEngine(object):
    def __init__(self, pd_model, pd_det_threshold=0.5, pd_nms_threshold=0.3, pd_input_resize=0, 
        max_detected_persons=0, min_person_width=100, device=-1):

        self.min_person_width = min_person_width
        self.max_person_height = 600

        self.__detector_yolox = YoloxPredictor(model_name='yolox-l', model_weight=pd_model, 
            test_size=pd_input_resize, test_conf=pd_det_threshold, 
            nmsthre=pd_nms_threshold, maxppl=max_detected_persons, device=device)

    def YoloxPersonDetect(self, bgrimg):
        # rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        # pilimg = Image.fromarray(rgbimg)

        t1 = time.time()
        #output: in darknet format (cx,cy,w,h)
        detections = self.__detector_yolox.inference(bgrimg)

        # #filter out small detections
        dts = self.__filterDts(detections)

        t2 = time.time() - t1
        #logging.debug('Detection Time: {}'.format(t2))

        return dts 

    def YoloxPersonDetectBatch(self, bgrimgs):
        t1 = time.time()

        # alldetections = []
        # for img in bgrimgs:
        #     detections = self.__detector_yolox.inference(img)
        #     alldetections.append(detections)

        alldetections = self.__detector_yolox.inferenceBatch(bgrimgs)

        if alldetections is None:
            return None

        #filter out small box detections
        dts = self.__filterDtsBatch(alldetections)

        t2 = time.time() - t1
        #logging.debug('Detection Time: {}'.format(t2))

        return dts

    def __filterDts(self, dts):
        # detections = dts
        # #filter out detection which have width above threshold
        # detections = detections[detections[:,2] >= self.min_person_width]

        if dts is None:
            return None 

        dts = np.array(dts)
        width = dts[:,2]
        height = dts[:,3]
        #print(width)

        min_dim = np.minimum(width, height)
        max_dim = np.maximum(width, height)
        #print(min_dim)

        # dts = dts[dts[:,3]>=self.min_person_width]
        # dts = dts[dts[:,4]<=self.max_person_height]

        dts = dts[min_dim>=self.min_person_width]
        max_dim = max_dim[min_dim>=self.min_person_width]
        dts = dts[max_dim<=self.max_person_height]

        if len(dts):
            return dts #np.array(filter_dts) #detections
        else:
            return None

    def __filterDtsBatch(self, dts):
        filter_detection = []
        for dt in dts:
            if dt is None:
                filter_detection.append(None)
                continue

            detections = self.__filterDts(dt)
            filter_detection.append(detections)

        return filter_detection
        # return np.array(filter_detection)

    def worker(self, input_q, output_q):
        while True:
            bgrimg = input_q.get()
            dts = self.YoloxPersonDetect(bgrimg)
            if dts is None:
                continue
            output_q.put((bgrimg, dts))
            # queue.task_done() # this is new 

class SkeletonEngine(object):
    def __init__(self, kp_model, kp_cfg, device=-1):

        # print("initializing lmk model", kp_model, kp_cfg)
        self.kp = HumanPoseDetection(kp_model, kp_cfg, device)

    def __get_angle_batch(self, imgw, imgh, detections):
        dir = []
        for bb in detections:
            x,y,w,h = bb[:4]
            theta = self.__get_angle(imgw, imgh, x, y)
            dir.append(theta)

        return dir

    def __get_angle(self, imgw, imgh, x, y):
        # https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
        top_x, top_y = imgw/2, 0
        center_x, center_y = imgw/2, imgh/2

        a = math.sqrt((top_x-x)**2 + (top_y-y)**2)
        b = math.sqrt((top_x-center_x)**2 + (top_y-center_y)**2)
        c = math.sqrt((center_x-x)**2 + (center_y-y)**2)

        A = math.acos((b**2 + c**2 - a**2) / (2*b*c))
        if x > center_x:
            return (math.pi + math.pi - A)

        return A 

    def SkeletonDetect(self, bgrimg, dts):
        t1 = time.time()
        imgh, imgw = bgrimg.shape[:2]
        bboxs = dts[:,:4]

        #get box angles
        rot = self.__get_angle_batch(imgw, imgh, dts)

        #detect body landmarks
        lmks, lmk_confs = self.kp.detect_batch(bgrimg, bboxs, rot)

        #crop upper body
        t2 = time.time() - t1
        #logging.debug('Skeleton Detection time: {}'.format(t2))

        #create cxcywh, angle, conf box format
        conf = dts[:,4]*dts[:,5]
        detections = np.zeros((len(dts), 6))
        detections[:,:4] = bboxs
        detections[:,4] = rot
        detections[:,5] = conf

        return lmks, lmk_confs, detections

    def worker(self, input_q, output_q):
        while True:
            bgrimg, dts = input_q.get()
            lmks, lmk_confs, new_dts = self.SkeletonDetect(bgrimg, dts)
            output_q.put((bgrimg, new_dts, lmks, lmk_confs))

class FeatureEngine(object):
    def __init__(self, pr_model, LMK_VISIBILITY_THRESHOLD, device=-1):

        # Initializing person matching module
        self.body_emb_layer = "fc1_output"
        self.pr_size = 128
        self.LMK_VISIBILITY_THRESHOLD = LMK_VISIBILITY_THRESHOLD

        self.__ubody_embedding = UbodyEmbedding(model_path=pr_model,
                                             model_epoch=int(0),
                                             device=device,
                                             input_size=self.pr_size,
                                             emb_layer_name=self.body_emb_layer)

        # self.__fullbody_embedding = FullbodyEmbedding(model_name='osnet_ain_x1_0',
        #                                     model_path='./models/Body/osnet_ain_ms_d_c.pth.tar', 
        #                                     device=device)
        # mean body
        # self.mean_body = np.array([
        #         [28.0, 55.0],
        #         [100.0, 55.0],
        #         [28.0, 127.0],
        #         [100.0, 127.0]], dtype=np.float32)

        # self.mean_body = np.array([
        #         [32.0, 58.0],
        #         [96.0, 58.0],
        #         [32.0, 122.0],
        #         [96.0, 122.0]], dtype=np.float32)

        # #org mean body
        # self.mean_body = np.array([
        #         [64.0, 54.0],
        #         [64.0, 126.0]], dtype=np.float32)

        self.mean_body = np.array([
                [64.0, 50.0],
                [64.0, 127.0]], dtype=np.float32)

    def __crop_rot_upper_body(self, img, detections):

        imgh, imgw = img.shape[:2]

        chips = []
        flags = []
        for bb in detections:
            x,y,w,h,angle = bb[:5]
            #print(x,y,w,h,imgh,imgw)
            if self.fisheye:
                theta = -self.__get_angle(imgw, imgh, x, y)

                c, s = np.cos(theta), np.sin(theta)

                R = np.asarray([[c, s], [-s, c]])
                pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
                rot_pts = []
                for pt in pts:
                    rot_pts.append(([x, y] + pt @ R).astype(int))

                contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])

                width = int(w)
                height = int(h)
                src_pts = contours.astype("float32")
                dst_pts = np.array([[0, 0],[width-1, 0], [width-1, height-1],
                            [0, height-1]], dtype="float32")

                tform = transform.SimilarityTransform()
                tform.estimate(src_pts, dst_pts)
                M = tform.params[0:2, :]
                chip = cv2.warpAffine(img, M, (self.pr_size, self.pr_size))
                flags.append(True)
            else:
                x1, y1 = int(x-w/2), int(y-h/2)
                x2, y2 = int(x1+w), int(y1+h)
                if x1<0:
                    x1=0
                if y1<0:
                    y1=0
                if x2>imgw:
                    x2 = imgw
                if y2>imgh:
                    y2 = imgh

                ub_width = x2-x1
                chip = img[y1:y1+ub_width,x1:x2].copy()
                chip = cv2.resize(chip, (self.pr_size, self.pr_size))
                flags.append(False)

            chips.append(chip)

        return chips, flags

    def __crop_upper_body(self, img, bbox):
        chips = []
        (H, W) = img.shape[:2]
        for bb in bbox:
            # print("BOUNDING BOX",bb)
            (x1, y1) = (bb[0], bb[1])
            (x2, y2) = (bb[2], bb[2])
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if x2>W:
                x2 = W
            if y2>H:
                y2 = H

            ub_width = x2-x1
            chip = img[y1:y1+ub_width,x1:x2].copy()
            # outfile = "test_ub_{}.jpg".format(self.count)
            # cv2.imwrite(outfile, chip)
            chip = cv2.resize(chip, (self.pr_size, self.pr_size))
            chip = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
            chips.append(chip)

        return np.array(chips)

    def __crop_rot_upper_body_lmk(self, input_img, detections, landmarks, lmk_confs):

        img = input_img.copy()
        imgh, imgw = input_img.shape[:2]

        chips = []
        flags = []
        for i in range(len(detections)):
            lmk = landmarks[i]
            confs = lmk_confs[i]

            left_hip, left_hip_conf = lmk[11], confs[11]
            right_hip, right_hip_conf = lmk[12], confs[12]
            middle_of_hip = [(left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2]
            middle_of_hip_conf = 0 if left_hip_conf < self.LMK_VISIBILITY_THRESHOLD or right_hip_conf < self.LMK_VISIBILITY_THRESHOLD else 1

            left_shoulder, left_shoulder_conf = lmk[5], confs[5]
            right_shoulder, right_shoulder_conf = lmk[6], confs[6]
            middle_of_shoulder = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2]
            middle_of_shoulder_conf = 0 if left_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD or right_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD else 1

            if middle_of_shoulder_conf>self.LMK_VISIBILITY_THRESHOLD  and middle_of_hip_conf>self.LMK_VISIBILITY_THRESHOLD:
                #print(colored("CROPPING WITH LMK", "blue"))

                src_pts = np.array([[middle_of_shoulder[0],middle_of_shoulder[1]],
                       [middle_of_hip[0],middle_of_hip[1]]], dtype=np.float32)

                tform = transform.SimilarityTransform()
                tform.estimate(src_pts, self.mean_body)
                M = tform.params[0:2, :]
                warped = cv2.warpAffine(img, M, (self.pr_size, self.pr_size), borderValue=0.0)
                chips.append(warped)
                flags.append(True)
            else:
                bb = detections[i]
                x,y,w,h,angle = bb[:5]
                #print(colored("CROPPING WITH AFFINE", "red"))

                theta = -rot[i]

                c, s = np.cos(theta), np.sin(theta)

                R = np.asarray([[c, s], [-s, c]])
                pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, -h/2+w/2], [-w/2, -h/2+w/2]])
                rot_pts = []
                for pt in pts:
                    rot_pts.append(([x, y] + pt @ R).astype(int))

                contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])

                width = int(w)
                height = int(h)
                src_pts = contours.astype("float32")
                dst_pts = np.array([[0, 0],[self.pr_size-1, 0], [self.pr_size-1, self.pr_size-1],
                            [0, self.pr_size-1]], dtype="float32")

                tform = transform.SimilarityTransform()
                tform.estimate(src_pts, dst_pts)
                M = tform.params[0:2, :]
                chip = cv2.warpAffine(img, M, (self.pr_size, self.pr_size))
                chips.append(chip)
                flags.append(True)

        return chips, flags

    def BodyFeatureExtraction(self, bgrimg, dts, lmks, lmk_confs):
        t1 = time.time()
        rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        body_crops, flags = self.__crop_rot_upper_body_lmk(rgbimg, dts, lmks, lmk_confs)
        #extract body features
        body_features = self.__ubody_embedding.extract_feature_batch(body_crops)
        # body_features = self.__fullbody_embedding(body_crops)
        t2 = time.time() - t1
        #print("feature ectraction time ", t2)

        return body_crops, np.array(body_features)

    def worker(self, input_q, output_q):
        while True:
            bgrimg, dts, lmks, lmk_confs = input_q.get()
            body_crops, body_features = self.BodyFeatureExtraction(bgrimg, dts, lmks, lmk_confs)
            output_q.put((bgrimg, dts, lmks, lmk_confs, body_crops, body_features))

    
