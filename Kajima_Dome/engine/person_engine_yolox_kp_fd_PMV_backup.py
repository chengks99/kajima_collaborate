import pathlib
import sys
import time
import math
import cv2
import torch
import logging

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skimage import transform
from PIL import Image
from termcolor import colored

import visualization as vsl
from yolox_detector import YoloxPredictor
from embedding import FaceEmbedding, UbodyEmbedding, FullbodyEmbedding
from retinaface import RetinaFace
from person_tracker_PMV import DomeTracker

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'pose'))
from detect import HumanPoseDetection

sys.path.append(str(scriptpath.parent / 'thermalcomfort'))
from clothatt_prediction import ClothPredictor

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

class BGRColor:
    WHITE   = (255, 255, 255)
    SILVER  = (192, 192, 192)
    GRAY    = (128, 128, 128)
    GREY    = (128, 128, 128)
    BLACK   = (0, 0, 0)

    BLUE    = (255, 0, 0)
    NAVY    = (128, 0, 0)
    LIME    = (0, 255, 0)
    GREEN   = (0, 128, 0)
    RED     = (0, 0, 255)
    MAROON  = (0, 0, 128)

    YELLOW  = (0, 255, 255)
    OLIVE   = (0, 128, 128)
    GOLD    = (0, 215, 255)

    CYAN    = (255, 255, 0)
    AQUA    = (255, 255, 0)
    TEAL    = (128, 128, 0)

    MAGENTA = (255, 0, 255)
    FUCHSIA = (255, 0, 255)
    PURPLE  = (128, 0, 128)

    CUSTOM_COLOR = (35, 142, 107)


def draw_box(image, box, label='', color=BGRColor.CYAN, font_size=0.7):
    """
        image: image matrix (h, w, c)
        box: numpy array of shape (n x 5). [x1, x2, y1, y2, score]
        label: text to put above box
        color: BGR colour value for box and text label colour
        font_size: Size of text label
    """
    img = image.copy()

    #RAPID det box: cx, cy, w, h, angle, conf

    x1 = int(box[0] - box[2]/2)
    y1 = int(box[1] - box[3]/2)

    # bbox = list(map(int,box))
    # Draw bounding box and label:
    # label = '{}_{}'.format(label, bbox[2])
    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=color, thickness=2)

    textsize = cv2.getTextSize(label, 0, font_size, thickness=3)[0]
    print("FONT SIZE ", textsize)
    cv2.rectangle(img, (x1, y1-5-textsize[1]), (x1+textsize[0], y1-5), color=BGRColor.BLUE, thickness=-1)
    cv2.putText(img, label, (x1, y1 - 5), fontFace=0, fontScale=font_size, color=BGRColor.YELLOW, thickness=3)
    cv2.putText(img, label, (x1, y1 - 5), fontFace=0, fontScale=font_size, color=color, thickness=2)

    return img

def draw_box_batch_org(image, bboxs, tids, face_labels):
    # label = "tid_"+str(tids)+"_"+face_labels 
    label = face_labels 
    if face_labels == 'UnK':
        box_color = BGRColor.RED
    else:
        box_color = BGRColor.WHITE

    image = draw_box(image, bboxs[0], label, color=box_color, font_size=1.0)

    return image

def draw_box_batch(image, bboxs, tids, face_labels, pmvs):
    # label = "tid_"+str(tids)+"_"+face_labels 
    print("LABELS ", face_labels)
    for i in range(len(face_labels)):
        label = "{}_pmv_{}".format(face_labels[i], pmvs[i])
        if label == 'UnK':
            box_color = BGRColor.RED
        else:
            box_color = BGRColor.WHITE

        image = draw_box(image, bboxs[i], label, color=box_color, font_size=1.0)

    return image

class PersonEngine(object):
    def __init__(self, cfg_person, cfg_cam, face_details, body_details, redis_conn) -> None:
        self.redis_conn = redis_conn
        self.face_details = face_details
        self.body_details = body_details

        mPath = cfg_person.get('model_path', {})
        bInfo = cfg_person.get('body_model_info', {})
        pInfo = cfg_person.get('pmv_model_info', {})
        fInfo = cfg_person.get('face_model_info', {})

        self.min_person_width = bInfo.get('min_person_width', 100)
        self.frame_num = 0
        self.fisheye = cfg_cam.get('fisheye', 0)
        self.LMK_VISIBILITY_THRESHOLD = 0.85
        self.pr_threshold = bInfo.get('pr_threshold', 0.3)

        self.cam_roi = [
            np.array(cfg_cam.get('cam1_roi', []), dtype=np.int).reshape(4, -1),
            np.array(cfg_cam.get('cam2_roi', []), dtype=np.int).reshape(4, -1),
        ]

        self.detection_engine = DetectionEngine(
            pd_model=mPath.get('pd_model', ''),
            pd_det_threshold=bInfo.get('pd_det_threshold', 0.5),
            pd_nms_threshold=bInfo.get('pd_nms_threshold', 0.3),
            pd_input_resize=bInfo.get('pd_input_resize', 0),
            max_detected_persons=bInfo.get('max_detected_person', 0),
            min_person_width=bInfo.get('min_person_width', 100),
            device=bInfo.get('device', 0)
        )
        logging.debug('Detection Engine loaded ...')

        self.skeleton_engine = SkeletonEngine(
            kp_model=mPath.get('kp_model', ''),
            kp_cfg=mPath.get('kp_cfg', ''),
            LMK_VISIBILITY_THRESHOLD=self.LMK_VISIBILITY_THRESHOLD,
            device=bInfo.get('device', 0),
            camroi=self.cam_roi
        )
        logging.debug('Skeleton Engine loaded ...')

        self.feature_engine = FeatureEngine(
            pr_model=mPath.get('pr_model', ''),
            fr_model=mPath.get('fr_model', ''),
            cloth_model=mPath.get('cloth_model', ''),
            LMK_VISIBILITY_THRESHOLD=0,
            device=bInfo.get('device', 0)
        )
        logging.debug('Feature Engine loaded ...')

        env = {'tr': pInfo.get('radiant_temp', 22.6), 'tdb': pInfo.get('room_temp', 22.6), 'to': pInfo.get('room_temp', 22.6), 'rh': pInfo.get('rel_humidity', 57.8), 'v': pInfo.get('air_speed', 0.1)}
        self.FaceBodyTracker = DomeTracker(
            pr_threshold=self.pr_threshold,
            fr_threshold=fInfo.get('fr_threshold', 0.4),
            face_db_file=fInfo.get('face_database', ''),
            body_db_file=bInfo.get('body_database', ''),
            body_feat_life=bInfo.get('body_feat_life', 32400),
            pmv_model=mPath.get('pmv_model', ''),
            env_variables=env,
            face_details=self.face_details,
            # self.body_details=body_details,
            body_details=self.body_details,
            redis_conn=self.redis_conn
        )
        logging.debug('Face body Tracker loaded ...')

    def person_face_updates (self, face_details):
        self.face_details = face_details
        self.FaceBodyTracker.face_updates(face_details)
    
    def person_body_updates (self, body_details):
        self.body_details = body_details
        self.FaceBodyTracker.body_updates(body_details)

    def YoloxPersonBoxTrackMatchFeatureBatch(self, bgrimgs):

        self.frame_num += 1

        t1 = time.time()
        org_dts = self.detection_engine.YoloxPersonDetectBatch(bgrimgs)

        t2 = time.time() - t1
        print("detections time ", t2)

        # if org_dts is None:
        #     return None

        # for i in range(len(bgrimgs)):          
        #     visualization.draw_dt_on_np(bgrimgs[i], org_dts[i], color=(255,255,0))
        #     # bgrimgs[i] = visualization.draw_yolox(bgrimgs[i], org_dts[i])

        # person should be visible in both cameras
        if org_dts[0] is not None and org_dts[1] is not None:
            # t1 = time.time()
            lmks, lmk_confs, roi_dts = self.skeleton_engine.SkeletonDetectBatch(bgrimgs, org_dts)

            body_crops, body_feats, face_crops, face_feats, face_confs, cloth_att, actions = self.feature_engine.FeatureExtractionBatch(bgrimgs, roi_dts, lmks, lmk_confs)
            print("Face Confs ", face_confs)
            trackids, subids, pmvs = self.FaceBodyTracker.update(roi_dts, lmks, lmk_confs, body_crops, body_feats, face_crops, face_feats, face_confs, cloth_att)

            print("TRACK ids label ", trackids, subids)
            # print(len(body_crops), len(body_feats), len(face_crops), len(face_feats), len(face_confs))
 
            for i in range(len(bgrimgs)):          
                # detections = dts[i]

                visualization.draw_dt_on_np(bgrimgs[i], org_dts[i], color=(255,0,255))
                visualization.draw_dt_on_np(bgrimgs[i], roi_dts[i])

                # lmk = srtd_lmks[i]
                # lmk_conf = srtd_lmk_confs[i]
                # visualization.draw_lmks(bgrimgs[i], lmks[i], lmk_confs[i], (255,255,255), self.LMK_VISIBILITY_THRESHOLD)
                visualization.draw_lmks(bgrimgs[i], lmks[i], lmk_confs[i], (255,255,255), 0)

                # for k in range(len(lmk)):
                #     self.__draw_lmks(bgrimgs[i], lmk[k], lmk_conf[k], (255,255,255))

                if len(roi_dts)>0:
                    bgrimgs[i] = draw_box_batch(bgrimgs[i], roi_dts[i], trackids[i], subids[i], pmvs[i])
                # bgrimgs[i] = draw_box_batch_org(bgrimgs[i], bboxs, trackids, subids)

                cv2.polylines(bgrimgs[i], [self.cam_roi[i]], True, color=BGRColor.CYAN, thickness=2)


            for i in range(len(bgrimgs)):          
                cv2.polylines(bgrimgs[i], [self.cam_roi[i]], True, color=BGRColor.CYAN, thickness=2)
        else:
            for i in range(len(bgrimgs)):          
                cv2.polylines(bgrimgs[i], [self.cam_roi[i]], True, color=BGRColor.CYAN, thickness=2)
            self.FaceBodyTracker.Increment_track_age()
            return None

        # print("Face Conf ", face_conf0, face_conf1)

        return roi_dts, body_crops, subids, trackids, lmks, bgrimgs, pmvs, actions
        # # return srtd_dts
        # return org_dts

class DetectionEngine(object):
    def __init__(self, pd_model, pd_det_threshold=0.5, pd_nms_threshold=0.3, pd_input_resize=0, 
        max_detected_persons=0, min_person_width=100, device=-1):

        self.min_person_width = min_person_width
        # self.iou_th = 0.6 # 0.35 #
        # self.iou_th_x = 0.75

        print(pd_model, pd_det_threshold, pd_nms_threshold, pd_input_resize, max_detected_persons, min_person_width, device)

        self.__detector_yolox = YoloxPredictor(model_name='yolox-l', model_weight=pd_model, 
            test_size=pd_input_resize, test_conf=pd_det_threshold, 
            nmsthre=pd_nms_threshold, maxppl=max_detected_persons, device=device)

    def YoloxPersonDetect(self, bgrimg):
        # rgbimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB)
        # pilimg = Image.fromarray(rgbimg)

        t1 = time.time()
        #output: in darknet format (cx,cy,w,h)
        detections = self.__detector_yolox.inference(bgrimg)

        # #filter out small box detections
        dts = self.__filterDts(detections)

        t2 = time.time() - t1
        print("YoloX Detection time ", t2)

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
        print("YoloX Detection time batch", t2)

        return dts

    def __filterDts(self, dts):
        # detections = dts
        # #filter out detection which have width above threshold
        # detections = detections[detections[:,2] >= self.min_person_width]

        if dts is None:
            return None 

        dts = np.array(dts)
        dts = dts[dts[:,3]>=self.min_person_width]

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
            bgrimgs = input_q.get()
            dts = self.YoloxPersonDetectBatch(bgrimgs)
            if len(dts) == 0:
                continue
            output_q.put((bgrimgs, dts))
            # queue.task_done() # this is new 

class SkeletonEngine(object):
    def __init__(self, kp_model, kp_cfg, LMK_VISIBILITY_THRESHOLD, device=-1, camroi=-1):

        # print("initializing lmk model", kp_model, kp_cfg)
        self.kp = HumanPoseDetection(kp_model, kp_cfg, device)
        self.camroi = camroi
        self.LMK_VISIBILITY_THRESHOLD = LMK_VISIBILITY_THRESHOLD
        self.iou_th = 0.6 # 0.35 # 
        self.iou_th_x = 0.75

    def __get_angle_batch(self, imgw, imgh, detections):
        dir = []
        for bb in detections:
            x,y,w,h,angle = bb[:5]
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

        #detect body landmarks
        lmks, lmk_confs = self.kp.detect_batch(bgrimg, bboxs)
        #crop upper body
        t2 = time.time() - t1
        print("Skeleton Detection time ", t2)

        return lmks, lmk_confs

    def SkeletonDetectBatch(self, bgrimgs, detections):
        t1 = time.time()
        lmks = []
        lmk_confs = []
        new_dts = []

        #loop over two images
        for i in range(len(detections)):
            dts = detections[i]
            # print("BEFORE ", dts)

            #filter out overlapping boxes
            flt_dts = self.__filterOverlapBox(dts)
            if not len(flt_dts):
                lmks.append([])
                lmk_confs.append([])
                new_dts.append([])
                continue

            # dts = np.array(dts)
            # imgh, imgw = bgrimg.shape[:2]

            bboxs = flt_dts[:,:4]
            # bboxs = dts[:,:4]
            # angles = dts[:,4:5]
            # print("ANGLES", angles)            

            #get box angles
            # rot = self.__get_angle_batch(imgw, imgh, dts)
            #detect body landmarks
            img_lmk, img_lmk_conf = self.kp.detect_batch(bgrimgs[i], bboxs)

            # #filter out skeletons outside roi for each camera
            filter_lmks, filter_lmk_confs, filter_dts = self.__filterLmksBatch(img_lmk, img_lmk_conf, dts, self.camroi[i])

            # print("AFTER ", filter_dts)

            lmks.append(filter_lmks)
            lmk_confs.append(filter_lmk_confs)
            new_dts.append(filter_dts)

            # lmks.append(img_lmk)
            # lmk_confs.append(img_lmk_conf)
            # new_dts.append(dts)

        t2 = time.time() - t1
        print("Skeleton Detection time batch ", t2)

        # return lmks, lmk_confs, new_dts
        return lmks, lmk_confs, new_dts

    def __filterOverlapBox(self, dts):
        #filter out strong overlapping bbox 

        filter_detection = []

        iou_img = [] 
        for i in range(len(dts)):
            bb1 = dts[i]
            #convert cxcywh to x1y1x2y2 format
            bbox1 = (bb1[0]-bb1[2]/2, bb1[1]-bb1[3]/2, bb1[0]+bb1[2]/2, bb1[1]+bb1[3]/2) 

            #calculate sum of iou of current box with other boxes
            bb_iou = 0
            for j in range(len(dts)):
                if i==j:
                    continue
                bb2 = dts[j]
                
                #convert cxcywh to x1y1x2y2 format
                bbox2 = (bb2[0]-bb2[2]/2, bb2[1]-bb2[3]/2, bb2[0]+bb2[2]/2, bb2[1]+bb2[3]/2)

                bb_iou += get_overlap(bb1, bb2)

            iou_img.append(bb_iou) #np.sum(iou_array)
            #filter out boxes based on iou overlap
            if bb_iou<self.iou_th:
                filter_detection.append(dts[i])

        # print("iou img ", len(dts), iou_img)
        # print("org ", dts)
        # print("filter ", filter_detection)

        return np.array(filter_detection)

    def __filterLmks(self, lmk, roi):
        # The function returns +1, -1, or 0 to indicate if a point is inside, outside, or on the contour, respectively
        landmarks = lmk
        result = cv2.pointPolygonTest(roi, tuple(landmarks[15]), False) 
        if result==-1:
            result = cv2.pointPolygonTest(roi, tuple(landmarks[16]), False) 

        # print(landmarks, roi, result)
        return result

    def __filterLmksBatch(self, lmks, lmk_confs, dts, camroi):
        # 15: "left_ankle",
        # 16: "right_ankle"

        filter_lmks = []
        filter_lmk_confs = []
        filter_dts = []
        for i in range(len(lmks)):
            # if lmks[i] is None:
            #     continue;
            result = self.__filterLmks(lmks[i], camroi)
            if result>=0:
                #check visibility of nose and eyes lmks
                #if above threshold, means frontal face
                conf = lmk_confs[i]
                if conf[0]<self.LMK_VISIBILITY_THRESHOLD or conf[1]<self.LMK_VISIBILITY_THRESHOLD or conf[2]<self.LMK_VISIBILITY_THRESHOLD:
                    continue

                filter_lmks.append(lmks[i])
                filter_lmk_confs.append(lmk_confs[i])
                filter_dts.append(dts[i])

        return filter_lmks, filter_lmk_confs, filter_dts

    def worker(self, input_q, output_q):
        while True:
            bgrimgs, detections = input_q.get()
            lmks, lmk_confs, roi_dts = self.SkeletonDetectBatch(bgrimgs, detections)
            output_q.put((bgrimgs, roi_dts, lmks, lmk_confs))

class FeatureEngine(object):
    def __init__(self, pr_model, fr_model, cloth_model, LMK_VISIBILITY_THRESHOLD, device=-1):

        # Initializing person matching module
        self.body_emb_layer = "fc1_output"
        self.pr_size = 128
        self.LMK_VISIBILITY_THRESHOLD = LMK_VISIBILITY_THRESHOLD

        self.__ubody_embedding = UbodyEmbedding(model_path=pr_model,
                                             model_epoch=int(0),
                                             device=device,
                                             input_size=self.pr_size,
                                             emb_layer_name=self.body_emb_layer)

        self.__fullbody_embedding = FullbodyEmbedding(model_name='osnet_ain_x1_0',
                                            model_path='./models/Body/osnet_ain_ms_d_c.pth.tar', 
                                            device=device)

        self.cloth_predictor = ClothPredictor(cloth_model, device)

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

        # # Initializing face matching module
        self.fr_size = 112
        self.fd_size = 144

        # fd mean face
        self.fd_mean_face = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366]], dtype=np.float32)
        self.fd_mean_face[:, 0] += 8.0
        self.fd_mean_face[:, :] += 16.0 #offset for 144 size for x and y (32/2)

        # fr mean face
        self.fr_mean_face = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
        self.fr_mean_face[:, 0] += 8.0

        self.face_emb_layer = "fc1_output"
        self.__face_embedding = FaceEmbedding(model_path=fr_model,
                                             model_epoch=int(0),
                                             device=device,
                                             input_size=self.fr_size,
                                             emb_layer_name=self.face_emb_layer)

        self.fd_model_path = './models/Face/mnet1.0_NoFixSt'
        self.fd_threshold = 0.5
        self.fd_input_resize = 64
        self.__detector = RetinaFace(self.fd_model_path, 0, device, 'net3')

        #load action recognition model
        model_locatioon = "./models/action_jk.pth"

        if device<0:
            self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cuda:{}".format(device))
        
        self.actionNet = Net()
        self.actionNet.load_state_dict(torch.load(model_locatioon))
        # self.actionNet = torch.load(model_locatioon)

        self.actionNet = self.actionNet.to(self.device) 
        self.actionNet.eval()       


    def __get_scale(self, img, target_size):
        MAX_SIZE = 1200

        if target_size == 0:
            return 1.0

        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])

        # if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)

        # Prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > MAX_SIZE:
            im_scale = float(MAX_SIZE) / float(im_size_max)

        # print("img shape: {}, min: {}, max: {}".format(img.shape, im_size_min, im_size_max))
        # print('target_size:', target_size)
        # print('MAX_SIZE:', MAX_SIZE)
        # print('im_scale:', im_scale)

        # im_scale = 1.0
        return im_scale

    def __detect_faces(self, img, max_faces=0):
        """
        Based on RetinaFace, which does image scaling and box rescaling
        within its detect() function.
        """
        im_scale = self.__get_scale(img, target_size=self.fd_input_resize)
        boxes, landmarks = self.__detector.detect(img, threshold=self.fd_threshold, scales=[im_scale], do_flip=False)

        if boxes is None or boxes.shape[0] == 0:
            return None

        # print("boxes:\n{}\n{}".format(boxes, boxes.shape))
        # print("landmarks:\n{}\n{}".format(landmarks, landmarks.shape))

        if max_faces > 0:
            boxes = boxes[:max_faces]
            landmarks = landmarks[:max_faces]

        return boxes, landmarks

    def __extract_face_features(self, imgs):
        # Batch feature extraction:
        features = self.__face_embedding.extract_feature_batch(imgs)
        return np.array(features)

    def __align_and_crop_single_fd(self, input_img, landmark):
        img = input_img.copy()

        dst = np.array([[landmark[2][0],landmark[2][1]],
               [landmark[1][0],landmark[1][1]],
               [landmark[0][0],landmark[0][1]]], dtype=np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(dst, self.fd_mean_face)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(img, M, (self.fd_size, self.fd_size), borderValue=0.0)
        return warped

    def __align_and_crop_single_fr(self, input_img, landmark):
        img = input_img.copy()

        dst = landmark

        tform = transform.SimilarityTransform()
        tform.estimate(dst, self.fr_mean_face)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(img, M, (self.fr_size, self.fr_size), borderValue=0.0)
        return warped

    def __align_and_crop_fd(self, img, landmark, lmk_conf):
        chip = self.__align_and_crop_single_fd(img, landmark)            
        face_conf = lmk_conf[2]+lmk_conf[1]+lmk_conf[0]

        return chip, np.array(face_conf)

    def __align_and_crop_fr(self, img, landmarks):
        chips = []
        chip = self.__align_and_crop_single_fr(img, landmarks)            
        chips.append(chip)

        return np.array(chips)

    def __crop_rot_upper_body(self, img, detections):

        imgh, imgw = img.shape[:2]

        chips = []
        flags = []
        for bb in detections:
            x,y,w,h,angle = bb[:5]
            print(x,y,w,h,imgh,imgw)
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

    def __crop_rot_upper_body_lmk(self, input_img, detection, landmark, lmk_conf):

        img = input_img.copy()
        imgh, imgw = input_img.shape[:2]

        flag = True

        lmk = landmark
        confs = lmk_conf

        left_hip, left_hip_conf = lmk[11], confs[11]
        # right_hip, right_hip_conf = lmk[12], confs[12]
        middle_of_hip = [(left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2]
        # middle_of_hip_conf = 0 if left_hip_conf < self.LMK_VISIBILITY_THRESHOLD or right_hip_conf < self.LMK_VISIBILITY_THRESHOLD else 1

        left_shoulder, left_shoulder_conf = lmk[5], confs[5]
        # right_shoulder, right_shoulder_conf = lmk[6], confs[6]
        middle_of_shoulder = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2]
        # middle_of_shoulder_conf = 0 if left_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD or right_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD else 1

        print(colored("CROPPING WITH LMK", "blue"))

        src_pts = np.array([[middle_of_shoulder[0],middle_of_shoulder[1]],
               [middle_of_hip[0],middle_of_hip[1]]], dtype=np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(src_pts, self.mean_body)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(img, M, (self.pr_size, self.pr_size), borderValue=0.0)
        chip = warped

        return chip, flag

    def __crop_rot_upper_body_lmk_batch(self, input_img, detections, landmarks, lmk_confs):

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
            # middle_of_hip_conf = 0 if left_hip_conf < self.LMK_VISIBILITY_THRESHOLD or right_hip_conf < self.LMK_VISIBILITY_THRESHOLD else 1

            left_shoulder, left_shoulder_conf = lmk[5], confs[5]
            right_shoulder, right_shoulder_conf = lmk[6], confs[6]
            middle_of_shoulder = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2]
            # middle_of_shoulder_conf = 0 if left_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD or right_shoulder_conf < self.LMK_VISIBILITY_THRESHOLD else 1

            print(colored("CROPPING WITH LMK", "blue"))

            src_pts = np.array([[middle_of_shoulder[0],middle_of_shoulder[1]],
                   [middle_of_hip[0],middle_of_hip[1]]], dtype=np.float32)

            tform = transform.SimilarityTransform()
            tform.estimate(src_pts, self.mean_body)
            M = tform.params[0:2, :]
            warped = cv2.warpAffine(img, M, (self.pr_size, self.pr_size), borderValue=0.0)
            chips.append(warped)
            flags.append(True)

        return chips, flags

    def __extract_face_features(self, crops):
        # Batch feature extraction:
        features = self.__face_embedding.extract_feature_batch(crops)
        return np.array(features)

    def __extract_body_features(self, crops):
        features = self.__ubody_embedding.extract_feature_batch(crops)
        # features = self.__fullbody_embedding(crops)
        return np.array(features)

    def FeatureExtractionBatch(self, bgrimgs, detections, lmks, lmk_confs):
        t1 = time.time()
        body_crops = []
        face_crops = []
        face_confs = []
        body_feats = []
        face_feats = []
        cloth_att = []
        actions = [] #not used currently

        #loop over two cameras
        for i, dts in enumerate(detections):

            # bboxs = dts[:,:4]
            lmk = lmks[i]
            lmk_conf = lmk_confs[i]

            rgbimg = cv2.cvtColor(bgrimgs[i], cv2.COLOR_BGR2RGB)

            img_body_crops, flags = self.__crop_rot_upper_body_lmk_batch(rgbimg, dts, lmk, lmk_conf)
            body_crops.append(img_body_crops)

            img_face_crops = []
            img_face_confs = []   
            for k in range(len(lmk)):
                fd_face_crop, fd_face_conf = self.__align_and_crop_fd(rgbimg, lmk[k], lmk_conf[k])
                # cv2.imwrite("test_face.jpg", fd_face_crop)

                detector_output = self.__detect_faces(fd_face_crop, max_faces=1)
                if detector_output is None:
                    chips = []
                    chip = fd_face_crop[16:128, 16:128, :]
                    chips.append(chip)
                    fr_face_crop = np.array(chips)
                    # print("NONE ", fr_face_crop.shape)
                else:
                    boxes, face_lmk = detector_output  # Unpack return value
                    fr_face_crop = self.__align_and_crop_fr(fd_face_crop, face_lmk[0])
                    # print(fr_face_crop.shape, boxes, face_lmk)
                    # cv2.imwrite("test_face.jpg", fr_face_crop)

                img_face_crops.extend(fr_face_crop)
                img_face_confs.extend(fd_face_conf)

            #extract body features
            body_features = []
            if len(img_body_crops)>0:
                body_features = self.__extract_body_features(img_body_crops)
            body_feats.append(body_features)

            #extract face features
            face_features = []
            if len(img_face_crops)>0:
                face_features = self.__extract_face_features(img_face_crops)
            face_feats.append(face_features)

            face_crops.append(img_face_crops)
            face_confs.append(img_face_confs)

            #extract cloth attributes
            cloth_attributes = self.cloth_predictor.model_inference(bgrimgs[i], lmk, lmk_conf)
            cloth_att.append(cloth_attributes)

            # #estimate actions
            img_dim = bgrimgs[i].shape[:2]
            human_actions = self.__action_recognition(lmk, img_dim)
            actions.append(human_actions)

        t2 = time.time() - t1
        print("feature ectraction time ", t2)

        return body_crops, body_feats, face_crops, face_feats, face_confs, cloth_att, actions

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

        actions = []
        if len(polar_lmks)>0:
            with torch.no_grad():
                inputs = torch.FloatTensor(polar_lmks)
                inputs = inputs.cuda(self.device)
                outputs = self.actionNet(inputs)

                results = torch.argmax(outputs, dim=1)
                actions = results.cpu().numpy()
                print("action ", actions)

        # results = list(self.flatten(results))

        return actions

    def FeatureExtractionBatch_org(self, bgrimgs, detections, lmks, lmk_confs):
        t1 = time.time()
        body_crops = []
        face_crops = []
        face_confs = []
        for i, dts in enumerate(detections):
            # bboxs = dts[:,:4]
            lmk = lmks[i]
            lmk_conf = lmk_confs[i]

            rgbimg = cv2.cvtColor(bgrimgs[i], cv2.COLOR_BGR2RGB)

            body_crop, flags = self.__crop_rot_upper_body_lmk_batch(rgbimg, dts, lmk, lmk_conf)
            body_crops.extend(body_crop)

            fd_face_crop, face_conf = self.__align_and_crop_fd(rgbimg, lmk[0], lmk_conf[0])
            print("FD size ", fd_face_crop.shape)
            detector_output = self.__detect_faces(fd_face_crop, max_faces=1)
            if detector_output is None:
                chips = []
                chip = fd_face_crop[16:128, 16:128, :]
                chips.append(chip)
                fr_face_crop = np.array(chips)
                print("NONE ", fr_face_crop.shape)
            else:
                boxes, landmarks = detector_output  # Unpack return value
                fr_face_crop = self.__align_and_crop_fr(fd_face_crop, landmarks[0])
                print(fr_face_crop.shape, boxes, landmarks)

            face_crops.extend(fr_face_crop)
            face_confs.extend(face_conf)

        #extract body features
        body_features = self.__extract_body_features(body_crops)
        face_features = self.__extract_face_features(face_crops)

        t2 = time.time() - t1
        print("feature ectraction time ", t2)
        print(face_features)

        return body_crops, body_features, face_crops, face_features, face_confs

    def worker(self, input_q, output_q):
        while True:
            bgrimgs, dts, lmks, lmk_confs = input_q.get()
            body_crops, body_features, face_crops, face_features, face_confs, cloth_att, actions = self.FeatureExtractionBatch(bgrimgs, dts, lmks, lmk_confs)
            output_q.put((bgrimgs, dts, lmks, lmk_confs, body_crops, body_features, face_crops, face_features, face_confs, cloth_att, actions))

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou

def get_iou_x(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ix0min = min(pred_box[0], gt_box[0])
    ix0max = max(pred_box[0], gt_box[0])
    ix1min = min(pred_box[2], gt_box[2])
    ix1max = max(pred_box[2], gt_box[2])

    w0 = pred_box[2] - pred_box[0]
    w1 = gt_box[2] - gt_box[0]

    # iymin = max(pred_box[1], gt_box[1])
    # iymax = min(pred_box[3], gt_box[3])

    # iw = np.maximum(ixmax-ixmin+1., 0.)
    # ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    # inters = iw*ih

    # 3. calculate the area of union
    # uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
    #        (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
    #        inters)

    # 4. calculate the overlaps between pred_box and gt_box
    # iou = inters / uni
    # iou = (ixmax-ixmin) / (ixmax+ixmin)
    # iou = (ix1min - ix0max)/(ix1max + ix0min)
    iou = (ix1min - ix0max)/(max(w0, w1))

    # print("min-max\n", ix1min, ix0max, w0, w1)

    return iou

def get_overlap(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area pred box
    area = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.))

    # # 4. calculate the overlaps between pred_box and gt_box
    # iou = inters / uni

    # 5. calculate the overlaps with respect to own box area
    iou = inters / area

    return iou