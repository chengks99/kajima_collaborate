import os
import logging
# import base64

import cv2
import numpy as np
from skimage import transform
from sklearn.metrics.pairwise import cosine_similarity


import time
import math
import pandas as pd
from termcolor import colored

#related to thermal comfort
from pykalman import KalmanFilter
from xgboost.sklearn import XGBRegressor

import pathlib
import sys

scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'common'))
from database import FaceDatabasePersonal as FaceDatabase
from database import BodyDatabaseFront as BodyDatabase
from jsonutils import json2str

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

class DomeTracker:
    def __init__(self, pr_threshold, fr_threshold, face_db_file, body_db_file, body_feat_life, pmv_model, env_variables, face_details=None, body_details=None, redis_conn=None):
        """
        initialize the next unique object ID to keep track of mapping 
        a given object ID to its centroid 
        """
        self.redis_conn = redis_conn

        # Initializing face matching module
        self.LMK_VISIBILITY_THRESHOLD = 0.8
        # self.tr_threshold = 0.7
        self.pr_threshold = pr_threshold
        self.fr_threshold = fr_threshold

        # self.width_threshold = 100#150
        # self.norm_threshold = 10
        # self.area_threshold = 20000
        # self.angle_threshold = 50#10
        self.body_db_file = body_db_file
        self.face_db_file = face_db_file
        self.face_details = face_details
        self.body_details = body_details

        # print("Face DATABASE", face_db_file)
        # print(self.face_details['fvList'])
        if face_details is None:
            self.face_database, self.face_db_features, self.face_db_empty_flag = InitFaceDatabase(db_file=self.face_db_file)
        else:
            self.face_database, self.face_db_features, self.face_db_empty_flag = InitFaceDatabase(db_dict=self.face_details['fvList'])

        if not self.face_db_empty_flag:
            print("LOADED face DB shape ", self.face_db_features.shape)
            import time
            self.old_face_db_time = time.time()
        else:
            self.old_face_db_time = -1.0
            print("LOADED empty face DB shape ")

        self.body_feat_life = body_feat_life
        if body_details is None:
            self.body_database = BodyDatabase(db_file=self.body_db_file, body_feat_life=self.body_feat_life)
        else:
            self.body_database = BodyDatabase(db_dict=self.body_details['fvList'], body_feat_life=self.body_feat_life)

        self.nextObjectID = 0
        self.track_age = 10 #5
        self.max_track_age = 100 #float('inf') #10 #1000
        self.max_track_len = 1
        # self.iou_th = 0.35
        # self.iou_th_x = 0.75

        #thermal comfort model
        self.tc_predictor = ThermalComfort(pmv_model, env_variables)    

        # print("Loaded Thermal Comfort")
        self.columns = [
            'track_id', 'sub_id', 'rects', 'lmks', 'lmk_confs', 'lastbodyfeat', 'avgbodyfeat', 'face_feat',
            'len', 'body_crop', 'face_crop', 'time_stamp', 'age', 'db_flag', 'person_details',
            'iclo', 'iclo_kalman', 'pmv', 'pmv_kalman'
        ]

        self.tracks = pd.DataFrame(columns=self.columns)  # The "database"

        # # self.body_db = pd.DataFrame(columns=['name', 'featuref', 'featureb'])
        # self.body_db = pd.DataFrame(columns=['name', 'time_stamp', 'features'])

        # self.body_database, self.body_db_features, self.body_db_empty_flag = InitBodyDatabase(self.body_db_file)
        # if not self.body_db_empty_flag:
        #     self.body_old_db_time = os.path.getmtime(self.face_db_file)
        # else:
        #     self.body_old_db_time = -1.0

        # print("Dome Tracked finished initialize")

    def face_updates (self, face_details):
        self.face_details = face_details
        self.face_database, self.face_db_features, self.face_db_empty_flag = InitFaceDatabase(db_dict=self.face_details['fvList'])

    def body_updates (self, body_details):
        self.body_details = body_details
        self.body_database = BodyDatabase(db_dict=self.body_details['fvList'], body_feat_life=self.body_feat_life)

    def __save_database(self, save_file=None):
        # Commit the current self.data to a .pkl file.
        #select old enough and long enough tracks to write to database

        # rows_to_match = self.tracks.loc[(self.tracks['age'] > self.track_age) & (self.tracks['len']>7) & (self.tracks['db_flag']==0)]
        rows_to_match = self.tracks.loc[(self.tracks['age'] > self.track_age) & (self.tracks['len']>self.max_track_len) & (self.tracks['db_flag']==0)]
        idx_to_match = rows_to_match.index.tolist()

        if idx_to_match:
            for idx in idx_to_match:
                sub_id = self.tracks.at[idx, 'sub_id']
                if "Unk" in sub_id:
                    continue

                print("FOUND track to save ", idx)
                track_id = self.tracks.at[idx, 'track_id']
                avg_feat = self.__get_body_avg_feat(track_id)
                person_details = self.tracks.at[idx, 'person_details']
                self.tracks.at[idx, 'db_flag'] = 1

                self.body_database.add_data(sub_id, avg_feat, person_details)

                print(colored("SUBID ADDED DATABASE", "red"), sub_id, self.tracks.at[idx, 'len'])

                # if self.database.is_name_in_database(text):
                # self.database.add_data(text, feature_vec)

                # print("Removing track id ", idx)
                # self.tracks = self.tracks.drop(index=idx)
                self.redis_conn.publish('person.body.notify', json2str({'features': avg_feat.tolist(), 'name': sub_id, 'person_details': {key: int(value) for key, value in person_details.items()}}))
                # logging.info(json2str({'features': avg_feat.tolist(), 'name': sub_id, 'person_details': {key: int(value) for key, value in person_details.items()}}))
            # if not self.body_details is None:
            #     self.body_database.save_database()
            # else:
            #     self.redis_conn.publish('person.body.notify', json2str({'features': avg_feat, 'name': sub_id, 'person_details': person_details}))
        # else:
        # return string

    def __register_track(self, subid, rects, lmks, lmk_confs, body_cropf, bodyfeatf, face_cropf, facefeatf, person_details, cloth_attf):
        """
        add centroid and feature to tracking table
        """
        print(colored("TRACK ADDED {}".format(self.nextObjectID),'red'))
        feat_norm = np.linalg.norm(facefeatf)
        print("FACE FEAT NORM ", feat_norm)
        
        # #check visibility of eyes and nose
        # match_flag = -1
        # if lmk_conf[0]>self.LMK_VISIBILITY_THRESHOLD and lmk_conf[1]>self.LMK_VISIBILITY_THRESHOLD and lmk_conf[2]>self.LMK_VISIBILITY_THRESHOLD:
        #     match_flag = 0
        
        pmv = self.tc_predictor.model_inference(cloth_attf['iclo'], person_details)
        print("PERSON THERMAL ", person_details)

        row = {
            'track_id': self.nextObjectID,
            'sub_id': subid,
            'rects': rects,
            'lmks': lmks,
            'lmk_confs': lmk_confs,
            'lastbodyfeat': bodyfeatf,
            'avgbodyfeat': bodyfeatf,
            'len': 1,
            'body_crop': body_cropf,
            'face_crop': face_cropf,
            'face_feat': facefeatf,
            'age': 0,
            'db_flag': 0,
            'person_details': person_details,
            'cloth_att': cloth_attf,
            'iclo': cloth_attf['iclo'],
            'pmv': pmv,
            'iclo_kalman': KF(cloth_attf['iclo']),
            'pmv_kalman': KF(pmv)
        }
        self.tracks = self.tracks.append(row, ignore_index=True)
        self.nextObjectID += 1
        
        if self.nextObjectID>=99:
            self.nextObjectID = 0
        
        return self.nextObjectID-1, pmv

    def __register_nonmatch_track(self, rects, lmks, lmk_confs, body_cropf, bodyfeatf, face_cropf, facefeatf, cloth_attf):
        """
        add centroid and feature to tracking table
        """
        print(colored("NonMatched TRACK ADDED {}".format(self.nextObjectID),'red'))
        feat_norm = np.linalg.norm(facefeatf)
        print("FACE FEAT NORM ", feat_norm)
        
        # #check visibility of eyes and nose
        # match_flag = -1
        # if lmk_conf[0]>self.LMK_VISIBILITY_THRESHOLD and lmk_conf[1]>self.LMK_VISIBILITY_THRESHOLD and lmk_conf[2]>self.LMK_VISIBILITY_THRESHOLD:
        #     match_flag = 0
        
        # pmv = 0.0
        person_details = {'gender': np.nan, 'race': np.nan, 'age': np.nan} 
        pmv = self.tc_predictor.model_inference(cloth_attf['iclo'], person_details)

        trackid = self.nextObjectID
        subid = "Unk_{}".format(trackid)

        row = {
            'track_id': trackid,
            'sub_id': subid,
            'rects': rects,
            'lmks': lmks,
            'lmk_confs': lmk_confs,
            'lastbodyfeat': bodyfeatf,
            'avgbodyfeat': bodyfeatf,
            'len': 1,
            'body_crop': body_cropf,
            'face_crop': face_cropf,
            'face_feat': facefeatf,
            'age': 0,
            'db_flag': 0,
            'person_details': [],
            'cloth_att': cloth_attf,
            'iclo': cloth_attf['iclo'],
            'pmv': pmv,
            'iclo_kalman': KF(cloth_attf['iclo']),
            'pmv_kalman': KF(pmv)
        }
        self.tracks = self.tracks.append(row, ignore_index=True)
        self.nextObjectID += 1

        if self.nextObjectID>=99:
            self.nextObjectID = 0
        
        return trackid, subid, pmv

    def __update_track(self, trackid, rects, lmks, lmk_confs, body_cropf, bodyfeatf, face_cropf, face_featf, cloth_attf):
        """
        update old tracks with new centroid and feature:
        """
        #update
        if trackid in self.tracks['track_id'].values:  # Exists
            idx = self.tracks.index[self.tracks['track_id'] == trackid].item()
            self.tracks.at[idx, 'rects'] = rects
            self.tracks.at[idx, 'lmks'] = lmks
            self.tracks.at[idx, 'lmk_confs'] = lmk_confs
            self.tracks.at[idx, 'face_crop'] = face_cropf
            self.tracks.at[idx, 'body_crop'] = body_cropf
            self.tracks.at[idx, 'lastbodyfeat'] = bodyfeatf
            self.tracks.at[idx, 'face_feat'] = face_featf

            #smooth and update iclo
            self.tracks.at[idx, 'cloth_att'] = cloth_attf
            icloKalmanFilter = self.tracks.at[idx, 'iclo_kalman']
            smooth_iclo = icloKalmanFilter.update(cloth_attf['iclo'])
            self.tracks.at[idx, 'iclo'] = smooth_iclo
            
            #match face if unknown
            face_label = self.tracks.at[idx, 'sub_id'] 
            person_details = self.tracks.at[idx, 'person_details']
            if "Unk" in face_label:
                _, face_label, person_details = self.__face_match_database(face_featf)

            if not "Unk" in face_label:
                print("P ", self.tracks.at[idx, 'person_details'], person_details)
                self.tracks.at[idx, 'person_details'] = person_details
                self.tracks.at[idx, 'sub_id'] = face_label

                #replace avg body feat
                self.tracks.at[idx, 'avgbodyfeat'] = bodyfeatf
                self.tracks.at[idx, 'len'] = 1
                self.tracks.at[idx, 'db_flag'] = 0

            else:
                person_details = {'gender': np.nan, 'race': np.nan, 'age': np.nan} 

            #predict pmv from iclo
            print("CLOTH ATT ", cloth_attf, face_label)
            pmv = self.tc_predictor.model_inference(smooth_iclo, person_details)

            #smooth and update pmv
            pmvKalmanFilter = self.tracks.at[idx, 'pmv_kalman']
            smooth_pmv = pmvKalmanFilter.update(pmv)
            self.tracks.at[idx, 'pmv'] = smooth_pmv

            #update current feature if track age is below threshold else replace features    
            # print("track AGE track id", self.tracks.at[idx, 'age'], trackid)
            
            if self.tracks.at[idx, 'age'] < self.track_age or self.tracks.at[idx, 'len'] < 30:
                print(colored("Not old, Updating track length", "red"), self.tracks.at[idx, 'len'])
                self.tracks.at[idx, 'avgbodyfeat'] += bodyfeatf
                self.tracks.at[idx, 'len'] += 1
            else:
                print(colored("Replacing track","red"))
                self.tracks.at[idx, 'avgbodyfeat'] = bodyfeatf
                self.tracks.at[idx, 'len'] = 1
                self.tracks.at[idx, 'db_flag'] = 0

            self.tracks.at[idx, 'age'] = 0
            
            return self.tracks.at[idx, 'pmv']
        
        else :

            return 0

    def __get_body_avg_feat(self, trackid):
        if trackid in self.tracks['track_id'].values:  # Exists
            idx = self.tracks.index[self.tracks['track_id'] == trackid].item()
            length = self.tracks.at[idx, 'len']
            avgfeat = self.tracks.at[idx, 'avgbodyfeat']

            avgfeat = avgfeat/length
        return avgfeat

    def Increment_track_age(self):
        self.__save_database()
        rows_to_update = self.tracks.loc[self.tracks['age'] >= 0]
        idx_to_update = rows_to_update.index.tolist()
        for idx in idx_to_update:
            self.tracks.at[idx, 'age'] += 1
            print("Increment track age of track id ", self.tracks.at[idx, 'age'], self.tracks.at[idx, 'track_id'])

    def __delete_old_tracks(self):
        """
        Delete old track
        """
        rows_to_del = self.tracks.loc[self.tracks['age'] > self.max_track_age]
        idx_to_del = rows_to_del.index.tolist()
        for idx in idx_to_del:
            print("Removing track id ", idx)
            self.tracks = self.tracks.drop(index=idx)

    def __face_match_database(self, face_feature, threshold=None):
        """
        features:
            Matrix of feature vectors to get similarity indices for.
        """
        face_feature = np.reshape(face_feature, (1, -1)) 
        assert(len(face_feature))==1

        sim_thresh = threshold if threshold is not None else self.fr_threshold

        db_index = []
        label = 'Unk'
        person_details = []
        
        if not self.face_db_empty_flag:
            print("face Matching with database features")
            print("face matching ", self.face_db_features.shape, face_feature.shape)
            db_index, _ = MatchFeatures(face_feature, self.face_db_features, sim_thresh)

            if db_index[0] > -1:
#                label = self.face_database.get_name_from_index(db_index[0])
                label, person_details = self.face_database.get_details_from_index(db_index[0])
                print(type(label))
                print("PFactors ", person_details)
        else:
            db_index.append(-1)

        return db_index, str(label), person_details

    def __face_match_tracks(self, face_feature, threshold=None):
        """
        features:
            Matrix of feature vectors to get similarity indices for.
        """
        face_feature = np.reshape(face_feature, (1, -1)) 
        assert(len(face_feature))==1

        track_id = -1
        label = 'Unk'
        if self.nextObjectID>0:
            print("face Matching with track features")
            sim_thresh = threshold if threshold is not None else self.fr_threshold

            track_face_feats = np.stack(self.tracks['face_feat'].tolist())
            track_sub_ids = np.stack(self.tracks['sub_id'].tolist())
            track_trackids = np.stack(self.tracks['track_id'].tolist())

            db_index, _ = MatchFeatures(face_feature, track_face_feats, sim_thresh)

            if db_index[0] > -1:
                label = track_sub_ids[db_index[0]]
                track_id = track_trackids[db_index[0]]

        print("Track Matching face", label)
        return track_id, label

    def __body_match_tracks(self, body_feature, threshold=None):
        """
        features:
            Matrix of feature vectors to get similarity indices for.
        """
        body_feature = np.reshape(body_feature, (1, -1)) 
        assert(len(body_feature))==1

        track_id = -1
        label = 'Unk'
        if self.nextObjectID>0:
            print("body Matching with track features")
            sim_thresh = threshold if threshold is not None else self.pr_threshold

            track_body_feats = np.stack(self.tracks['lastbodyfeat'].tolist())
            track_sub_ids = np.stack(self.tracks['sub_id'].tolist())
            track_trackids = np.stack(self.tracks['track_id'].tolist())

            db_index, _ = MatchFeatures(body_feature, track_body_feats, sim_thresh)

            if db_index[0] > -1:
                label = track_sub_ids[db_index[0]]
                track_id = track_trackids[db_index[0]]

        print("Track Matching body", label)
        return track_id, label

    def __body_match_cameras(self, body_feature1, body_feature2, threshold=None):
        """
        features:
            Matrix of feature vectors to get similarity indices for.
        """
        # body_feature1 = np.reshape(body_feature1, (1, -1)) 
        # assert(len(body_feature1)==len(body_feature2))

        sim_thresh = threshold if threshold is not None else 0.3

        # matches = []
        # print("body Matching across twin dome cameras")
        # for i in range(len(body_feature1)):
        #     bodyfeat = body_feature1[i]
        #     bodyfeat = np.reshape(bodyfeat, (1, -1)) 

        #     db_index, _ = MatchFeatures(bodyfeat, body_feature2, sim_thresh)

        #     matches.append(db_index[0])

        db_index, _ = MatchFeatures(body_feature1, body_feature2, sim_thresh)
        print("Twin camera Body Matching", db_index)
        return db_index

    # def __match_faces(self, face_feature):
    #     """
    #     extract face features of active tracks and match with face db
    #     """
    #     #match new face features with previous tracked face features
    #     face_feature = np.reshape(face_feature, (1, -1)) 
    #     if self.nextObjectID>0:
    #         print("Track Matching")
    #         track_face_feats = np.stack(self.tracks['face_feat'].tolist())
    #         track_sub_ids = np.stack(self.tracks['sub_id'].tolist())

    #         db_index, face_label = self.__face_match_tracks(face_feature, track_face_feats, track_sub_ids)
    #         if face_label == "Unk":
    #             face_label = self.__face_match_database(face_feature)
    #     else:
    #         print("FaceDB Matching")
    #         db_index, face_label = self.__face_match_database(face_feature)

    #     return face_label[0]

    def update(self, rects, lmks, lmk_confs, body_crops, body_features, face_crops, face_features, face_confs, cloth_atts):
        """
        update tracking table with new centroids and fetures
        """
        # bboxs = dts[:,:4]            
        # angles = dts[:,4:5]           

        #### EDITED BY JOE
        #load updated face database
        # if os.path.isfile(self.face_db_file): #not self.body_db_empty_flag:
        #     curr_face_db_time = os.path.getmtime(self.face_db_file)
        # else:
        #     curr_face_db_time = -1.0

        '''
        print("Face DB {} {}".format(self.old_face_db_time, curr_face_db_time))
        if curr_face_db_time>self.old_face_db_time:
            self.face_database, self.face_db_features, self.face_db_empty_flag = InitFaceDatabase(self.face_db_file)
            self.old_face_db_time = curr_face_db_time
            print("Loaded UPDATED FaceDB")
        '''
            
        #match faces with current tracks/face-dB
        track_ids = []
        face_labels = []
        pmvs = []
        for k in range(len(rects)):
            track_ids_img = []
            face_labels_img = []
            pmvs_img = []
            for i in range(len(rects[k])):
                bboxs = rects[k][i]
                lmk = lmks[k][i]
                lmk_conf = lmk_confs[k][i]
                face_crop = face_crops[k][i]
                body_crop = body_crops[k][i]
                face_feat = face_features[k][i]
                body_feat = body_features[k][i]
                # iou_bb = iou[k][i]
                cloth_att = cloth_atts[k][i]

                # # if iou_bb[0]>self.iou_th or iou_bb[1]>self.iou_th:
                # # if iou_bb[0]>self.iou_th_x or iou_bb[1]>self.iou_th_x:
                # if iou_bb>self.iou_th_x:
                #     continue

                # cv2.imwrite("test_face.jpg", face_crop)
                
                #matches face feature with existing tracks
                track_id = -1
                if not self.tracks.empty: 
                    # track_id_face, face_label = self.__face_match_tracks(face_feat)
                    track_id, face_label = self.__body_match_tracks(body_feat)
                    
                #matches face feature with database
                if track_id==-1:
                    _, face_label, person_details = self.__face_match_database(face_feat)

                #update esisting matched tracks or register matched faces
                pmv = 0
                if track_id>-1:
                    print(colored("UPDATING TRACKS", "green"))
                    pmv = self.__update_track(track_id, bboxs, lmk, lmk_conf, body_crop, 
                        body_feat, face_crop, face_feat, cloth_att)
                elif "Unk" in face_label:
                    print(colored("ADDING TO TRACKS as unkown", "blue"))
                    track_id, face_label, pmv = self.__register_nonmatch_track(bboxs, lmk, lmk_conf, body_crop, 
                        body_feat, face_crop, face_feat, cloth_att)
                else:
                    idx = self.tracks.loc[self.tracks['sub_id'] == face_label]
                    idx = idx.index.tolist()
                    if idx:  # Exists

                        print("FACE Exists: multiple Matching case")

                        ## Get the tracl id of the corresponding idx

                        track_id = idx[0] #self.tracks.at[idx[0], 'track_id']
                        # track_id = self.tracks["track_id"].iloc[track_id]
                        # track_id = idx[1] #self.tracks.at[idx[0], 'track_id']-- this will print track id
                        pmv = self.__update_track(track_id, bboxs, lmk, lmk_conf, body_crop, 
                            body_feat, face_crop, face_feat, cloth_att)
                    else:
                        print(colored("ADDING TO TRACKS", "blue"))
                        track_id, pmv = self.__register_track(face_label, bboxs, lmk, lmk_conf, body_crop, 
                            body_feat, face_crop, face_feat, person_details, cloth_att)
 
                track_ids_img.append(track_id)
                face_labels_img.append(face_label)                   
                pmvs_img.append(pmv)

            track_ids.append(track_ids_img)
            face_labels.append(face_labels_img)                   
            pmvs.append(pmvs_img)

        self.Increment_track_age()
        print("TRACK ids label ", track_ids, face_labels)
        self.__delete_old_tracks()

        # return track_id, face_label
        return track_ids, face_labels, pmvs

def MatchFeatures(features, feat_to_compare_with, sim_thresh):
    """
    Do an N:M comparison of feature vectors.
    Pass a single feature vector to <features> to do 1:M comparison.
    Pass a single feature vector to <feat_to_compare_with> to do N:1 comparison.
    Pass a single feature vector to both to do 1:1 comparison.

    features:
        Matrix of feature vectors to get similarity indices for.
    feat_to_compare_with:
        The feature vector to compare with (usually existing/database features).
        Pass a single feature vector here to do N:1 comparison.
    threshold:
        Optional parameter to manually set the similarity threshold.

    Returns:
        A numpy array of length N.
        Each element contains the index of the closest match in
        feat_to_compare_with (between 0 and M-1) without repetition.
        There may not be N successful matches. If there was no match
        (score under threshold) the element will be -1.

        Example:
            Comparing 4:7 might return something like [-1 6 0 -1], which means:
                <features>[0] and [3] had no matches
                <feature>[1] matched <feat_to_compare_with>[6]
                <feature>[2] matched <feat_to_compare_with>[0]
        Example:
            Comparing 8:4 might return something like [-1 -1 3 1 -1 0 -1 -1], which means:
                <features>[2] matched <feat_to_compare_with>[3]
                <features>[3] matched <feat_to_compare_with>[1]
                <features>[6] matched <feat_to_compare_with>[0]
                Remaining <features> didn't get matched. In other words, no faces
                were matched with <feat_to_compare_with>[2].
    """

    debug = False


    mat_curr = np.array(features)
    mat_prev = np.array(feat_to_compare_with)
    print(mat_curr.shape, mat_prev.shape)

    num_curr = mat_curr.shape[0]  # "current frame" features
    num_prev = mat_prev.shape[0]  # "prev frame" or "database" features

    #feat_curr = self.__crypto.decrypt_vec(mat_curr, (num_curr, -1))
    #feat_prev = self.__crypto.decrypt_vec(mat_prev, (num_prev, -1))

    if debug:
        print("=" * 20)
        print("sim threshold:", sim_thresh)
        # print("mat_prev.shape:", mat_prev.shape)
        # print("mat_curr.shape:", mat_curr.shape)
        # print("feat_prev.shape:", feat_prev.shape)
        # print("feat_curr.shape:", feat_curr.shape)

    # Similarity matrix (rows: previous, cols: current):
    #sim = cosine_similarity(feat_prev, feat_curr)
    sim = cosine_similarity(mat_prev.reshape(num_prev,-1), mat_curr.reshape(num_curr,-1))
    print("SIM\n", sim)

    # Return matrix with num_curr elements, all set to -1:
    return_indices = np.full(num_curr, -1)
    sim_scores = np.full(num_curr, -1.0)

    # for i in range(num_curr):
    #     #each column represents similarity score with db
    #     #get maximum score for each column
    #     best_idx = np.argmax(sim[:, i])
    #     if sim[best_idx, i]>sim_thresh:
    #         return_indices[i] = best_idx
    #         sim_scores[i] = sim[best_idx, i]


    for i in range(num_curr):
        # Find index of the highest sim score (row: previous, col: current):
        # This means the highest similarity value will be taken first.
        best_idx = np.unravel_index(np.argmax(sim), shape=sim.shape)

        if debug:
            print("i:", i)
            # print("sim:\n{}".format(sim))
            # print("best_idx:", best_idx)
            print("sim[best_idx].item()", sim[best_idx].item())

        if sim[best_idx] < sim_thresh:
            # This means there are no more similarity scores above the threshold (default=0.3)
            # By breaking, the remaining return_indices will be left as -1.
            # print(">> break\n")
            break
        else:
            # Use best_idx col value to select the relevant element in return_indices,
            # then set that element to the best_idx row value, which is the "best match"
            # from feat_existing:
            return_indices[best_idx[1]] = best_idx[0]
            sim_scores[best_idx[1]] = sim[best_idx]

            # Set the whole column to -1. "This face has been checked already."
            sim[:, best_idx[1]] = -1

            # Set the entire row to -1 for next iteration, to prevent multiple columns
            # being matched to one row (e.g. "two faces to the same name"):
            sim[best_idx[0], :] = -1

            if debug:
                print("return_indices: {}".format(return_indices))
                print()

    if debug:
        print("return_indices: {}".format(return_indices))

    # print(return_indices, sim_scores)
    return return_indices, sim_scores

def InitFaceDatabase(db_file=None, db_dict=None, verbose=False):
    flag = False
    if not db_file is None:
        database = FaceDatabase(db_file)
    if not db_dict is None:
        database = FaceDatabase(db_dict=db_dict)
    if database.is_empty():
        flag = True
        print(">> WARNING: No entries in database for comparison.")
    print(database.get_name_list())
    print(database.print_data())
    db_features = database.get_features_as_array()  # For use in Match_Old()
    return database, db_features, flag


class ThermalComfort():
    def __init__(self, pmv_model, env_variables):
        """
        pmv_model: json path of XGBoost model
        env_variables: dict() contains keys:-  
                v(air speed), rh(relative humidity), 
                tdb(temp dry bulb/room temp), tr(radiant temp)
        met: int metabolic rate
        """
        self.pmv_estimator = XGBRegressor()
        self.pmv_estimator.load_model(pmv_model)
        self.env_variables = env_variables
#        self.met = met
#        self.age_enconding = {'children': 0, 'youth': 1, "young adults": 2 ,'middle-aged adults': 3, 'old adults': 4}
#        self.gen_enconding = {'Male': 1, 'Female': 0}
#        self.race_encoding = {'Asian': 0, 'Caucasian': 1, 'Indian': 2}
#        self.signals = SignalSmoothing()

    def v_relative(self, v, met):
        """Estimates the relative air speed which combines the average air speed of
        the space plus the relative air speed caused by the body movement. Vag is assumed to
        be 0 for metabolic rates equal and lower than 1 met and otherwise equal to
        Vag = 0.3 (M – 1) (m/s)
        Parameters
        ----------
        v : float or array-like
            air speed measured by the sensor, [m/s]
        met : float
            metabolic rate, [met]
        Returns
        -------
        vr  : float or array-like
            relative air speed, [m/s]
        """
    
        return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)
    
    def model_inference(self, iclo, person_detail, met=1.2):
        
#        print("PMV track ", personal_detail)
#        print("PMV track ", iclo)
        
        p = person_detail
        vr = float(self.v_relative(v=self.env_variables['v'], met=met))

        modelip = pd.DataFrame([[iclo, met, vr, self.env_variables['rh'], 
                                self.env_variables['tdb'], self.env_variables['tr'],
                                p['age'], p['gender'], p['race']]], 
                            columns=['gtClo', 'gtMet', 'gtAS', 'gtRelHum','gtTA', 'gtRT',
                                     'Age', 'Gender', 'Race'])
        
        pmv = self.pmv_estimator.predict(modelip)[0]
        print("PMV ", iclo, person_detail, met, pmv)
        
        return pmv
    
class KF():
    def __init__(self, value):
        self.mean = value ## mean
        self.cov = 1 ## cov
        self.kalmanTransCov = 0.5
        self.kfilter =  KalmanFilter(transition_matrices=[1],
                              observation_matrices=[1],
                              initial_state_mean=self.mean,
                              initial_state_covariance=self.cov,
                              observation_covariance=10,
                              transition_covariance=self.kalmanTransCov) ## kalman filter
               
        
    def update(self, value):
        temp_state_means, temp_state_cov = (self.kfilter.filter_update(self.mean,
                                self.cov,
                                observation = value,
                                observation_covariance = np.asarray([5])))
        
        self.mean, self.cov = temp_state_means, temp_state_cov
        
        return round(self.mean[0][0],2)
