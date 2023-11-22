import os
# import base64

import cv2
import numpy as np
from skimage import transform
# from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# from Engines.rapid_detector import RapidDetector
# from Pose.detect import HumanPoseDetection
# from Engines.embedding import UbodyEmbedding

from rapid_detector_16fp import RapidDetector
from embedding_16fp import UbodyEmbedding

from collections import OrderedDict
from scipy.spatial import distance as dist

from PIL import Image
import time
import math
import pandas as pd
from termcolor import colored
from queue import Queue

import sys
import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'pose'))
from detect_16fp import HumanPoseDetection

sys.path.append(str(scriptpath.parent / 'common'))
from database import Database

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

class FishEyeTracker:
    def __init__(self, pr_threshold, body_db_file):
        """
        initialize the next unique object ID to keep track of mapping 
        a given object ID to its centroid 
        """

        # Initializing face matching module
        # self.LMK_VISIBILITY_THRESHOLD = 0.8
        self.tr_threshold = 0.5 #pr_threshold #
        self.pr_threshold = pr_threshold

        self.iou_threshold = 0.2
        self.norm_threshold = 8 #10
        self.dist_threshold = 300

        self.body_db_file = body_db_file
        self.body_database, self.body_db_features, self.body_db_empty_flag = InitDatabase(self.body_db_file)
        logging.debug("Body database loaded ...")
        if not self.body_db_empty_flag:
            self.old_db_time = os.path.getmtime(self.body_db_file)
        else:
            self.old_db_time = -1.0

        self.nextObjectID = 0
        self.track_age = 30
        self.non_matched_age = 15

        self.columns = [
            'track_id', 'sub_id', 'bbox', 'lmk', 'lmk_conf', 'lastfeat',
            'avgfeat', 'len', 'time_stamp', 'age'
        ]

        self.tracks = pd.DataFrame(columns=self.columns)  # The "database"

    def save_database(self, save_file=None):
        # Commit the current self.data to a .pkl file.
        if save_file:
            path = save_file
        else:
            path = self.db_file

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        avg_feats = self.__get_avg_feat()
        sub_id = np.stack(self.tracks['sub_id'].tolist())
        
        for i in range(len(sub_id)):
            row = {'name': sub_id[i], 'features': avg_feats[i]}
            self.body_db = self.body_db.append(row, ignore_index=True)

        self.body_db.to_pickle(path)
        string = "Database saved to {}. {} identities in database.".format(path, len(self.body_db))
        print(string)

    def __register_track(self, bbox, lmk, lmk_conf, feature, label):
        """
        add centroid and feature to tracking table
        """

        print(colored("TRACK ADDED {}".format(self.nextObjectID),'red'))       
      
        row = {
            'track_id': self.nextObjectID,
            'sub_id': label,
            'bbox': bbox,
            'lmk': lmk,
            'lmk_conf': lmk_conf,
            'lastfeat': feature,
            'avgfeat': feature,
            'len': 1,
            'age': 0
        }
        self.tracks = self.tracks.append(row, ignore_index=True)
        self.nextObjectID += 1

    def __update_track(self, trackid, bbox, lmk, lmk_conf, feature, iou):
        """
        update old tracks with new centroid and feature:
        """
        if trackid in self.tracks['track_id'].values:  # Exists
            idx = self.tracks.index[self.tracks['track_id'] == trackid].item()

            norm = int(np.linalg.norm(feature))

            self.tracks.at[idx, 'bbox'] = bbox
            self.tracks.at[idx, 'lmk'] = lmk
            self.tracks.at[idx, 'lmk_conf'] = lmk_conf
            self.tracks.at[idx, 'age'] = 0
            
            if iou<self.iou_threshold:
                self.tracks.at[idx, 'lastfeat'] = feature
                norm = np.linalg.norm(feature)
                if norm>self.norm_threshold:
                    self.tracks.at[idx, 'avgfeat'] += feature
                    self.tracks.at[idx, 'len'] += 1

    def __get_avg_feat(self):
        avg_feats = np.stack(self.tracks['avgfeat'].tolist())
        # print(avg_feats.shape)
        length = np.stack(self.tracks['len'].tolist())
        for i in range(len(length)):
            avg_feats[i, :] = avg_feats[i, :]/length[i]
        return avg_feats

    def __increment_track_age(self):
        rows_to_update = self.tracks.loc[self.tracks['age'] >= 0]
        idx_to_update = rows_to_update.index.tolist()
        for idx in idx_to_update:
            self.tracks.at[idx, 'age'] += 1

    def __delete_old_tracks(self):
        """
        Delete old track
        """
        rows_to_del = self.tracks.loc[self.tracks['age'] > self.track_age]
        idx_to_del = rows_to_del.index.tolist()
        if len(idx_to_del)>0:
            self.tracks = self.tracks.drop(index=idx_to_del)
 
    def __body_match(self, features, threshold=None):
        """
        features:
            Matrix of feature vectors to get similarity indices for.
        """

        sim_thresh = threshold if threshold is not None else self.pr_threshold
        labels = []
        if not self.body_db_empty_flag:
            print(colored("DATABASE MATCHING","red"))

            db_index, db_scores = MatchFeatures(features, self.body_db_features, sim_thresh)

            for i in range(len(features)):
                if db_index[i] == -1:
                    label = 'UnK'
                else:
                    label = self.body_database.get_name_from_index(db_index[i])
                labels.append(label)
        else:
            db_index = []
            label = 'UnK'
            for i in range(len(features)):
                db_index.append(-1)
                labels.append(label)

        # print("BODY DB", db_index, db_scores)

        return db_index, labels

    def __match_bodys(self):
        """
        match body features of active tracks and match with body db
        """
        # print(self.tracks)
        rows_to_match = self.tracks.loc[(self.tracks['age'] == 0) & (self.tracks['match_flag']==0)]
        idx_to_match = rows_to_match.index.tolist()

        if len(idx_to_match)>0:
            # print("BODY MATCHING ", len(idx_to_match), idx_to_match)
            bodyfeatures = [] 
            for idx in idx_to_match:
                # feat = self.tracks.at[idx, 'lastfeat']
                feat = self.tracks.at[idx, 'avgfeat']
                length = self.tracks.at[idx, 'len']
                feat = feat/length
                bodyfeatures.append(feat)

            body_indices, body_labels = self.__body_match(bodyfeatures)

            # print(body_indices, body_labels)

            for i, idx in enumerate(idx_to_match):
                if body_indices[i]>=0:
                    self.tracks.at[idx, 'sub_id'] = body_labels[i]
                    self.tracks.at[idx, 'match_flag'] = 1
        # else:
        #     print("NO BODY MATCHING")
        #     print(self.tracks)


    def __box_stats(self, rects, img_dim):
        print("IMAGE DIM", img_dim)
        half_h = img_dim[0]/2
        half_w = img_dim[1]/2

         # obtain box coordinates in (x,y,w,h) format and ious w.r.t. other boxes
        area = np.zeros((len(rects), 1), dtype="float")
        dist = np.zeros((len(rects), 1), dtype="float")
        bboxs =  np.zeros((len(rects), 4), dtype="float")

        for (i, (cX, cY, width, height)) in enumerate(rects):
            # inputCentroids[i] = (int(cX), int(cY))
            area[i] = int(width*height)
            dist[i] = abs(half_h-cY)+abs(half_w-cX)
            bboxs[i] = (cX-width, cY-height, cX+width, cY+height)

        # for i in range(0, len(rects)):
        #     cx, cy, w, h = rects[i]
        #     w = w/2
        #     h = h/2
        #     bboxs[i] = (cx-w, cy-h, cx+w, cy+h)

        iou = np.full(len(rects), 0.)
        for i in range(0, len(bboxs)):
            bb1 = bboxs[i]
            iou_array = []
            for j in range(0, len(bboxs)):
                bb2 = bboxs[j]
                iou_array.append(get_iou(bb1, bb2))

            iou_array[i] = 0 #make iou wrt itself 0
            iou[i] = np.sum(iou_array)

        return bboxs, area, dist, iou

    def update(self, rects, lmks, lmk_confs, features, img_dim):
        """
        update tracking table with new centroids and fetures
        """
        # print(flags)
        # sim_thresh = threshold if threshold is not None else self.pr_threshold
        # if labels is None :
        #     labels = []
        #     for i in range(len(rects)):
        #         labels.append("UnK")

        #load updated database
        if os.path.isfile(self.body_db_file): #not self.body_db_empty_flag:
            curr_db_time = os.path.getmtime(self.body_db_file)
        else:
            curr_db_time = -1.0

        print("DB {} {}".format(self.old_db_time, curr_db_time))
        if curr_db_time>self.old_db_time:
            self.body_database, self.body_db_features, self.body_db_empty_flag = InitDatabase(self.body_db_file)
            self.old_db_time = curr_db_time
            print("Loaded UPDATED BodyDB")


        #increment track age
        self.__increment_track_age()

        print("DETECTIONS Tracks", len(rects), len(self.tracks))

        # obtain box coordinates in (x,y,w,h) format and ious w.r.t. other boxes
        bboxs, area, dist, iou  = self.__box_stats(rects, img_dim)

        # rect_width = rects[:,2:3]

        # track_index = np.full(len(rects), -1) #np.zeros((len(rects), 1), dtype="int")
        subids = ["UnK"] * len(rects) # np.full(len(rects), -1) #[] #np.zeros((len(rects), 1), dtype="int")
        trackids = np.full(len(rects), -1) #[]

        # no tracking objects so we need to
        # try to match the input features to database and add to tracking table
        if self.nextObjectID == 0:
            body_indices, body_labels = self.__body_match(features)

            for i in range(len(features)):
                if body_indices[i]>=0:
                    self.__register_track(bboxs[i], lmks[i], lmk_confs[i], features[i], body_labels[i])
                    # track_index[i] = self.nextObjectID-1
                    subids[i] = body_labels[i]
                    trackids[i] = self.nextObjectID-1
                else:
                    subids[i] = "UnK"
                    trackids[i] = -1

        # otherwise, try to match the input features to existing object
        else:
            # cx = np.stack(self.tracks['cx'].tolist())
            # cy = np.stack(self.tracks['cy'].tolist())
            dbBoxes = np.stack(self.tracks['bbox'].tolist())
            last_feats = np.stack(self.tracks['lastfeat'].tolist())
            avg_feats = self.__get_avg_feat()
            track_id = np.stack(self.tracks['track_id'].tolist())
            sub_id = np.stack(self.tracks['sub_id'].tolist())
            print(colored("TRAAAAAAACK DATABBBBBBASE","red"))
            print(track_id, sub_id)

            # dbBoxes = np.zeros((len(cx), 2), dtype="int")
            # for i in range(len(cx)):
            #     dbCentroids[i] = (cx[i], cy[i])

            # print(len(last_feats), len(avg_feats), len(dbCentroids))
            # print(len(inputCentroids), len(rects), len(features))
            # ct_index, ct_scores = MatchPts(inputCentroids, dbCentroids, dist_thresh=100.0)
            ct_index, ct_scores = MatchBoxes(bboxs, dbBoxes, iou_thresh=0.1)

            # sim_thresh = 0.3 #self.pr_threshold #*0.7
            print(colored("Last BODY FEATs {} {}".format(len(features), len(last_feats)), 'blue'))
            last_index, last_scores = MatchFeatures(features, last_feats, sim_thresh=self.tr_threshold)
            avg_index, avg_scores = MatchFeatures(features, avg_feats, sim_thresh=self.tr_threshold)

            print("LAST", last_index, last_scores)
            print("AVG", avg_index, avg_scores)
            print("IOU", ct_index, ct_scores)

            non_matched_features = []
            non_matched_index = [] 

            for i in range(len(rects)):
                # print("Area", i, area[i], iou[i])
                # if last_index[i] == -1 and avg_index[i] == -1 and ct_index[i] == -1:
                #     norm = int(np.linalg.norm(features[i]))
                #     print(colored("AREA IOU NORM ","blue"), area[i], iou[i], norm)
                #     if area[i]>=self.area_threshold and iou[i]<self.iou_threshold and norm>=self.norm_threshold:
                #         print(colored("ADDING TO TRACKS", "blue"))
                #         # self.__register_track(inputCentroids[i], lmks[i], lmk_confs[i], features[i])
                #         self.__register_track(bboxs[i], lmks[i], lmk_confs[i], features[i])
                #         track_index[i] = self.nextObjectID-1
                # elif last_index[i] == -1 and avg_index[i] == -1 and ct_scores[i]>self.iou_threshold:
                # # elif dist[i] < self.dist_threshold and ct_scores[i]>self.iou_threshold:
                #     print(colored("ctUPDATING TRACK {}".format(ct_index[i]), "green"))
                #     self.__update_track(ct_index[i], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i], area[i])
                #     track_index[i] = track_id[ct_index[i]]

                #match found in existing tracks
                # elif last_index[i] == -1 and avg_index[i] == -1 and ct_scores[i]>self.iou_threshold:
                #box is in center of fisheye camera
                if dist[i] < self.dist_threshold and ct_scores[i]>self.iou_threshold:
                    print(colored("ctUPDATING TRACK {}".format(track_id[ct_index[i]]), "green"))
                    self.__update_track(track_id[ct_index[i]], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i])
                    # track_index[i] = track_id[ct_index[i]]
                    subids[i] = sub_id[ct_index[i]]
                    trackids[i] = track_id[ct_index[i]]
                elif ct_index[i] == last_index[i] and last_index[i] != -1:
                    print(colored("ltUPDATING TRACK {}".format(track_id[ct_index[i]]), "green"))
                    # track_age = self.tracks.at[track_id[ct_index[i]], 'age']
                    # if track_age!=0:
                    #     self.__update_track(track_id[ct_index[i]], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i])
                    #     subids[i] = sub_id[ct_index[i]]
                    #     trackids[i] = track_id[ct_index[i]]
                    self.__update_track(track_id[ct_index[i]], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i])
                    # track_index[i] = track_id[ct_index[i]]
                    subids[i] = sub_id[ct_index[i]]
                    trackids[i] = track_id[ct_index[i]]
                elif ct_index[i] == avg_index[i] and avg_index[i] != -1:
                    print(colored("avUPDATING TRACK {}".format(track_id[ct_index[i]]), "green"))
                    # track_age = self.tracks.at[track_id[ct_index[i]], 'age']
                    # if track_age!=0:
                    #    self.__update_track(track_id[ct_index[i]], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i])
                    #     subids[i] = sub_id[ct_index[i]]
                    #     trackids[i] = track_id[ct_index[i]]
                    self.__update_track(track_id[ct_index[i]], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i])
                    # track_index[i] = track_id[avg_index[i]]
                    subids[i] = sub_id[ct_index[i]]
                    trackids[i] = track_id[ct_index[i]]
                elif avg_index[i] == last_index[i] and last_index[i] != -1:
                    print(colored("UPDATING TRACK {}".format(track_id[avg_index[i]]), "green"))
                    self.__update_track(track_id[avg_index[i]], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i])
                    # track_index[i] = track_id[last_index[i]]
                    subids[i] = sub_id[avg_index[i]]
                    trackids[i] = track_id[avg_index[i]]
                # elif avg_scores[i]>self.tr_threshold:
                #     print(colored("avUPDATING TRACK {}".format(last_index[i]), "green"))
                #     self.__update_track(avg_index[i], dist[i], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i], area[i])
                #     track_index[i] = track_id[avg_index[i]]
                # elif last_scores[i]>self.tr_threshold:
                #     print(colored("avUPDATING TRACK {}".format(last_index[i]), "green"))
                #     self.__update_track(last_index[i], dist[i], bboxs[i], lmks[i], lmk_confs[i], features[i], iou[i], area[i])
                #     track_index[i] = track_id[avg_index[i]]
                else:
                    non_matched_index.append(i)
                    non_matched_features.append(features[i])
                
                print(trackids[i], subids[i])

            if len(non_matched_index)>0:
                body_indices, body_labels = self.__body_match(non_matched_features)
                # print("STATUS\n", trackids, subids)
                print(colored("MATCHED MISSING FEATURES","red"), len(non_matched_features), len(features))
                print(body_indices, body_labels)
                for i in range(len(non_matched_index)):
                    idx = non_matched_index[i]
                    if body_indices[i]>=0:
                        print("\n", self.tracks['sub_id'].values)
                        if body_labels[i] in self.tracks['sub_id'].values:
                            tid = self.tracks.index[self.tracks['sub_id'] == body_labels[i]].item()
                            #check for duplicate, if track age is zero, it means it is already updated
                            #multiple match scenario
                            track_age = self.tracks.at[tid, 'age']
                            if track_age>self.non_matched_age:
                                self.__update_track(tid, bboxs[idx], lmks[idx], lmk_confs[idx], features[idx], iou[idx])
                                subids[idx] = body_labels[i]
                                trackids[idx] = tid
                        else:
                            self.__register_track(bboxs[idx], lmks[idx], lmk_confs[idx], features[idx], body_labels[i])
                            # track_index[idx] = self.nextObjectID-1
                            subids[idx] = body_labels[i]
                            trackids[idx] = self.nextObjectID-1


        # # # #match face crops
        # # self.__match_bodys()

        # # map modified centroids ids to input boxes
        # subids = [] #np.zeros((len(rects), 1), dtype="int")
        # trackids = []

        # for i in range(len(track_index)):
        #     tid = track_index[i]
        #     if tid>=0:
        #         if tid in self.tracks['track_id'].values:  # Exists
        #             idx = self.tracks.index[self.tracks['track_id'] == tid].item()
        #             sid = self.tracks.at[idx, 'sub_id']
        #             subids.append(sid)
        #             tid = self.tracks.at[idx, 'track_id']
        #             trackids.append(tid)
        #         else:
        #             print(colored("TRACKING ERROR"), 'red')
        #     else:
        #         subids.append("UnK")
        #         trackids.append(-1)

        # print(trackids, subids)
        # print("CONSISTENCY", len(trackids), len(features))

        # delete old tracks
        # self.__delete_old_tracks()

        # return trackids
        return trackids, subids

def MatchBoxes(inBoxes, dbBoxes, iou_thresh):
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

    # print("IOU ", inBoxes)
    # print("IOU ", dbBoxes)
    num_curr = len(inBoxes)  # "current frame" features
    num_prev = len(dbBoxes)  # "prev frame" or "database" features

    #feat_curr = self.__crypto.decrypt_vec(mat_curr, (num_curr, -1))
    #feat_prev = self.__crypto.decrypt_vec(mat_prev, (num_prev, -1))

    if debug:
        print("=" * 20)
        print("iou threshold:", iou_thresh)
        # print("mat_prev.shape:", mat_prev.shape)
        # print("mat_curr.shape:", mat_curr.shape)
        # print("feat_prev.shape:", feat_prev.shape)
        # print("feat_curr.shape:", feat_curr.shape)

    # Similarity matrix (rows: previous, cols: current):
    #sim = cosine_similarity(feat_prev, feat_curr)
    iou_mat = np.zeros((num_prev, num_curr), dtype="float32")
    for i in range(0, len(dbBoxes)):
        bb1 = dbBoxes[i]
        for j in range(0, len(inBoxes)):
            bb2 = inBoxes[j]
            iou_mat[i,j] = get_iou(bb1, bb2)
    # print(iou_mat)

    # Return matrix with num_curr elements, all set to -1:
    return_indices = np.full(num_curr, -1)
    iou_scores = np.full(num_curr, -1.0)

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
        best_idx = np.unravel_index(np.argmax(iou_mat), shape=iou_mat.shape)

        if debug:
            print("i:", i)
            # print("sim:\n{}".format(sim))
            # print("best_idx:", best_idx)
            print("iou_mat[best_idx].item()", iou_mat[best_idx].item())

        if iou_mat[best_idx] < iou_thresh:
            # This means there are no more similarity scores above the threshold (default=0.3)
            # By breaking, the remaining return_indices will be left as -1.
            # print(">> break\n")
            break
        else:
            # Use best_idx col value to select the relevant element in return_indices,
            # then set that element to the best_idx row value, which is the "best match"
            # from feat_existing:
            return_indices[best_idx[1]] = best_idx[0]
            iou_scores[best_idx[1]] = iou_mat[best_idx]

            # Set the whole column to -1. "This face has been checked already."
            iou_mat[:, best_idx[1]] = -1

            # Set the entire row to -1 for next iteration, to prevent multiple columns
            # being matched to one row (e.g. "two faces to the same name"):
            iou_mat[best_idx[0], :] = -1

            if debug:
                print("return_indices: {}".format(return_indices))
                print()

    if debug:
        print("return_indices: {}".format(return_indices))

    # print(return_indices, sim_scores)
    return return_indices, iou_scores

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
    print(sim)

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

def InitDatabase(db_file, verbose=False):
    flag = False
    #print("Reading body database ", db_file)
    database = Database(db_file)
    print(database)
    if database.is_empty():
        flag = True
        logging.warning("No entries in database for comparison.")
    db_features = database.get_features_as_array()  # For use in Match_Old()
    #print(database.get_name_list())
    return database, db_features, flag
