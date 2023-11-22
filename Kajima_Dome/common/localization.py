from collections import Iterable
import datetime as dt
import numpy as np
from math import *

class Localize(object) :
    def __init__(self,camera) :
        self.camera = camera

        self.generate_lookup_table()

        self.combined_results = {}

        ###### ASSUMING IMPLEMENTATION OF REDIS HERE

    def into_dict(self,results) :
        dictionary_results = {}
        
        boxes, person_crop, face_labels, tids, lmks, image, pmvs, actions  = results

        ### TIDS is consisting information such as the tracking ID of the person (only number has no meaning) -> useful when comparing it with previous frame (need to saved the previous frame)
        dictionary_results["tids"] = tids

        ### Face Lables is consisting information that has the human (take the last 3 digit usually and the compared with the one on the database)
        dictionary_results["labels"] = face_labels

        ### Head points/nose information useful for doing localization
        head_point_x = []
        head_point_y = []
        for i in range(len(tids)) :
            head_point_x.append(lmks[i][0,0])
            head_point_y.append(lmks[i][0,1])

        dictionary_results["head_points_x"] = head_point_x
        dictionary_results["head_points_y"] = head_point_y
        
        ### PMV Results, will be saved into the database directly has no information needed for the localization
        dictionary_results["pmvs"] = pmvs

        ### Actions of the said person - Standing or Sitting
        dictionary_results["pose"] = actions

        return dictionary_results

    def find_location(self,info) :
        now = dt.datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
        
        ## Generate and Split points depending on the pose
        self.get_point(info)

        ## Undistort the points depending on the pose then do homography matrix

        ## Standing Pose 
        self.combined_results[dt_string] = []
        for point in self.opf_standing :
            new_point = self.undistort_points([point[0],point[1]])
            new_point = self.floorplan_location_standing(new_point)
            self.combined_results[dt_string].append([new_point[0],new_point[1],point[2]])

        ## Sitting Pose 

        for point in self.opf_sitting :
            new_point = self.undistort_points([point[0],point[1]])
            new_point = self.floorplan_location_sitting(new_point)
            self.combined_results[dt_string].append([new_point[0],new_point[1],point[2]])
        

        ### Need to apply tracking on this part of the code, algorithm thinking, one person should be able
        ### to stay in a frame for atleast another 2 seconds before he moves on, therefore need to recheck if the person 
        ### is out or not from the scene, rare cases happend when someone is occluded but even when someone is occluded the person 
        ### is still there, therefore it will not be much of a problem. -> One way to implement this is by putting the age on the ID of the person along for the updates.
        ### It will record only the last dt_string then -> compared it, at the same time it will remove the one that already aging in this case 3 but can be configurable.

        return self.combined_results, now

    def floorplan_location_standing(self,point) : 
        
        px = (self.camera.hom_mat_stand[0][0]*point[0] + self.camera.hom_mat_stand[0][1]*point[1] + self.camera.hom_mat_stand[0][2]) / ((self.camera.hom_mat_stand[2][0]*point[0] + self.camera.hom_mat_stand[2][1]*point[1] + self.camera.hom_mat_stand[2][2]))
        py = (self.camera.hom_mat_stand[1][0]*point[0] + self.camera.hom_mat_stand[1][1]*point[1] + self.camera.hom_mat_stand[1][2]) / ((self.camera.hom_mat_stand[2][0]*point[0] + self.camera.hom_mat_stand[2][1]*point[1] + self.camera.hom_mat_stand[2][2]))

        pp = [px,py]

        return pp

    def floorplan_location_sitting(self,point) : 
        
        px = (self.camera.hom_mat_sit[0][0]*point[0] + self.camera.hom_mat_sit[0][1]*point[1] + self.camera.hom_mat_sit[0][2]) / ((self.camera.hom_mat_sit[2][0]*point[0] + self.camera.hom_mat_sit[2][1]*point[1] + self.camera.hom_mat_sit[2][2]))
        py = (self.camera.hom_mat_sit[1][0]*point[0] + self.camera.hom_mat_sit[1][1]*point[1] + self.camera.hom_mat_sit[1][2]) / ((self.camera.hom_mat_sit[2][0]*point[0] + self.camera.hom_mat_sit[2][1]*point[1] + self.camera.hom_mat_sit[2][2]))

        pp = [px,py]

        return pp

    def flatten(self,lists) :
        for item in lists:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in self.flatten(item):
                    yield x
            else:        
                yield item 

    def get_point(self,results) :

        results = self.into_dict(results)

        self.tids = results["tids"]
        self.face_labels = results["labels"]
        self.heads_x = results["head_points_x"]
        self.heads_y = results["head_points_y"]
        self.poses = results["pose"]

        self.opf_standing = [] ## Output per framne
        self.opf_sitting = []

        tid_threshold = []

        for tid,face_label,head_x,head_y,pose in zip(self.tids,self.face_labels,self.heads_x,self.heads_y,self.poses) :
            if tid not in tid_threshold :
                tid_threshold.append(tid)
                if face_label == "UnK" :
                    face_label = "Unknown"
                
                if pose == 0 : ## Standing
                    self.opf_standing.append([head_x,head_y,face_label])


                elif pose == 1 : ## Sitting
                    self.opf_sitting.append([head_x,head_y,face_label])

            else :
                continue 

        
    def generate_lookup_table(self,different = 0.05) :
        self.table = {}
        angle = np.arange(0.0,90.0,different)

        ud_vector_temp = np.squeeze(self.camera.ud_vector)
        k1,k2,k3,k4 = ud_vector_temp

        for j in angle :
            theta = radians(j)

            theta_d = theta * (1 + k1 * (theta)**2 + k2 * (theta)**4 + k3 * (theta)**6 + k4 * (theta)**8)

            self.table[-theta_d] = -theta
            self.table[theta_d] = theta


    def find_nearest(self,array, value):
        array = np.asarray(array)
        array2 = array.astype(np.float32)

        idx = (np.abs(array2 - value)).argmin()
        return array[idx]

    def undistort_points(self,point) :

        cam_int = np.asarray(self.camera.cam_int)

        fx = cam_int[0,0]
        fy = cam_int[1,1]
        cx = cam_int[0,2]
        cy = cam_int[1,2]

        original_point = point

        sphere_point   = [(point[0]-cx)/fx,(point[1]-cy)/fy]

        theta_d        = sqrt(sphere_point[0]**2 + sphere_point[1]**2)

        number_list    = list(self.table.keys())

        chosen_number  = self.find_nearest(number_list,theta_d)

        theta          = self.table[chosen_number]

        scale          = tan(theta) / theta_d

        pi             = [i * scale for i in sphere_point]
        pi             = np.array([pi[0],pi[1],1])

        new_pixel      = np.matmul(cam_int,pi)

        ud_pixel       = [new_pixel[0]/new_pixel[2],new_pixel[1]/new_pixel[2]]

        return ud_pixel

'''
import cv2 
import os
import json 
import datetime
import pandas as pd
import torch
import torchvision
import math
from math import *
from time import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PoseEstimation import skeletonDataset
from io import BytesIO
from datetime import datetime
from collections import Iterable
from save_image import save_frame

class Localization(object) :

    def __init__(self,Camera,Net,Floorplan,model,db,save = True) :
        
        self.db = db
        self.mycursor = self.db.cursor()
        self.floorplan = Floorplan
        self.camera = Camera
        # self.microphone = Microphone

        self.net = Net
        self.net = torch.load(model).eval()
        self.save = True
        with open(self.camera.lookup) as f :
            self.degree_table = json.load(f)

    def flatten(self,lis) :
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, str):
                for x in self.flatten(item):
                    yield x
            else:        
                yield item

    def find_location(self,csv_table) :
        now = datetime.now()
        dt_string = now.strftime('%Y-%m-%d %H:%M:%S.%f')
        microsecond = int(now.strftime("%f"))
        
        self.tracking_table, self.number, self.column_name = self.load_csv(csv_table)
        dataset = self.tracked_table() 
        # print(dataset)
        # print(dataset.shape)
        # dataset_test = skeletonDataset(dataset)


        # testloader = torch.utils.data.DataLoader(dataset_test, batch_size=256,
        #                                  shuffle=False, num_workers=1)

        # dataset = dataset[:, :34].astype(np.float64)
        # dataset = np.squeeze(dataset, 0)
        # dataset = torch.tensor(dataset).float()
        raw_results = self.check_accuracy(dataset,self.net)
        
        results = list(self.flatten(raw_results))

        pose_track = self.check_pose(results)

        location = self.find_pixel(pose_track)
        print("##################################")
        print("##################################")
        print(location)
        print("THIS FRAME HAS {} PEOPLE DETECTED".format(len(location.keys())))
        print("##################################")
        print("##################################")
        current_floorplan = self.floorplan.image.copy()
        save_frame(current_floorplan,location)

        if not location :
            return None
        
        new_location = {}
        for i in location.keys() :
            sql_command = "SELECT human_id FROM human_table WHERE first_name= '{}'".format(i)
            print(sql_command)
            self.mycursor.execute(sql_command)

            sql_query = self.mycursor.fetchall()
            if len(sql_query) == 1 :
                for x in sql_query :
                    human_id = x[0]
            new_location[human_id] = location[i]

        return new_location

        # sql_command = "INSERT INTO location_table (cam_id,loc_x,loc_y,time,human_id,microsecond) VALUES (%s,%s,%s,%s,%s,%s)"

        # value = []

        # for i in new_location.keys() :
        #     pos_x = new_location[i][0]
        #     pos_y = new_location[i][1]
        #     human_id = i
        #     value.append((self.camera.id,pos_x,pos_y,dt_string,human_id,microsecond))  


        # self.mycursor.executemany(sql_command,value)

        # self.db.commit()

        # if self.save == True : 

        #     sql_command0 = "SELECT * FROM socdist_table WHERE id=(SELECT id FROM rules_table WHERE id=(SELECT rules_id FROM ai2rules_table WHERE task2cam_id=(SELECT id FROM task2cam_table WHERE cam_id={})))".format(self.camera.id)
        #     self.mycursor.execute(sql_command0)

        #     sql_query = self.mycursor.fetchall()

        #     if len(sql_query) == 1 :
        #         for x in sql_query :
        #             rules_id = x[0]
        #             threshold = float(x[1])

        #     real_location = self.real_distance(location)

        #     violation = self.check_socdist(real_location,threshold)  #[[id1,id2,dist],[...]]
        #     if len(violation) == 0 :
        #         return location
        #     else :
        #         sql_command = "INSERT INTO incident_table (datetime,unique_id) VALUES (%s,%s)"

        #         data_amount = len(violation)

        #         val = [(dt_string,1)] * data_amount
        #         # print(val)
        #         self.mycursor.executemany(sql_command,val)

        #         self.db.commit()

        #         sql_command2 = "INSERT INTO socdistincident_table (incident_id,datetime,human_id,cam_id,location_id,socdist_id) VALUES (%s,%s,%s,%s,%s,%s)"
        #         val2 = []

        #         sql_command3 = "SELECT MAX(incident_id) FROM incident_table WHERE unique_id=1"

        #         self.mycursor.execute(sql_command3)

        #         sql_query = self.mycursor.fetchall()

        #         if len(sql_query) == 1 :
        #             for x in sql_query :
        #                 # print(x)
        #                 number = x[0] - data_amount + 1

        #         for i in range(data_amount) :
        #             sql_command4 = "SELECT MAX(id) FROM location_table WHERE human_id = {}".format(violation[i][0])
        #             self.mycursor.execute(sql_command4)

        #             sql_query = self.mycursor.fetchall()


        #             if len(sql_query) == 1 :
        #                 for x in sql_query :
        #                     location_id1 = x[0]
        #             sql_command5 = "SELECT MAX(id) FROM location_table WHERE human_id = {}".format(violation[i][1])
        #             self.mycursor.execute(sql_command5)

        #             sql_query = self.mycursor.fetchall()

        #             if len(sql_query) == 1 :
        #                 for x in sql_query :
        #                     location_id2 = x[0]

        #             val2.append((number,dt_string,violation[i][0],self.camera.id,location_id1,rules_id))
        #             val2.append((number,dt_string,violation[i][1],self.camera.id,location_id2,rules_id))
        #             number += 1
        #         self.mycursor.executemany(sql_command2,val2)

        #         self.db.commit()

        # return location

    def distance(self,point1,point2) :
        point1 =  [float(i) for i in point1]
        point2 =  [float(i) for i in point2]
        distance = math.sqrt((point1[0]-point2[0])**2 +(point1[1]-point2[1])**2)

        return distance

    def check_socdist(self,location,threshold) :
        
        violation = []
        all_keys = list(location.keys())
        length = len(all_keys)

        for i in range(0,length) :
            for j in range(i+1,length) :

                cur_dist = self.distance(location[all_keys[i]],location[all_keys[j]])

                if cur_dist < threshold :
                    violation.append([all_keys[i],all_keys[j],cur_dist])



        return violation

    def floorplan_location(self,points,pose) : 
        
        p = points

        if pose == "sitting" :
            px = (self.camera.hom_mat_sit[0][0]*p[0] + self.camera.hom_mat_sit[0][1]*p[1] + self.camera.hom_mat_sit[0][2]) / ((self.camera.hom_mat_sit[2][0]*p[0] + self.camera.hom_mat_sit[2][1]*p[1] + self.camera.hom_mat_sit[2][2]))
            py = (self.camera.hom_mat_sit[1][0]*p[0] + self.camera.hom_mat_sit[1][1]*p[1] + self.camera.hom_mat_sit[1][2]) / ((self.camera.hom_mat_sit[2][0]*p[0] + self.camera.hom_mat_sit[2][1]*p[1] + self.camera.hom_mat_sit[2][2]))

        elif pose == "standing" :
            px = (self.camera.hom_mat_stand[0][0]*p[0] + self.camera.hom_mat_stand[0][1]*p[1] + self.camera.hom_mat_stand[0][2]) / ((self.camera.hom_mat_stand[2][0]*p[0] + self.camera.hom_mat_stand[2][1]*p[1] + self.camera.hom_mat_stand[2][2]))
            py = (self.camera.hom_mat_stand[1][0]*p[0] + self.camera.hom_mat_stand[1][1]*p[1] + self.camera.hom_mat_stand[1][2]) / ((self.camera.hom_mat_stand[2][0]*p[0] + self.camera.hom_mat_stand[2][1]*p[1] + self.camera.hom_mat_stand[2][2]))

        p_after = (int(px),int(py))

        return p_after

    def real_distance(self,results) :
        
        real_position = {}

        height = self.floorplan.image.shape[0]
        width = self.floorplan.image.shape[1]

        for i in results.keys() :

            real_location_x = results[i][0] * self.floorplan.scale_x
            real_location_y = results[i][1] * self.floorplan.scale_y
            real_position[i] = [real_location_x,real_location_y]

        return real_position

    def find_pixel(self,info) :
        pixel_locations = {}
        current_info = info

        for i in current_info.keys() :
            current_headx = float(current_info[i][1][0])
            current_heady = float(current_info[i][1][1])

            head_pixel = [current_headx,current_heady]
            pose = current_info[i][0]
            undistort_head_pixel = self.undistort_points(head_pixel)
            location_floor = self.floorplan_location(undistort_head_pixel,pose)

            pixel_locations[i] = location_floor
            
        return pixel_locations 

    def find_nearest(self,array, value):
        array = np.asarray(array)
        array2 = array.astype(np.float32)

        idx = (np.abs(array2 - value)).argmin()
        return array[idx]

    def undistort_points(self,point) :

        cam_int = np.asarray(self.camera.cam_int)

        fx = cam_int[0,0]
        fy = cam_int[1,1]
        cx = cam_int[0,2]
        cy = cam_int[1,2]

        original_point = point

        sphere_point   = [(point[0]-cx)/fx,(point[1]-cy)/fy]

        theta_d        = sqrt(sphere_point[0]**2 + sphere_point[1]**2)

        number_list    = list(self.degree_table.keys())

        chosen_number  = self.find_nearest(number_list,theta_d)

        theta          = self.degree_table[chosen_number]

        scale          = tan(theta) / theta_d

        pi             = [i * scale for i in sphere_point]
        pi             = np.array([pi[0],pi[1],1])

        new_pixel      = np.matmul(cam_int,pi)

        ud_pixel       = [new_pixel[0]/new_pixel[2],new_pixel[1]/new_pixel[2]]

        return ud_pixel

    def check_pose(self,results) :
        current_results = {}
        count = 0
        head_x,head_y = self.tracking_table["lmk_x1"], self.tracking_table["lmk_y1"]
        
        for i in range(self.number) :
            if results[count] == 0 :
                pose = "standing"
            elif results[count] == 1 :
                pose = "sitting"

            if str(self.tracking_table["subid"].iloc[i][-3:]) == "UnK" :
                current_results["Unknown"] = [pose,[head_x.iloc[i],head_y.iloc[i]]]
            else :
                current_results[str(self.tracking_table["subid"].iloc[i][-3:])] = [pose,[head_x.iloc[i],head_y.iloc[i]]]

            count += 1
        
        return current_results
        
    def load_csv(self,csv_table) :
        
        try :
            if csv_table.split('.')[1] == "csv" :
                with open(csv_table) as myFile :
                    headrow = next(myFile)
                columns = [x.strip() for x in headrow.split(',')]
                data = pd.read_csv(csv_table, names=columns)

        except :
            columns = ["frame","tid","subid","cx","cy","width","height","angle","conf","lmk_x1","lmk_y1","lmk_x2","lmk_y2","lmk_x3","lmk_y3","lmk_x4","lmk_y4","lmk_x5","lmk_y5","lmk_x6","lmk_y6","lmk_x7","lmk_y7","lmk_x8","lmk_y8","lmk_x9","lmk_y9",'lmk_x10',"lmk_y10","lmk_x11",'lmk_y11','lmk_x12','lmk_y12',"lmk_x13","lmk_y13","lmk_x14","lmk_y14","lmk_x15","lmk_y15","lmk_x16","lmk_y16","lmk_x17","lmk_y17"]
            data = pd.DataFrame(csv_table,columns = columns)

        csv_data = data
        # csv_data = csv_data.drop(csv_data[csv_data.tid == "-1"].index)
        csv_data.loc[(csv_data.tid == '-1'),'subid']='1'
        anno_num = len(csv_data["frame"].index)

        return csv_data,anno_num,columns
    
    def annotate(self,range_list) :
        list1 = [None] * range_list

        return list1

    def convert(self,pointx,pointy) :

        radius = [2992/2,2992/2]

        theta = atan(sqrt(pointx**2 + pointy**2)/radius[0])
        psi = atan(pointy/pointx)

        new_point = [psi,theta]

        return psi, theta
        
    def tracked_table(self) :
        
        all_list = []
        new_column = self.column_name[9:]
        new_list = self.annotate(self.number)

        tid = self.tracking_table["subid"]
        frames = self.tracking_table["frame"]

        for i in range(self.number) :
            cur_list = []
            cur_id = tid.iloc[i][-3:]
            cur_frames = frames.iloc[i]
            it = iter(new_column)
            for j in it :
                cur_it = [j,next(it)]
                cur_x, cur_y = float(self.tracking_table[cur_it[0]].iloc[i]) , float(self.tracking_table[cur_it[1]].iloc[i])
                cur_psi,cur_theta = self.convert(cur_x,cur_y)
                cur_list.append(cur_psi)
                cur_list.append(cur_theta)

            all_list.append(cur_list)

        # temp_data = pd.DataFrame(all_list,columns = new_column)
        # dataset = temp_data.assign(pose = new_list)

        # return 
        return all_list

    # def check_accuracy(self, loader, model):
    #     # running_accuracy = 0
    #     num_samples = 0
    #     model.eval()
    #     new_out = []
    #     with torch.no_grad():
    #         for i in loader : #array [batch,size1,size2]

    #             inputs = torch.FloatTensor(loader)
    #             # for i, data in enumerate(loader, 0):
    #             # get the inputs; data is a list of [inputs, labels]
    #             # inputs = data
    #             # labels = labels.reshape(-1)
    #             inputs = inputs.cuda()
    #             # labels = labels.cuda()
    #             # print(inputs.shape)
    #             outputs = model(inputs)
    #             # running_accuracy += self.accuracy(outputs, labels)

    #             num_samples += 1
    #             results = torch.argmax(outputs, dim=1)
    #             results = results.tolist()
    #             new_results = results.copy()

    #             new_out.append(new_results)
                
    #     return new_out

    def check_accuracy(self, loader, model):
        # running_accuracy = 0
        num_samples = 0
        model.eval()
        new_out = []
        with torch.no_grad():
            # for i in loader : #array [batch,size1,size2]
            
            inputs = torch.FloatTensor(loader)
            # for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs = data
            # labels = labels.reshape(-1)
            inputs = inputs.cuda()
            # labels = labels.cuda()
            # print(inputs.shape)
            outputs = model(inputs)
            # running_accuracy += self.accuracy(outputs, labels)

            num_samples += 1
            print(outputs)
            results = torch.argmax(outputs, dim=1)
            print(results)
            results = results.tolist()
            new_results = results.copy()
            print(new_results)
            # new_out.append(new_results)
        # print(new_out)
        return new_results

    def accuracy(self,predictions, labels): 
        classes = torch.argmax(predictions, dim=1)

        return torch.mean((classes == labels).float())
'''