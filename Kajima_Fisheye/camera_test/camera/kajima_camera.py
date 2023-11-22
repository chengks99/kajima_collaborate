import mysql.connector
import cv2
import subprocess
import time
import base64
import os
import numpy as np
from io import BytesIO
import shutil
from queue import Queue
import sys
import argparse
import glob
import matplotlib.pyplot as plt
from datetime import datetime

# Joe's Library
from camera import Camera
from microphone import Microphone 
from floorplan import Floorplan
from Localization2 import Localization
from PoseEstimation import *
from common.initialise import *
import json
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
def format_time() :
    t = datetime.now()
    s = t.strftime('%Y%m%d%H%M%S%f')
    return int(s[:-3])
##### 1. Initialize Localization Module, JK's Engine and Camera Pulling (tent.)
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

def overlay_text(img, font_size, *texts):
    # img = image.copy()
    for i, text in enumerate(texts):
        cv2.putText(img, text, (10, (i+1)*50), 0, font_size, color=BGRColor.RED, thickness=3)
        cv2.putText(img, text, (10, (i+1)*50), 0, font_size, color=BGRColor.WHITE, thickness=2)
    return img

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

    box[0] = box[0] - box[2]/2
    box[1] = box[1] - box[3]/2

    bbox = list(map(int,box))
    # Draw bounding box and label:
    # label = '{}_{}'.format(label, bbox[2])
    textsize = cv2.getTextSize(label, 0, font_size, thickness=3)[0]
    # cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color=color, thickness=2)
    cv2.rectangle(img, (bbox[0], bbox[1]-5-textsize[1]), (bbox[0]+textsize[0], bbox[1]-5), color=BGRColor.BLUE, thickness=-1)
    cv2.putText(img, label, (bbox[0], bbox[1] - 5), fontFace=0, fontScale=font_size, color=BGRColor.YELLOW, thickness=4)
    cv2.putText(img, label, (bbox[0], bbox[1] - 5), fontFace=0, fontScale=font_size, color=color, thickness=3)

    # print(bbox[2]-bbox[0])
    return img

class detectengine(object) :

    def __init__(self,config_person) :

        self.PersonEngine = init_person_engine(config_person) 
        print("init succesful")
        self.input_q = Queue(1)  # fps is better if queue is higher but then more lags
        self.detect_q = Queue()
        self.skeleton_q = Queue()
        self.feature_q = Queue()
        self.output_q = Queue()

        self.detect_t = Thread(target=self.PersonEngine.detection_engine.worker, args=(self.input_q, self.detect_q))
        self.detect_t.daemon = True
        self.detect_t.start()
        print("detect_t succesful")
        self.skeleton_t = Thread(target=self.PersonEngine.skeleton_engine.worker, args=(self.detect_q, self.skeleton_q))
        self.skeleton_t.daemon = True
        self.skeleton_t.start()
        print("skeleton_t succesful")
        self.feature_t = Thread(target=self.PersonEngine.feature_engine.worker, args=(self.skeleton_q, self.feature_q))
        self.feature_t.daemon = True
        self.feature_t.start()
        print("feature_t succesful")
        self.person_t = Thread(target=self.PersonEngine.worker, args=(self.feature_q, self.output_q))
        self.person_t.daemon = True
        self.person_t.start()
        print("person_t succesful")
    def run(self,image):

        self.input_q.put(image)
        if self.output_q.empty():
            self.output = None
            pass  # fill up queue
        else:
            self.output = self.output_q.get()

        return self.output

def check_dups(results1,results2) :
    print(results1)
    print(results2)
    keys1 = list(results1.keys())
    keys2 = list(results2.keys())

    duplicate = {}
    common_ID = list(set(keys1).intersection(keys2))
    count = 0
    for i in common_ID :
        if i == "999" :
            temp1 = results1.pop(i)
            temp2 = results2.pop(i)
            for j in temp2 :
                temp1.append(j)
            duplicate['999'] = temp1
            continue

        else :
            current = []
            dups1 = results2.pop(i)
            dups2 = results1.pop(i)
            current.append(dups1)
            # current.append(dups2)
            duplicate[i] = current

    results = {**results1,**results2,**duplicate}
    return results

def access_database(camera_id,conn,config) :
    camera = Camera(camera_id,conn)
    print("camera accessed")
    floorplan = Floorplan(int(camera.floor_id),conn)
    print("floorplan accessed")
    microphone = Microphone(int(camera.floor_id),conn)
    print("microphone accessed")

    net = Net()
    net.to(device)
    model_location = "/home/jk/Release/data/Camera" + str(camera_id) + "/pose1.pth"

    Localize = Localization(camera,net,floorplan,model_location,conn,save = False) 
    print("Localize accessed")
    jkmodel = detectengine(config)
    print("jkmodel accessed")
    return Localize , jkmodel

def minusone (list1) :
    minusone = []
    for i,j in zip(list1,list1[1:]) :
        minusone.append(j-i)

    return minusone

def processed_image(image,information) :

    # image = image[200:2792,200:2792,:]

    output = information[1].run(image)

    if output : 
        boxes, person_crop, face_labels, tids, lmks, image = output
        localize = []

        for i in range(len(tids)):
            label = face_labels[i] 
            if face_labels[i] == 'UnK':
                box_color = BGRColor.RED
            else:
                box_color = BGRColor.WHITE
            print(" got output 2")
            image = draw_box(image, boxes[i], label, color=box_color, font_size=2.0)
            cur_localize = [count,tids[i],face_labels[i]]
            bb = boxes[i]
            lmk = lmks[i]

            for bbox in range(len(bb)):
                cur_localize.append(bb[bbox].item())

            for point in range(len(lmk)):
                cur_localize.append(lmk[point,0])
                cur_localize.append(lmk[point,1])

            localize.append(cur_localize)
        # print(localize)
        if len(localize) == 0 :
            return {},image
        else : 
            current_location = information[0].find_location(localize)
    else :
        return {},image
        
    return current_location,image

if __name__ == "__main__" :
    import pathlib
    scriptpath = pathlib.Path(__file__).parent.resolve()
    sys.path.append(str(scriptpath.parent / 'common'))

    import argsutils as au
    from adaptor import add_common_adaptor_args

    parser = au.init_parser('Kajima Camera Application')
    add_common_adaptor_args(parser, type='air', id='Camera1')
    args = au.parse_args(parser)


    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'    

    ### Configuration 
    path0 = './config/config_body0.json'
    path1 = './config/config_body1.json'
    path2 = './config/config_body2.json'

    config_camera = './config/config_camera.json'

    ##### 2. Connect to Database
    print("test connect")
    connection = mysql.connector.connect(
        host = "127.0.0.1",
        user = "root",
        password = "Welcome123",
        database = "mockup_db",
        port = "3306"
    )
    print("can connect")

    ### Test case whether engine can be done inside another class
    ### Initialize database and model for each camera 
    # camera_ids = ["camera22","camera23","camera24","camera25"]
    camera_ids = ["camera18","camera19","camera20","camera21","camera22"]
    # camera_ids = ["camera007101","camera007102","camera007103","camera007104"]
    # camera_ids = ["camera20"]
    information = {}
    print("Im here #1")
    count = 0 
    for i in camera_ids :
        cam_id = str(i[6:])
        print(cam_id)
        if count % 3 == 0 :
            cur_localize, cur_engine = access_database(cam_id,connection,path0)
        if count % 3 == 1 :
            cur_localize, cur_engine = access_database(cam_id,connection,path1)
        else :
            cur_localize, cur_engine = access_database(cam_id,connection,path2)
        print("can access data")
        stream = init_stream(config_camera, cur_localize.camera.source, 0).start()
        print("can access stream")
        information[cam_id] = [cur_localize,cur_engine,stream]
        count += 1

    print("Initialize success")
    tot_timestamp1 = []
    tot_timestamp2 = []
    tot_timestamp3 = []
    tot_timestamp4 = []
    tot_timestamp5 = []
    tot_timestamp6 = []
    count = 0
    try : 
        keys = sorted(list(information.keys()))
        while (True) :
            ### Read Image then check time stamp
            cur_time = format_time()
            # print(information)
            _,current_image1, timestamp1 = information[keys[0]][2].read()
            _,current_image2, timestamp2 = information[keys[1]][2].read()
            _,current_image3, timestamp3 = information[keys[2]][2].read()
            _,current_image4, timestamp4 = information[keys[3]][2].read()
            # _,current_image5, timestamp5 = information[keys[4]][2].read()      


            camera_location1,image1 = processed_image(current_image1,information[keys[0]])
            camera_location2,image2 = processed_image(current_image2,information[keys[1]])
            camera_location3,image3 = processed_image(current_image3,information[keys[2]])
            camera_location4,image4 = processed_image(current_image4,information[keys[3]])
            # camera_location5,image5 = processed_image(current_image5,information[keys[4]])
            # print(camera_location1)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            # print(i)
            # print(camera_location1)
            # print(camera_location2)
            # print(camera_location3)
            # print(camera_location4)
            # print(camera_location5)
            # print(timestamp1)
            # tot_timestamp1.append(timestamp1)
            # # print(timestamp2)
            # tot_timestamp2.append(timestamp2)
            # # print(timestamp3)
            # tot_timestamp3.append(timestamp3)
            # print(timestamp4)
            # tot_timestamp4.append(timestamp4)
            # print(timestamp5)
            # tot_timestamp5.append(timestamp5)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            combine_camera = check_dups(camera_location1,camera_location2)
            combine_camera = check_dups(combine_camera,camera_location3)
            combine_camera = check_dups(combine_camera,camera_location4)
            # combine_camera = check_dups(combine_camera,camera_location5)

            
            ## Check Image and put it into the thread for management system
            if combine_camera == {} :
                # res = {}
                # with open("./Results/results/A184" + str(cur_time) + '.json','w+') as json_file :
                #     json.dump(res,json_file)
                continue
            else :
                try : 
                    with open("./Results/results/A184" + str(cur_time) + '.json','w+') as json_file :
                        json.dump(combine_camera,json_file)
                except :
                    continue
                # cv2.imwrite("/home/jk/Release/data/Camera18/" + str(cur_time) + ".png",image1)
                # cv2.imwrite("/home/jk/Release/data/Camera19/" + str(cur_time) + ".png",image2)
                # cv2.imwrite("/home/jk/Release/data/Camera20/" + str(cur_time) + ".png",image3)
                # cv2.imwrite("/home/jk/Release/data/Camera21/" + str(cur_time) + ".png",image4)
                # cv2.imwrite("/home/jk/Release/data/Camera23/" + str(cur_time) + ".png",image5)
            count += 1
    except Exception as e:
        trace_back = sys.exc_info()[2]
        line = trace_back.tb_lineno
        raise FlowException("Process Exception in line {}".format(line), e)

    # information[keys[0]][2].stop()
    # information[keys[1]][2].stop()
    # information[keys[2]][2].stop()
    # information[keys[3]][2].stop()
    # information[keys[4]][2].stop()


    # new1 = minusone(tot_timestamp1)
    # new2 = minusone(tot_timestamp2)
    # new3 = minusone(tot_timestamp3)
    # new4 = minusone(tot_timestamp4)
    # new5 = minusone(tot_timestamp5)

    # time = list(range(1,11))
    # print("TTTTTTTTTTTTTTTTTTTTTTTT")
    # print(len(time))
    # print(len(new1))
    # print("TTTTTTTTTTTTTTTTTTTTTTTT")
    # plt.plot(time,new1,label = "camera18")
    # plt.plot(time,new2,label = "camera19")
    # plt.plot(time,new3,label = "camera20")
    # plt.plot(time,new4,label = "camera21")
    # plt.plot(time,new5,label = "camera22")
    # plt.plot(time,new6,label = "camera23")

    # # naming the x axis 
    # plt.xlabel('frame number')
    # # naming the y axis
    # plt.ylabel('time difference each frame')

    # plt.title('Test exponential or linear')

    # plt.legend()

    # plt.savefig('testtest.png')
    

    # ### Test Image into the Localization Module
    # start_time = time.time() 
    # # list_image = glob.glob("./Raw/*.png")
    # # print(list_image)

    # # print(information)
    # # print(information[20][1])
    # ### Feed image into the engine
    

    # print("HERE")

    # avg_time = 0
    # t1 = time.time()
    # keys = information.keys()
    # count = 0

    # print("yes")
    # try :
    #     for i in range(100) :
    #         loop_start_time = time.time()
    #         for j in keys : 

    #             ### Pre-processed Image 
    #             print(i)
    #             # current_image = cv2.imread(i)
    #             ret, current_image = stream.read()
    #             # cv2.imwrite("./Image/count_" + str(count) + ".png",current_image)
    #             current_image = current_image[200:2792,200:2792,:]
    #             # cv2.imwrite("./Image/count_" + str(count) + ".png",current_image)
    #             ### Check information on the run image
    #             # print(current_image)
    #             output = information[j][1].run(current_image)

    #             ### Check running time
    #             t2 = time.time() - t1
    #             avg_time += t2
    #             print("Detection time ", t2)
    #             # print(output)
    #             ### Unpack JK's Engine Results then feed into Localization
    #             if output :
    #                 print(" got output 1")
    #                 boxes, person_crop, face_labels, tids, lmks, image = output  # Unpack output: (N x 5) and (N x DIM)
    #                 print(boxes)
    #                 localize = []

    #                 for i in range(len(tids)):
    #                     label = face_labels[i] 
    #                     if face_labels[i] == 'UnK':
    #                         box_color = BGRColor.RED
    #                     else:
    #                         box_color = BGRColor.WHITE
    #                     print(" got output 2")
    #                     image = draw_box(image, boxes[i], label, color=box_color, font_size=2.0)

    #                     loop_dur = time.time() - loop_start_time

    #                     calc_fps = 1 / loop_dur
    #                     info_text = "Run time: {:.3f} s".format(time.time() - start_time)
    #                     loop_t_text = "Frame time: {:.3f} s".format(loop_dur)
    #                     fps_text = "Calculated FPS: {:.3f}".format(calc_fps)
    #                     image = overlay_text(image, 1.0, info_text, loop_t_text, fps_text)

    #                 cv2.imwrite("./Image/count_" + str(count) + ".png",image)
    #                 print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx")
    #                 print("WRITING SUCCESS")
    #                 print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    #                 # print(face_labels)

    #         count += 1  
    # except Exception as e:
    #     trace_back = sys.exc_info()[2]
    #     line = trace_back.tb_lineno
    #     raise FlowException("Process Exception in line {}".format(line), e)