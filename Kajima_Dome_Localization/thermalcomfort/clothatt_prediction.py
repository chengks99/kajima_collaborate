import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from skimage import transform

# from mmcv import Config
from builder import build_predictor
from checkpoint import load_checkpoint
import roi_predictor_resnet as clothcfg


class ClothPredictor():
    def __init__(self, ca_model, device):
        
        if device<0:
            self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cuda:{}".format(device))
        
        ## target img size
        self.crop_size = 224
        self.LMK_THRESH = 0.6
        
        self.h_top, self.w_top = (267, 264)
        self.h_bot, self.w_bot = (315, 184)
        self.h_full, self.w_full = (539, 264)

        ## upper body, lower body coordinates
        self.img_pts1 = np.float32([[208.,  25.], [ 61.,  20.], [173., 245.], [78., 241.]])
        self.img_pts2 = np.float32([[137.,  25.], [42.,  21.], [142., 168.], [ 38., 164.]])

        self.mean_top = np.array([
                [51.75, 16.78], [176.48, 20.97],
                [66.18, 202.18], [146.78, 205.54]], dtype=np.float32)
        self.mean_bot = np.array([
                [51.13, 14.93], [166.78, 17.77],
                [46.26, 116.62], [172.86, 119.46]], dtype=np.float32)
        self.mean_full = np.array([
                [51.75, 8.31], [176.48, 10.39],
                [66.18, 100.15], [146.78, 101.81]], dtype=np.float32)

        self.category_names = ["short_Sleeve_Top","long_Sleeve_Top","short_Sleeve_Outwear",
                                "long_Sleeve_Outwear","vest","sling","shorts","trousers","skirt",
                                "short_sleeve_Dress","long_sleeve_Dress","vest_Dress","sling_Dress"]
        
        self.iclomap = {"sling":0.12,"short_Sleeve_Top":0.19,"long_Sleeve_Top":0.34,
            "vest":0.13,"short_Sleeve_Outwear":0.245,"long_Sleeve_Outwear":0.36,
            "sling_Dress":0.23,"vest_Dress":0.27,"short_sleeve_Dress":0.29,"long_sleeve_Dress":0.4,
            "shorts":.08,"trousers":0.24,"skirt":0.185} 
        
        self.__load_model(ca_model)

        # dummy_input = torch.rand(1, 3, self.crop_size, self.crop_size)
        # dummy_input.to(self.device)
        # dummy_lmk = torch.from_numpy(np.asfarray([[[0,0]]*8])) 
        # dummy_lmk.to(self.device)
        # _, dummy_output = self.clothmodel(dummy_input, attr=None, landmark=dummy_lmk, return_loss=False)
        # del dummy_input, dummy_output

    def __load_model(self, ca_model):

        
        self.clothmodel = build_predictor(clothcfg.model)
        load_checkpoint(self.clothmodel, ca_model, map_location='cpu')

        self.clothmodel.eval()
        
        self.clothmodel.to(self.device)

    def __get_img_tensor(self, img, get_size=False):
        original_w, original_h = img.size

        img_size = (self.crop_size, self.crop_size)  # crop image to (224, 224)

        img.thumbnail(img_size, Image.ANTIALIAS)
        img = img.convert('RGB')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        img_tensor = transform(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)

        if get_size:
            return img_tensor, original_w, original_h
        return img_tensor

    def __align_imglmk(self, lmk, src_pts, tgt_pts, img, tgt_sizeh,tgt_sizew):
        tform = transform.SimilarityTransform()
        tform.estimate(src_pts, tgt_pts)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(
            img, M, (tgt_sizeh,tgt_sizew), borderValue=(255, 255, 255))
        transformed_lmk = M.dot(np.insert(lmk,2,1.0,axis=1).T).T
        return img, transformed_lmk

    def __preprocess(self, image_bgr, lmk):
        
        objPoints1, objPoints2 = np.array([lmk[5], lmk[6], lmk[11], lmk[12]]), np.array([
            lmk[11], lmk[12], lmk[13], lmk[14]])
        
        frame1,transformed_points1 = self.__align_imglmk(lmk, objPoints1, self.img_pts1, image_bgr.copy(), self.w_top, self.h_top)
        orih1, oriw1, _ = frame1.shape
        frame1 = cv2.resize(frame1, dsize=(self.crop_size, self.crop_size))
        
        frame2,transformed_points2 = self.__align_imglmk(lmk, objPoints2, self.img_pts2, image_bgr.copy(), self.w_bot, self.h_bot)
        orih2, oriw2, _ = frame2.shape
        frame2 = cv2.resize(frame2, dsize=(self.crop_size, self.crop_size))

        frame3,transformed_points3 = self.__align_imglmk(lmk, objPoints1, self.img_pts1, image_bgr.copy(), self.w_full, self.h_full)
        orih3, oriw3, _ = frame3.shape
        frame3 = cv2.resize(frame3, dsize=(self.crop_size, self.crop_size))

        ## require eight (5,6,7,8,11,12,13,14) skeleton points in order for cloth model 
        vlmk1 = []
        for i in [5, 6, 7, 8, 11, 12]:
            vlmk1.append([int(transformed_points1[i][0]*(self.crop_size/oriw1)),
                         int(transformed_points1[i][1]*(self.crop_size/orih1))])

        vlmk1.extend([[0,0]]*2)
        
        vlmk2 = [[0,0]]*4
        for i in [11, 12, 13, 14]:
            vlmk2.append([int(transformed_points2[i][0]*(self.crop_size/oriw2)),
                         int(transformed_points2[i][1]*(self.crop_size/orih2))])
            
        vlmk3 = []
        for i in [5, 6, 7, 8, 11, 12, 13, 14]:
            vlmk3.append([int(transformed_points3[i][0]*(self.crop_size/oriw3)),
                         int(transformed_points3[i][1]*(self.crop_size/orih3))])

        return frame1, frame2, frame3, np.array([np.array(vlmk1), np.array(vlmk2), np.array(vlmk3)])

    def __preprocess_new(self, image_bgr, lmk):
        dst_top = np.array([[lmk[6][0],lmk[6][1]], [lmk[5][0],lmk[5][1]],
                       [lmk[12][0],lmk[12][1]], [lmk[11][0],lmk[11][1]]], dtype=np.float32)

        dst_bot = np.array([[lmk[12][0],lmk[12][1]], [lmk[11][0],lmk[11][1]],
                       [lmk[14][0],lmk[14][1]], [lmk[13][0],lmk[13][1]]], dtype=np.float32)

        dst_full = np.array([[lmk[6][0],lmk[6][1]], [lmk[5][0],lmk[5][1]],
                       [lmk[12][0],lmk[12][1]], [lmk[11][0],lmk[11][1]]], dtype=np.float32)

        tform = transform.SimilarityTransform()

        tform.estimate(dst_top, self.mean_top)
        M = tform.params[0:2, :]
        top = cv2.warpAffine(image_bgr, M, (self.crop_size, self.crop_size), borderValue=0.0)
        top_lmk = M.dot(np.insert(lmk, 2, 1.0, axis=1).T).T

        tform.estimate(dst_bot, self.mean_bot)
        M = tform.params[0:2, :]
        bot = cv2.warpAffine(image_bgr, M, (self.crop_size, self.crop_size), borderValue=0.0)
        bot_lmk = M.dot(np.insert(lmk, 2, 1.0, axis=1).T).T

        tform.estimate(dst_full, self.mean_full)
        M = tform.params[0:2, :]
        full = cv2.warpAffine(image_bgr, M, (self.crop_size, self.crop_size), borderValue=0.0)
        full_lmk = M.dot(np.insert(lmk, 2, 1.0, axis=1).T).T
 
        ## require eight (5,6,7,8,11,12,13,14) skeleton points in order for cloth model 
        vlmk1 = []
        for i in [5, 6, 7, 8, 11, 12]:
            vlmk1.append([int(top_lmk[i][0]), int(top_lmk[i][1])])
        vlmk1.extend([[0,0]]*2)
        
        vlmk2 = [[0,0]]*4
        for i in [11, 12, 13, 14]:
            vlmk2.append([int(bot_lmk[i][0]), int(bot_lmk[i][1])])
            
        vlmk3 = []
        for i in [5, 6, 7, 8, 11, 12, 13, 14]:
            vlmk3.append([int(full_lmk[i][0]), int(full_lmk[i][1])])
       
        return top, bot, full, np.array([np.array(vlmk1), np.array(vlmk2), np.array(vlmk3)])

    def model_inference(self, image_bgr, lmks, lmk_confs):
        cloth_attributes = []
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for lmk, lmk_conf in zip(lmks, lmk_confs):
#            frame1, frame2, frame3, rlmk = self.__preprocess(image_bgr, lmk)
#            frame1, frame2, frame3, rlmk = self.__preprocess_new(image_bgr, lmk)
            frame1, frame2, frame3, rlmk = self.__preprocess_new(image_rgb, lmk)

#            cv2.imwrite("frame1.jpg", frame1)
#            cv2.imwrite("frame2.jpg", frame2)
#            cv2.imwrite("frame3.jpg", frame3)
            
#            print("RLMK\n", rlmk)
            
#            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame1 = Image.fromarray(frame1)
            img_tensor1 = self.__get_img_tensor(frame1)
            
#            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame2 = Image.fromarray(frame2)
            img_tensor2 = self.__get_img_tensor(frame2)
            
#            frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            frame3 = Image.fromarray(frame3)
            img_tensor3 = self.__get_img_tensor(frame3)

            lm1, lm2, lm3 = rlmk[0], rlmk[1], rlmk[2]
            landmark_tensor1 = torch.from_numpy(np.asfarray([lm1]))            
            landmark_tensor2 = torch.from_numpy(np.asfarray([lm2]))            
            landmark_tensor3 = torch.from_numpy(np.asfarray([lm3]))

            imagesStack = torch.cat(
                tuple((img_tensor1, img_tensor2, img_tensor3))).to(self.device)
            landmarksStack = torch.cat(
                tuple((landmark_tensor1, landmark_tensor2, landmark_tensor3))).to(self.device)

            _, cate_prob = self.clothmodel(
                imagesStack, attr=None, landmark=landmarksStack, return_loss=False)
            
            #top 
            cateValue1 = cate_prob[0].data.cpu().numpy()[0:6] 
            df2catt = self.category_names[0:6][cateValue1.argmax()]
            
            #bottom
            cateValue2 = cate_prob[1].data.cpu().numpy()[6:9] 
            df2catb = self.category_names[6:9][cateValue2.argmax()]
            
            #full
            cateValue3 = cate_prob[2].data.cpu().numpy()[9:13] 
            df2catf = self.category_names[9:13][cateValue3.argmax()]
            
            
            #print("Cloth ATT ", df2catt, df2catb, df2catf)
            clothes = {'top': [df2catt, max(cateValue1)], 'bot': [df2catb, max(cateValue2)], 'full': [df2catf, max(cateValue3)]}
            
            if clothes['top'][1] > clothes['full'][1] or clothes['bot'][1] > clothes['full'][1]:
                iclot = self.iclomap[df2catt]
                if lmk_conf[11][0] > self.LMK_THRESH and lmk_conf[12][0] > self.LMK_THRESH and lmk_conf[13][0] > self.LMK_THRESH and lmk_conf[14][0] > self.LMK_THRESH:
                    iclob = self.iclomap[df2catb]
                    iclo = iclot+iclob
                else:
                    iclo = 2*iclot
            else:
                iclo = self.iclomap[df2catf]

            clothes['iclo'] = iclo
            cloth_attributes.append(clothes)
            
        return cloth_attributes