import os
import base64

import cv2
import mxnet as mx
import numpy as np
from skimage import transform
from sklearn.metrics.pairwise import cosine_similarity

import time
from collections import namedtuple

from face_embedding import FaceEmbedding
from retinaface import RetinaFace

class FaceEngine(object):
    def __init__(self, fd_model_path, fr_model_path,
                 fd_threshold=0.5, fd_input_resize=0, max_detected_faces=0,
                 fr_threshold=0.3,
                 device=-1, rgb=0,  emb_layer_name="fc1_output"):

        self.fd_model_path = fd_model_path
        self.fd_threshold = fd_threshold
        self.fd_input_resize = fd_input_resize
        self.max_detected_faces = max_detected_faces
        #self.rgb = rgb
        self.Batch = namedtuple("Batch", ["data"])
        
        self.fr_model_path = fr_model_path
        self.fr_threshold = fr_threshold
        self.device = mx.cpu(0)
        if device >= 0:
            self.device = mx.gpu(device)
        self.emb_layer_name = emb_layer_name

        print("Loading Face Detection Model...")
        self.__detector = RetinaFace(self.fd_model_path, 0, device, 'net3')
        print("Face Detector Loaded.\n")


        print("Loading Face Recognition Model...")
        self.__face_embedding = FaceEmbedding(model_path=self.fr_model_path,
                                             model_epoch=int(0),
                                             device=self.device,
                                             emb_layer_name=self.emb_layer_name)

        print("Loading race Model...")
        sym, arg_params, aux_params = mx.model.load_checkpoint("./models/race/model_mem_test_jk", 9)
        all_layers = sym.get_internals()
        race_sym = all_layers['softmax_output']
        
        self.race_mod = mx.mod.Module(symbol=race_sym, context=self.device, label_names=None)
        self.race_mod.bind(for_training=False, data_shapes=[("data", (1, 3, 112, 112))])
        self.race_mod.set_params(arg_params, aux_params, allow_missing=True)
        
        #Thermal race
        #	Asian (0), casian (1), Indian (2)
        
        #Original Race
        # 0-af, 1-as, 2-in, 3-cs
        
        #mapping from original to thermal race
        self.race_map = [2, 0, 2, 1]

        print("Loading age/gender Model...")
        sym, arg_params, aux_params = mx.model.load_checkpoint("./models/age_gender/model", 0)
        all_layers = sym.get_internals()
        age_sym = all_layers['fc1_output']
        
        self.age_mod = mx.mod.Module(symbol=age_sym, context=self.device, label_names=None)
        self.age_mod.bind(for_training=False, data_shapes=[("data", (1, 3, 112, 112))])
        self.age_mod.set_params(arg_params, aux_params)

        print("")

    @staticmethod
    def read_image(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Error: No image, or file doesn't exist.")
            return None
        return img

    @staticmethod
    def __preview_boxes(preview_img, boxes, landmarks, title="Preview"):
        """
        Generate a CV2 preview, given the bounding boxes and landmarks.
        Args:
            preview_img: image matrix (h, w, c)
            boxes: numpy array of shape (n x 5). [x1, y1, x2, y2, score]
            landmarks: numpy array of shape (n x 5 x 2). [[x1, y1] [x2 y2] ... [x5 y5]]
            title: window title
        """
        img = preview_img.copy()
        bbox = boxes.astype(int)
        lmks = landmarks.astype(int)
        for i in range(bbox.shape[0]):
            # For each face, draw the bbox:
            cv2.rectangle(img, (bbox[i, 0], bbox[i, 1]), (bbox[i, 2], bbox[i, 3]), color=(255, 255, 0), thickness=1)
            for j in range(5):
                color = (0, 0, 255)
                if j == 0 or j == 3:
                    color = (0, 255, 0)
                center = (lmks[i, j, 0], lmks[i, j, 1])
                cv2.circle(img, center=center, radius=1, color=color, thickness=2)

        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def __preview_stacked(matrix, axis=1, title="Preview"):
        """
        Stack cropped faces vertically or horizontally.
        Args:
            matrix: Image chips: (n, 112, 112, 3)
            title: window title
            axis: 0 for vertical, 1 for horizontal
        """
        print(matrix.shape)
        output = matrix[0]
        for i in range(1, matrix.shape[0]):
            output = np.concatenate((output, matrix[i]), axis=axis)
        cv2.imshow(title, output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def __get_scale(img, target_size):
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

    @staticmethod
    def __align_and_crop_single(input_img, landmark=None, output_dim=(112, 112), src=None, plot_points=False):
        img = input_img.copy()
        dst = landmark.copy()

        # Do this if landmark is [x1-5, y1-5]
        # dst = np.array(landmark, dtype=np.float32).reshape(2, 5).transpose()

        # dst should be [[x1, y1] [x2 y2] ... [x5 y5]]
        # print("dst:\n", dst)
        # print(dst.shape)

        if plot_points:
            for point in dst:
                cv2.circle(img, (int(point[0]), int(point[1])), radius=1, color=(0, 255, 255), thickness=1)

        # Arcface mean face
        if not isinstance(src, np.ndarray):
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            src[:, 0] += 8.0

        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]

        warped = cv2.warpAffine(img, M, output_dim, borderValue=0.0)
        return warped

    def __align_and_crop(self, img, landmarks, plot_points=False):
        chips = []
        for lmk in landmarks:
            chip = self.__align_and_crop_single(img, lmk, plot_points=plot_points)
            chips.append(chip)
        return np.array(chips)

    def __extract_features_single(self, img):
        # Feed forward through FaceEmbedding:
        feat = self.__face_embedding.extract_feature(img)
        return feat

        #feat_enc, shape, type = self.__crypto.encypt_vec(feat)
        #return feat_enc

    def __extract_features(self, imgs):
        # Batch feature extraction:
        features = self.__face_embedding.extract_feature_batch(imgs)
        #features_enc, _, _ = self.__crypto.encypt_vec(features)
        #return np.array(features_enc)
        return np.array(features)

    def __extract_race_batch(self, imgs):              
        imgs = np.swapaxes(imgs, 1, 3)
        imgs = np.swapaxes(imgs, 2, 3)
        
        races = []
        for chip in imgs:
            # self.exp_arg_params['softmax_label'] = mx.nd.zeros([len(imgs)])
            chip = chip[np.newaxis, :]
            self.race_mod.forward(self.Batch([mx.nd.array(chip)]))
            race = self.race_mod.get_outputs()[0].asnumpy()
            index = np.argsort(race[0])
            idx = index[-1]
            #print("RACE idx ", idx, index, self.race_map[idx], race)
            races.append(self.race_map[idx])
        return np.array(races)

    def __extract_age_gender_batch(self, imgs):              
        imgs = np.swapaxes(imgs, 1, 3)
        imgs = np.swapaxes(imgs, 2, 3)
        
        age_grp = []
        gender = []
        for chip in imgs:
            # self.exp_arg_params['softmax_label'] = mx.nd.zeros([len(imgs)])
            chip = chip[np.newaxis, :]
            self.age_mod.forward(self.Batch([mx.nd.array(chip)]))
            age_gender = self.age_mod.get_outputs()[0].asnumpy()
            
            #0-female, 1-male
            g = age_gender[0,0:2].flatten()
            index = np.argsort(g)
            idx = index[-1]
            gender.append(idx)
            
            a = age_gender[0,2:202].reshape( (100,2) )
            a = np.argmax(a, axis=1)
            age = int(sum(a))   
            
            #map absolute age to groups
            #Original age
            # 1-100
            #Thermal age
            # Age groups: 0-14 (0), 15-25 (1), 26-35 (2), 36-45 (3), 46+ (4)   
       
            # if age<15:
            #     group = 0
            # elif age<26:
            #     group = 1
            # elif age<36:
            #     group = 2
            # elif age<46:
            #     group = 3
            # else:
            #     group = 4

            # age_grp.append(group)
            age_grp.append(age)
            
            print("Gender idx ", idx, index, g, age)
            
        return np.array(age_grp), np.array(gender)

    def Features(self, img, max_faces=None, preview=False):
        """
        img:
            cv2 BGR image
        max_faces:
            Optional parameter to set maximum number of faces to process,
            follows config settings otherwise.

        Returns:
            N x DIM feature vectors
        """
        if max_faces:
            detector_output = self.__detect_faces(img, max_faces=max_faces)
        else:
            # Face Detection
            detector_output = self.__detect_faces(img, max_faces=self.max_detected_faces)

        if detector_output is None:
            return None
        boxes, landmarks = detector_output  # Unpack return value

        # Align and crop the original image
        # landmarks = landmarks.reshape()
        chips = self.__align_and_crop(img, landmarks)

        # Extract features from the crops
        features = self.__extract_features(chips)

        if preview:
            # Preview boxes and landmarks drawn on the image:
            self.__preview_boxes(img, boxes, landmarks, title="Faces Detected")

            # Preview the aligned crop output:
            self.__preview_stacked(chips, axis=1, title="Aligned and Cropped")

        return features
    
    def color_conversion (self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def PersonalFeatures(self, img, max_faces=None, preview=False):
        """
        img:
            cv2 BGR image
        max_faces:
            Optional parameter to set maximum number of faces to process,
            follows config settings otherwise.

        Returns:
            N x DIM feature vectors
        """
        if max_faces:
            detector_output = self.__detect_faces(img, max_faces=max_faces)
        else:
            # Face Detection
            detector_output = self.__detect_faces(img, max_faces=self.max_detected_faces)

        if detector_output is None:
            return None
        boxes, landmarks = detector_output  # Unpack return value

        # Align and crop the original image
        # landmarks = landmarks.reshape()
        chips = self.__align_and_crop(img, landmarks)

        # Extract features from the crops
        features = self.__extract_features(chips)
        
        races = self.__extract_race_batch(chips)
        age_grp, gender = self.__extract_age_gender_batch(chips)
        
        if preview:
            # Preview boxes and landmarks drawn on the image:
            self.__preview_boxes(img, boxes, landmarks, title="Faces Detected")

            # Preview the aligned crop output:
            self.__preview_stacked(chips, axis=1, title="Aligned and Cropped")

        return features, races[0], age_grp[0], gender[0]

    def BoxAndLandmarks(self, img, max_faces=None):
        """
        img:
            cv2 BGR image
        max_faces:
            Optional parameter to set maximum number of faces to process,
            follows config settings otherwise.

        Returns:
            N x 5 box (4-dimensional box location and box score, for each detected face)
            N x DIM feature vectors
        """
        start_t = time.time()
        if max_faces:
            detector_output = self.__detect_faces(img, max_faces=max_faces)
        else:
            # Face Detection
            detector_output = self.__detect_faces(img, max_faces=self.max_detected_faces)

        detect_t = time.time()

        if detector_output is None:
            return None
        #boxes, landmarks = detector_output  # Unpack return value
       
        print("detection time: {:.3f} ms".format((detect_t - start_t) * 1000))

        # boxes = boxes[:, :-1]  # Return only [x1, x2, y1, y2] without box score
        return detector_output

    def BoxAndFeatures(self, img, max_faces=None):
        """
        img:
            cv2 BGR image
        max_faces:
            Optional parameter to set maximum number of faces to process,
            follows config settings otherwise.

        Returns:
            N x 5 box (4-dimensional box location and box score, for each detected face)
            N x DIM feature vectors
        """
        start_t = time.time()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max_faces:
            detector_output = self.__detect_faces(img, max_faces=max_faces)
        else:
            # Face Detection
            detector_output = self.__detect_faces(img, max_faces=self.max_detected_faces)

        detect_t = time.time()

        if detector_output is None:
            return None
        boxes, landmarks = detector_output  # Unpack return value

        # Align and crop the original image
        chips = self.__align_and_crop(img, landmarks)
        align_t = time.time()
 
        #print(boxes)
        # Extract features from the crops
        features = self.__extract_features(chips)
        #print(features)
        feat_t = time.time()

        print("detection time: {:.3f} ms".format((detect_t - start_t) * 1000))
        print("align time    : {:.3f} ms".format((align_t - detect_t) * 1000))
        print("feat ext time : {:.3f} ms\n".format((feat_t - align_t) * 1000))

        # boxes = boxes[:, :-1]  # Return only [x1, x2, y1, y2] without box score
        return boxes, features
