from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

import torch
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# import human_pose_estimation.pose_estimation._init_paths
from config import config
from config import update_config
# from human_pose_estimation.pose_estimation.config import update_dir
import cv2
# import models
from pose_resnet import get_pose_net
import numpy as np
import math

class HumanPoseDetection():
    def __init__(self, model_path, cfg_path, device):
        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = config.CUDNN.ENABLED

        update_config(cfg_path)

        if device<0:
            self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cuda:{}".format(device))

        self.model = eval('get_pose_net')(
            config, is_train=False
        )

        self.model.load_state_dict(torch.load(model_path))
        # self.gpuid = device #JK
        self.model = self.model.to(self.device) #JK
        self.model.half()

        # gpus = [int(i) for i in config.GPUS.split(',')]
        # gpus = [device]
        # print("SKELETON GPUUUUU ", gpus)
        # self.model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

        # # define loss function (criterion) and optimizer
        # criterion = JointsMSELoss(
        #     use_target_weight=config.LOSS.USE_TARGET_WEIGHT
        # ).cuda()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])

    def _box2cs(self, box, image_width, image_height):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h, image_width, image_height)

    def _xywh2cs(self, x, y, w, h, image_width, image_height):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        
        # aspect_ratio = image_width * 1.0 / image_height
        aspect_ratio = image_height / image_width #JK
        # pixel_std = 200
        # pixel_std = 100

        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        # scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
        scale = np.array([w, h], dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


    def detect(self, img_numpy, bbox):
        x,y,w,h = bbox
        x1,y1 = x-w/2,y-h/2
        # x1,y1,w,h = x1*img_numpy.shape[0],y1*img_numpy.shape[1],w*img_numpy.shape[0],h*img_numpy.shape[1]
        bbox = [x1,y1,w,h]
        height, width = img_numpy.shape[0], img_numpy.shape[1] #JK
        c, s = self._box2cs(bbox, width, height) #JK
        # c, s = self._box2cs(bbox, img_numpy.shape[0], img_numpy.shape[1])
        r = 0

        trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
        input = cv2.warpAffine(
            img_numpy,
            trans,
            (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)


        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225]),
        #     ])

        input = self.transform(input).unsqueeze(0)
        input_ = input.cuda(self.device)
        # print(input.shape)

        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            # compute output heatmap
            output = self.model(input_)
            # compute coordinate
            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))
            # plot
            return list(preds[0]), list(maxvals[0])

    def detect_batch(self, img_numpy, detections, rot=None):
        height, width = img_numpy.shape[0], img_numpy.shape[1] #JK
        inputs = []
        centers = []
        scales = []
        # for bb in bboxs:
        for i in range(len(detections)):
            bb = detections[i]
            x,y,w,h = bb
            x1,y1 = x-w/2,y-h/2
            bbox = [x1,y1,w,h]
            c, s = self._box2cs(bbox, width, height) #JK
            r = rot[i] if rot is not None else 0 #JK
            # print("ROT", r)
            # r = 0
            trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
            input = cv2.warpAffine(img_numpy, trans,
                (int(config.MODEL.IMAGE_SIZE[0]), 
                int(config.MODEL.IMAGE_SIZE[1])),
                flags=cv2.INTER_LINEAR)

            input = self.transform(input).unsqueeze(0)
            # print(input.type, input.shape)

            inputs.extend(input)
            centers.append(c)
            scales.append(s)
        
        # inputs = self.transform(np.asarray(inputs)).unsqueeze(0)
        inputs = torch.stack(tuple(inputs))
        # print(inputs.shape)
        inputs_ = inputs.cuda(self.device)

        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            # compute output heatmap
            output = self.model(inputs_.half())
            # compute coordinate
            preds, maxvals = get_final_preds(config, output.clone().cpu().numpy(), 
                np.asarray(centers), np.asarray(scales))
            # plot
            return preds, maxvals

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    # scale_tmp = scale * 200.0
    # scale_tmp = scale * 100.0 #JK
    scale_tmp = scale #JK
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array([hm[py][px+1] - hm[py][px-1],
                                     hm[py+1][px]-hm[py-1][px]])
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords
