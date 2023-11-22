import cv2
import torch
import torchvision
import time
import os
import numpy as np

import sys
import pathlib
scriptpath = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scriptpath.parent / 'yoloxwrapper'))
from build import get_exp

def postprocess(prediction, num_classes, ratios, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    #convert cx, cy, w h format to x1, y1, x2, y2 for nms calculations
    # org_box = prediction.new(prediction.shape)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]

        detections = detections.cpu()
        bboxes = detections[:, 0:4]

        # preprocessing: resize
        bboxes /= ratios[i]

        # convert back to cx, cy, w, h format
        width = bboxes[:, 2] - bboxes[:, 0]
        height = bboxes[:, 3] - bboxes[:, 1]
        cx = bboxes[:, 0] + width/2
        cy = bboxes[:, 1] + height/2

        bboxes[:, 0] = cx
        bboxes[:, 1] = cy
        bboxes[:, 2] = width
        bboxes[:, 3] = height

        detections[:, 0:4] = bboxes

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    

    if len(prediction)==1:
        output = output[0]

    return output

class YoloxPredictor(object):
    def __init__(
        self,
        model_name,
        model_weight,
        test_size,
        test_conf,
        nmsthre,
        maxppl,
        device=0,
        fp16=False
    ):
        exp = get_exp(None, model_name)
        self.num_classes = exp.num_classes
        self.confthre = test_conf
        self.nmsthre = nmsthre
        self.test_size = test_size #(test_size, test_size)
        self.device = device
        self.fp16 = fp16

        exp.test_conf = self.confthre
        exp.nmsthre = self.nmsthre
        exp.test_size = self.test_size
        
        model = exp.get_model()

        # if device == "gpu":
        #     model.cuda()
        #     if fp16:
        #         model.half()  # to FP16
        # model.eval()
        
        # logger.info("loading checkpoint")
        ckpt = torch.load(model_weight, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        # logger.info("loaded checkpoint done.")
        self.yolox_model = model

        if device<0:
            self.device = torch.device("cpu")
        else: 
            self.device = torch.device("cuda:{}".format(device))
        self.yolox_model = self.yolox_model.to(self.device)

    def preproc(self, img, r):
        swap = swap=(2, 0, 1)
        if len(img.shape) == 3:
            padded_img = np.ones((self.test_size, self.test_size, 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones((self.test_size, self.test_size), dtype=np.uint8) * 114

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img

    def inference(self, img):

        ratio = min(self.test_size / img.shape[0], self.test_size / img.shape[1])

        img = self.preproc(img, ratio)

        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.cuda(self.device)

        # if self.device == "gpu":
        #     img = img.cuda()
        #     if self.fp16:
        #         img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.yolox_model(img)
            outputs = postprocess(
                outputs, self.num_classes, [ratio], self.confthre,
                self.nmsthre, class_agnostic=True
            )

        return outputs

    def inferenceBatch(self, imgs):

        ratios = []
        batch_img = []
        for img in imgs:
            ratio = min(self.test_size / img.shape[0], self.test_size / img.shape[1])
            ratios.append(ratio)

            pimg = self.preproc(img, ratio)
            pimg = torch.from_numpy(pimg).unsqueeze(0)
            pimg = pimg.float()
            batch_img.extend(pimg)

        batch_img = torch.stack(tuple(batch_img))
        batch_img = batch_img.cuda(self.device)

        # img = img.cuda(self.device)
        # if self.device == "gpu":
        #     img = img.cuda()
        #     if self.fp16:
        #         img = img.half()  # to FP16

        with torch.no_grad():
            outputs = self.yolox_model(batch_img)
            outputs = postprocess(
                outputs, self.num_classes, ratios, self.confthre,
                self.nmsthre, class_agnostic=True
            )

        return outputs

 
