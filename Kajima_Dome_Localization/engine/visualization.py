import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import transform

### start yolox visualization ###
def draw_yolox(img, output, conf=0.5):

    boxes = output[:, 0:4]
    scores = output[:, 5]

    for i in range(len(boxes)):
        box = boxes[i]
 
        cx = int(box[0])
        cy = int(box[1])
        hw = int(box[2]/2)
        hh = int(box[3]/2)

        x0 = cx - hw
        y0 = cy - hh
        x1 = cx + hw
        y1 = cy + hh

        color=(255,0,0)
        line_width = img.shape[0] // 300

        cv2.rectangle(img, (x0, y0), (x1, y1), color, line_width)
    
    return img

def draw_yolox_old(img, output, conf=0.5):

    boxes = output[:, 0:4]
    cls_ids = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        if cls_id == 0:
            score = scores[i]
            if score < conf:
                continue

            cx = int(box[0])
            cy = int(box[1])
            hw = int(box[2]/2)
            hh = int(box[3]/2)

            x0 = cx - hw
            y0 = cy - hh
            x1 = cx + hw
            y1 = cy + hh
    
            # x0 = int(box[0])
            # y0 = int(box[1])
            # x1 = int(box[2])
            # y1 = int(box[3])

            # color = (np.array([0.000, 0.447, 0.741]) * 255).astype(np.uint8).tolist()
            # text = '{}:{:.1f}%'.format('Person', score * 100)
            # txt_color = (0, 0, 0) if np.mean(np.array([0.000, 0.447, 0.741])) > 0.5 else (255, 255, 255)
            # font = cv2.FONT_HERSHEY_SIMPLEX
    
            # txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            # txt_bk_color = (np.array([0.000, 0.447, 0.741]) * 255 * 0.7).astype(np.uint8).tolist()

            color=(255,0,0)
            line_width = img.shape[0] // 300

            cv2.rectangle(img, (x0, y0), (x1, y1), color, line_width)
    
            # cv2.rectangle(
            #     img,
            #     (x0, y0 + 1),
            #     (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            #     txt_bk_color,
            #     -1
            # )
            # # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
            # cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 1.4, txt_color, thickness=3)

    return img

### end yolox visualization ###


def draw_xywha(im, x, y, w, h, angle, color=(255,0,0), linewidth=5):
    '''
    im: image numpy array, shape(h,w,3), RGB
    angle: degree
    '''
    # print("Vis Angle", angle)
    c, s = np.cos(angle/180*np.pi), np.sin(angle/180*np.pi)
    R = np.asarray([[c, s], [-s, c]])
    pts = np.asarray([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    rot_pts = []
    for pt in pts:
        rot_pts.append(([x, y] + pt @ R).astype(int))
    contours = np.array([rot_pts[0], rot_pts[1], rot_pts[2], rot_pts[3]])

    cv2.polylines(im, [contours], isClosed=True, color=color,
                thickness=linewidth, lineType=cv2.LINE_4)

    # cv2.imwrite("./output.jpg", im)
    # width = int(w)
    # height = int(h)
    # src_pts = contours.astype("float32")
    # dst_pts = np.array([[0, 0],[width-1, 0], [width-1, height-1],
    #                     [0, height-1]], dtype="float32")
    # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # warped = cv2.warpPerspective(im, M, (width, height))

    # M = cv2.getRotationMatrix2D( (x, y), float(angle), 1)
    # cv2.imwrite("rot_crop.jpg", warped)


def draw_dt_on_np(im, detections, print_dt=False, color=(255,0,0),
                  text_size=1):
    '''
    im: image numpy array, shape(h,w,3), RGB
    detections: rows of [x,y,w,h,a,conf], angle in degree
    '''
    # line_width = kwargs.get('line_width', im.shape[0] // 300)
    line_width = im.shape[0] // 300
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_bold = max(int(2*text_size), 1)
    for bb in detections:
        if len(bb) == 6:
            x,y,w,h,a,conf = bb
        else:
            x,y,w,h,a = bb[:5]
            conf = -1
        # x1, y1 = x - w/2, y - h/2
        if print_dt:
            print(f'[{x} {y} {w} {h} {a}], confidence: {conf}')
        draw_xywha(im, x, y, w, h, a, color=color, linewidth=line_width)
    #     if kwargs.get('show_conf', True):
    #         cv2.putText(im, f'{conf:.2f}', (int(x1),int(y1)), font, 1*text_size,
    #                     (255,255,255), font_bold, cv2.LINE_AA)
    #     if kwargs.get('show_angle', False):
    #         cv2.putText(im, f'{int(a)}', (x,y), font, 1*text_size,
    #                     (255,255,255), font_bold, cv2.LINE_AA)
    # if kwargs.get('show_count', True):
    #     caption_w = int(im.shape[0] / 4.8)
    #     caption_h = im.shape[0] // 25
    #     start = (im.shape[1] - caption_w, im.shape[0] // 20)
    #     end = (im.shape[1], start[1] + caption_h)
    #     # cv2.rectangle(im, start, end, color=(0,0,0), thickness=-1)
    #     cv2.putText(im, f'Count: {len(detections)}',
    #                 (im.shape[1] - caption_w + im.shape[0]//100, end[1]-im.shape[1]//200),
    #                 font, 1.2*text_size,
    #                 (255,255,255), font_bold*2, cv2.LINE_AA)


def draw_anns_on_np(im, annotations, draw_angle=False, color=(0,0,255), line_width=None):
    '''
    im: image numpy array, shape(h,w,3), RGB
    annotations: list of dict, json format
    '''
    line_width = line_width or im.shape[0] // 500
    for ann in annotations:
        x, y, w, h, a = ann['bbox']
        draw_xywha(im, x, y, w, h, a, color=color, linewidth=line_width)


def flow_to_rgb(flow, plt_show=False):
    '''
    Visualizing optical flow using a RGB image

    Args:
        flow: 2xHxW tensor, flow[0,...] is horizontal motion
    '''
    assert torch.is_tensor(flow) and flow.dim() == 3 and flow.shape[0] == 2

    flow = flow.cpu().numpy()
    mag, ang = cv2.cartToPolar(flow[0, ...], flow[1, ...], angleInDegrees=True)
    hsv = np.zeros((flow.shape[1],flow.shape[2],3), dtype=np.uint8)
    hsv[..., 0] = ang / 2
    hsv[..., 1] = mag
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if plt_show:
        plt.imshow(rgb)
        plt.show()
    return rgb


def tensor_to_npimg(tensor_img):
    tensor_img = tensor_img.squeeze()
    assert tensor_img.shape[0] == 3 and tensor_img.dim() == 3
    return tensor_img.permute(1,2,0).cpu().numpy()


def imshow_tensor(tensor_batch):
    batch = tensor_batch.clone().detach().cpu()
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)
    for tensor_img in batch:
        np_img = tensor_to_npimg(tensor_img)
        plt.imshow(np_img)
    plt.show()


def draw_lmks(img, landmarks, landmark_confs, color, LMK_VISIBILITY_THRESHOLD):
    '''
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
    17: "middle_of_shoulder"
    18: "middle_of_hip"
    '''

    line_pairs = [
        [0,1],[0,2],[1,3],[2,4],[0,17],[17,5],[17,6],[5,7],[6,8],[7,9],[8,10], 
        [17,18],[18,11],[18,12],[11,13],[13,15],[12,14],[14,16]
    ]
    
    for k in range(len(landmarks)):
        lmks = landmarks[k]
        lmk_confs = landmark_confs[k]

        left_shoulder, left_shoulder_conf = lmks[5], lmk_confs[5]
        right_shoulder, right_shoulder_conf = lmks[6], lmk_confs[6]
        middle_of_shoulder = [(left_shoulder[0]+right_shoulder[0])/2, (left_shoulder[1]+right_shoulder[1])/2]
        middle_of_shoulder_conf = 0 if left_shoulder_conf < LMK_VISIBILITY_THRESHOLD or right_shoulder_conf < LMK_VISIBILITY_THRESHOLD else 1
        # print(left_shoulder, right_shoulder, middle_of_shoulder)

        left_hip, left_hip_conf = lmks[11], lmk_confs[11]
        right_hip, right_hip_conf = lmks[12], lmk_confs[12]
        middle_of_hip = [(left_hip[0]+right_hip[0])/2, (left_hip[1]+right_hip[1])/2]
        middle_of_hip_conf = 0 if left_hip_conf < LMK_VISIBILITY_THRESHOLD or right_hip_conf < LMK_VISIBILITY_THRESHOLD else 1
        
        # print(np.asarray(middle_of_shoulder))
        lmks = np.insert(lmks, 34, np.asarray(middle_of_shoulder)) # 17
        lmks = np.insert(lmks, 36, np.asarray(middle_of_hip)) # 17
        lmks = np.reshape(lmks, (-1,2))

        # print(lmk_confs)
        lmk_confs = np.insert(lmk_confs, 17, np.asarray(middle_of_shoulder_conf)) #17
        lmk_confs = np.insert(lmk_confs, 18, np.asarray(middle_of_hip_conf)) #18
        lmk_confs = np.reshape(lmk_confs, (-1,1))
        # print(lmk_confs)

        for (x,y),v in zip(lmks, lmk_confs):
            r = 2 # radius
            cv2.circle(img, center=(int(x),int(y)), radius=r, color=(0,0,255), thickness=2)
        
        for lmk1, lmk2 in line_pairs:
            v1,v2 = lmk_confs[lmk1][0], lmk_confs[lmk2][0]
            if v1 < LMK_VISIBILITY_THRESHOLD or v2 < LMK_VISIBILITY_THRESHOLD:
                continue

            p1 = lmks[lmk1]
            p2 = lmks[lmk2]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, thickness=2)


def plt_show(im):
    plt.imshow(im)
    plt.show()