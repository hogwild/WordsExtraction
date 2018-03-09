import sys
sys.path.append("./tools/")
sys.path.append("./src/")
sys.path.append("./kalaok/")



import cv2
import numpy as np
import nms
import sys
sys.path.append("./tools/")
sys.path.append("./src/")
sys.path.append("./kalaok/")

from cfg import Config as cfg
from other import draw_boxes, resize_im, CaffeModel
import  caffe
from detectors import TextProposalDetector, TextDetector

#import os.path as osp
#from utils.timer import Timer






def near_wall_test(floor, wall, max_dist, delta_x, delta_y):
    '''判定floor往一个方向走时是否会在max_dist步内遇到wall'''

    if (delta_x * delta_y != 0):
        raise Exception("Unsupported")
    result = np.zeros_like(wall, dtype='int8')
    range_x = range(delta_x, wall.shape[0])
    if (delta_x < 0):
        range_x = range(wall.shape[0] + delta_x - 1, -1, -1)
    range_y = range(delta_y, wall.shape[1])
    if (delta_y < 0):
        range_y = range(wall.shape[1] + delta_y - 1, -1, -1)
    result[wall] = max_dist
    if (delta_x == 0):
        for y in range_y:
            result[:, y] = np.maximum(result[:, y], floor[:, y] * result[:, y - delta_y] - 1)
    else:
        for x in range_x:
            result[x, :] = np.maximum(result[x, :], floor[x, :] * result[x - delta_x, :] - 1)
    return np.logical_and(result > 0, floor)


    
def pick_char(img, mser, ch_height, bboxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    bin_im = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 5)
#    mask = np.zeros(gray.shape)
#    for box in bboxes:
#        mask[int(box[1]*0.9):int(box[3]*1.1), int(box[0]-ch_height):int(box[2]+ch_height)] = 1
#    mask = mask==0
#    gray[mask] = 0
#    bin_im[mask] = 0
    regions, boxes = mser.detectRegions(gray)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    keep = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        if h <= ch_height*1.1 and w <= ch_height*1.1 and w*h <= ch_height*ch_height*1.1:  
            keep.append([x, y, x + w, y + h])
    keep2 = np.array(keep)
    pick = nms.nms(keep2, 0.1)
    pick = list(pick)
#    for i in range(len(pick)-1, -1, -1):
#        if (pick[i][2]-pick[i][0])*(pick[i][3]-pick[i][1]) <= w*h*0.8:
#            pick.pop(i)
    return pick


def sort_bboxes(bboxes, MID):
    sorted_bboxes = {}
    sorted_bboxes['up'] = []
    sorted_bboxes['down'] = []
    for box in bboxes:
        if box[3] <= MID:
            sorted_bboxes['up'].append(box)
        else:
            sorted_bboxes['down'].append(box)
    sorted_bboxes['up'].sort(key=lambda x:x[0])
    sorted_bboxes['down'].sort(key=lambda x:x[0])
    return sorted_bboxes


def char_width(sorted_bboxes):
    up_line = sorted_bboxes['up']
    down_line = sorted_bboxes['down']
    w_up = []
    w_down = []
    for c in up_line:
        w_up.append(c[2]-c[0])
    for c in down_line:
        w_down.append(c[2]-c[0])
    width = np.array(w_up + w_down)
    w_hist, w_bin = np.histogram(width)
    idx = np.argmax(w_hist)
    w = (w_bin[idx]+w_bin[idx+1])//2
    return w

def char_refined(sorted_bboxes, char_height, text_lines):
    refined = [sorted_bboxes.pop(0)]
    while sorted_bboxes:
        if sorted_bboxes[0][0] - refined[-1][0] < char_height*0.6:
            temp = sorted_bboxes.pop(0)
            refined[-1][1] = min(refined[-1][1], temp[1])
            refined[-1][2] = max(refined[-1][2], temp[2])
            refined[-1][3] = max(refined[-1][3], temp[3])
        else:
            refined.append(sorted_bboxes.pop(0))
    return refined                



if __name__ == '__main__':
    global fps, cap, EMIT_THERSHOLD
    cap = cv2.VideoCapture("/home/hogwild/Documents/video_processing/samples/wangfeng.mkv")
    block_height = 53
    block_width = 44
    line1_x = 14
    line1_y = 81
    line2_x = 71
    line2_y = -66
    max_text_length = 13
    START_FRAME = 500
    EMIT_THERSHOLD = 2
    
    
    NET_DEF_FILE = "models/deploy.prototxt"
    MODEL_FILE = "models/ctpn_trained_model.caffemodel"
    caffe.set_mode_gpu()
    caffe.set_device(cfg.TEST_GPU_ID)
    
    # initialize the detectors
    text_proposals_detector = TextProposalDetector(CaffeModel(NET_DEF_FILE, MODEL_FILE))
    text_detector = TextDetector(text_proposals_detector)
  
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 限定字幕可能出现的范围：底部1/3，可选参数
    X = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 3
    Y = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.set(1, START_FRAME)
    ret, frame = cap.read()
#    frame = cv2.resize(frame[-X:,:], None, fx=0.5, fy=0.5)
    x,y,z = frame.shape
    countmap = np.zeros((X, Y), dtype='int8')
    mser = cv2.MSER_create()
    MID_region_y = X//2
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (not ret):
            break
        frame = frame[-X:, :]
#        frame = cv2.resize(frame, None, fx=0.5, fy=0.5) 
        output = np.zeros_like(frame)
        # 输入是bgr，先转rgb
        rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # 判定白底部分（潜在未唱文字前景）
        white_area = np.logical_and(hsv[:, :, 1] < 10, hsv[:, :, 2] > 170)
        # 判定黑色部分（潜在未唱文字边框）
        black_area = hsv[:, :, 2] < 70
        # 简单扩展一下黑色的范围，填补黑-白过渡区域
        black_area = cv2.distanceTransform(1 - black_area.astype('uint8'), maskSize=3, distanceType=cv2.DIST_C) < 2
        # 判定蓝色部分（潜在已唱文字前景，目前颜色指定为蓝色）
        blue_area = np.logical_and(np.logical_and(hsv[:, :, 0] <= 125, hsv[:, :, 0] >= 115),
                                   np.logical_and(hsv[:, :, 1] >= 210, hsv[:, :, 2] >= 20))

        # 三方向判断白底周围是否是黑色部分，还有一个方向(-y方向)不做判断
        # 因为在跑马字滚动到一半的时候，白底文字往左走可能会遇到蓝色而非黑色
        condition1 = near_wall_test(white_area, black_area, 75, 1, 0)
        condition2 = near_wall_test(white_area, black_area, 75, -1, 0)
        condition3 = near_wall_test(white_area, black_area, 75, 0, -1)
        condition_white = np.logical_and(np.logical_and(condition1, condition2), condition3)
        
        white_area = cv2.distanceTransform(1-white_area.astype('uint8'), maskSize=3, distanceType=cv2.DIST_C)<2
        condition4 = near_wall_test(blue_area, white_area, 75, 1, 0)
        condition5 = near_wall_test(blue_area, white_area, 75, -1, 0)
        condition6 = near_wall_test(blue_area, white_area, 75, 0, 1)
        condition7 = near_wall_test(blue_area, white_area, 75, 0, -1)
        condition_blue = np.logical_or(condition4, condition6)
        # 如果该像素在过去频繁被判定condition==True且突然变成blue_area
        # 那么它很有可能是字幕变化部分
        # countmap记录该像素近期判定为condition==True的频繁度
#        switch_cond = np.logical_and(countmap > 15, blue_area)
#        countmap -= 1
#        countmap = np.maximum(0, countmap)
#        countmap[condition] += 2
#        countmap = np.minimum(30, countmap)
#        countmap[blue_area] = 0
        
        condition = np.logical_or(condition_white, condition_blue)
#        fine_tune = cv2.distanceTransform(1-condition.astype('uint8'), maskSize=3, distanceType=cv2.DIST_C)<1
        
        output[condition] = [255, 255, 255]
#        output[blue_area] = [0, 0, 255]
#        output[switch_cond] = [0, 255, 255]
        text_lines = text_detector.detect(output)
        ch_height = 0
        ch_refined = []
        if len(text_lines) > 0:
            for box in text_lines:
#                print('box', box)
                ch_height += box[3]-box[1]
            ch_height /= len(text_lines)
#            print('char height,', ch_height)
            chars = pick_char(output, mser, ch_height, text_lines)
            s_boxes = sort_bboxes(chars, MID_region_y)
            for k in s_boxes.keys():
                if len(s_boxes[k]) > 0:
                    ch_refined += char_refined(s_boxes[k], ch_height, text_lines)

                
            
#            w = char_width(s_boxes)
#            print('char width:', w)
        if len(ch_refined) > 0:
            for (startX, startY, endX, endY) in ch_refined:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            
        
#        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
#        regions, boxes = mser.detectRegions(gray)
#        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#        keep = []
#        for c in hulls:
#            x, y, w, h = cv2.boundingRect(c)
#            keep.append([x, y, x + w, y + h])
#        keep2 = np.array(keep)
#        pick = nms.nms(keep2, 0.1)
#        for (startX, startY, endX, endY) in pick:
#            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
        im_with_text_lines = draw_boxes(frame, text_lines, color=(0, 255, 0), caption="Text Detection", wait=False)

#        cv2.imshow('frame', frame)
        cv2.imshow('output', output)
        
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
