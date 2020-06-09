from ctypes import *
import math
import random
import cv2

import argparse
import os
import numpy as np

import kitti_util
import time

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/maxim/Desktop/masterproef/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
    parser.add_argument('--split_file', type=str, default='~/Kitti/object/trainval.txt')
    args = parser.parse_args()

    assert os.path.isdir(args.data_path)
    image_dir = args.data_path + '/image_2/'
    disparity_dir = args.data_path + '/predict_disparity/'
    detections_dir = args.data_path + '/detections/'
    results_dir = args.data_path + '/results/'
    calib_dir = args.data_path + '/calib/'

    assert os.path.isdir(image_dir)

    if not os.path.isdir(detections_dir):
        os.makedirs(detections_dir)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    assert os.path.isfile(args.split_file)
    with open(args.split_file, 'r') as f:
        file_names = [x.strip() for x in f.readlines()]

    font = cv2.FONT_HERSHEY_SIMPLEX

    #yolo netwerk laden
    net = load_net("cfg/yolov3_gpu.cfg", "yolov3.weights", 0)   #model voor low-end gpus
    #net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)      #compleet model
    meta = load_meta("cfg/coco.data")
    
    baseline = 0.54             #afstand tussen de kleurencamera's van KITTI

    for fn in file_names:
        img_path = '{}/{}.png'.format(image_dir, fn)
        disparity_path = '{}/{}.npy'.format(disparity_dir, fn)
        detections_path = '{}/{}.png'.format(detections_dir, fn)
        results_path = '{}/{}.txt'.format(results_dir, fn)
        calib_file = '{}/{}.txt'.format(calib_dir, fn)
        calib = kitti_util.Calibration(calib_file)

        r = detect(net, meta, img_path)
        img = cv2.imread(img_path, 1)
        img_height = img.shape[0]
        img_width = img.shape[1]
        result_file = open(results_path, "w")
        disparity = np.load(disparity_path)
        print("inference on " + fn)

        #yolo detecties duiden het centrum van de detectie aan met width en height: 'label', score, (x-centrum, y-centrum, width, height)
        for detection in r:
            label = detection[0]
            score = detection[1]
            centerx = int(detection[2][0])
            centery = int(detection[2][1])
            width = int(detection[2][2])
            height = int(detection[2][3])
            length = 4.2
            
            #links boven hoek bounding box: x1, y1
            x1 = centerx - width/2
            y1 = centery - height/2
            #rechts onder hoek bounding box: x2, y2
            x2 = centerx + width/2
            y2 = centery + height/2

            #genereren van detection file
            if label == "car" and score > 0.6:     
                prediction_data = []
                prediction_data.append("Car ")
                prediction_data.append("-1 ")                                           #truncated: object leaves the picture
                prediction_data.append("-1 ")                                           #occluded: unknown (3)
                prediction_data.append("-10 ")                                          #alpha: viewing angle (-pi ... pi)

                #start_time = time.time()
                mask = disparity[centery][centerx] > 0
                depth = calib.f_u * baseline / (disparity[centery][centerx] + 1. - mask)
                x = ((centerx - calib.c_u) * depth) / calib.f_u + calib.b_x
                y = ((y2 - calib.c_v) * depth) / calib.f_v + calib.b_y
                #print("--- %s seconds ---" % (time.time() - start_time))

                if width < height*0.9 and not((centerx < img_width/6 and centery > img_height/2) or (centerx > img_width*5/6 and centery > img_height/2)):                                                      #auto wordt waarschijnlijk deels bedekt (verhouding klopt niet) -> detectie aanpassen met een w/h ratio van 1.5
                    links_gemiddeld = 0.0                    
                    new_width = height * 1.5                                            #breedte aanpassen naar een meer realistische verhouding
                    center_disp = disparity[centery][centerx]
                    for offset in range(int(new_width/2)):                                   #kijken waar de occlusie is
                        #print(offset)
                        xpos = centerx-offset
                        if xpos < 0:
                            xpos = 0
                        links_gemiddeld += disparity[centery][centerx-offset]
                    links_gemiddeld = links_gemiddeld/(new_width/2)
                    if links_gemiddeld > center_disp:                                   #linker deel heeft occlusie -> dat deel aanvullen
                        x1 = int(x2-new_width)
                    else:                                                               #rechter deel heeft occlusie
                        x2 = int(x1+new_width)
                                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)              #rood
                    cv2.circle(img, (centerx, centery), 1, (0, 0, 255), -1)
                    cv2.putText(img, label + ", %.2f"%score, (x1, y1-3), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, "d:%.2f"%depth, (centerx, centery-3), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)   
                
                elif height < 25:
                    if depth < 47.0:
                        height = 26
                        y1 = centery - height/2
                        y2 = centery + height/2
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 155, 0), 2)        #licht blauw
                        cv2.putText(img, label + ", %.2f"%score, (x1, y1-3), font, 0.5, (255, 155, 0), 1, cv2.LINE_AA)
                        cv2.putText(img, "d:%.2f"%depth, (centerx, centery-3), font, 0.5, (255, 155, 0), 1, cv2.LINE_AA)  
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)          #blauw
                        cv2.putText(img, label + ", %.2f"%score, (x1, y1-3), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(img, "d:%.2f"%depth, (centerx, centery-3), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)                       

                else:
                                    #print(label + " found, score: " + str(score))
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)              #groen
                    cv2.putText(img, label + ", %.2f"%score, (x1, y1-3), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, "d:%.2f"%depth, (centerx, centery-3), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                 
                prediction_data.append(str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" ") #bounding box
                prediction_data.append(str(1.5)+" "+str(1.8)+" "+str(length)+" ")       #dimensions: height, width, length
                prediction_data.append("%.3f"%x+" "+"%.3f"%y+" "+"%.3f"%(depth+length/2)+" ")      #location: x (breedte), y (diepte), z (hoogte)
                rot_y = math.pi/2
                prediction_data.append("%.3f"%rot_y+" ")                                            #rotation_y
                prediction_data.append("%.3f"%score)                                    #score
                for d in prediction_data:
                    result_file.write(str(d))
                result_file.write("\n") 
                
 

        #cv2.imwrite(detections_path, img)



    print r
    

