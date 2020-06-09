from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
import numpy as np
import time
import math
from utils import preprocess 
from models import *
import cv2

from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser(description='Generate Disparity')
parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
parser.add_argument('--split_file', type=str, default='~/Kitti/object/trainval.txt')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar', help='loading model')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

assert os.path.isdir(args.data_path)
image2_dir = args.data_path + '/image_2/'
image3_dir = args.data_path + '/image_3/'
disparity_dir = args.data_path + '/predict_disparity/'

assert os.path.isdir(image2_dir)
assert os.path.isdir(image3_dir)

if not os.path.isdir(disparity_dir):
    os.makedirs(disparity_dir)

assert os.path.isfile(args.split_file)
with open(args.split_file, 'r') as f:
    file_names = [x.strip() for x in f.readlines()]

torch.manual_seed(1)
if args.cuda:
	torch.cuda.manual_seed(1)

model = stackhourglass(192)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
            imgL = torch.FloatTensor(imgL).cuda()
            imgR = torch.FloatTensor(imgR).cuda()

        imgL, imgR= Variable(imgL), Variable(imgR)

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp

def main():
    processed = preprocess.get_transform(augment=False)


    for fn in file_names:
        image2_file = '{}/{}.png'.format(image2_dir, fn)
        image3_file = '{}/{}.png'.format(image3_dir, fn)
        #image2_file = '/content/gdrive/My Drive/Masterproef/Kitti/object/training/image_2/000002.png'
        #image3_file = '/content/gdrive/My Drive/Masterproef/Kitti/object/training/image_3/000002.png'

        imgL_o = (skimage.io.imread(image2_file))#.astype('float32'))
        imgR_o = (skimage.io.imread(image3_file))#.astype('float32'))

        imgL = processed(imgL_o).numpy()
        imgR = processed(imgR_o).numpy()
        imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
        imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

        # pad to width and hight to 16 times
        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16       
            top_pad = (times+1)*16 -imgL.shape[2]
        else:
            top_pad = 0
        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16                       
            left_pad = (times+1)*16-imgL.shape[3]
        else:
            left_pad = 0     
        imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
        imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

        start_time = time.time()
        pred_disp = test(imgL,imgR)
        print('time = %.2f' %(time.time() - start_time))
        if top_pad !=0 or left_pad != 0:
            img = pred_disp[top_pad:,:-left_pad]
        else:
            img = pred_disp
        img = (img*256).astype('uint16')
        #skimage.io.imsave(disparity_dir + '/' + fn + ".png", img)
        #skimage.io.imsave('/content/gdrive/My Drive/Masterproef/pseudo_lidar/psmnet/disparity.png', img)        

        np.save(disparity_dir + '/' + fn, pred_disp)
        ##cv2.imwrite(disparity_dir + '/' + fn + ".png", img)
        print('Finish Disparity {}'.format(fn))

if __name__ == '__main__':
    main()
