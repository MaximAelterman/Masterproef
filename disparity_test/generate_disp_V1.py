import argparse
import os

import numpy as np
import cv2
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Disparity')
    parser.add_argument('--data_path', type=str, default='~/Kitti/object/training/')
    parser.add_argument('--split_file', type=str, default='~/Kitti/object/trainval.txt')
    args = parser.parse_args()

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

    # Matcher init
    left_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=192, blockSize=21)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # Filter parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
     
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)  

    for fn in file_names:
        image2_file = '{}/{}.png'.format(image2_dir, fn)
        image3_file = '{}/{}.png'.format(image3_dir, fn)
        imgL = cv2.imread(image2_file, 0)
        imgR = cv2.imread(image3_file, 0)

        displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL).astype(np.float32)/16
        displ = np.int16(displ)#np.int8(displ)
        dispr = np.int16(dispr)#np.int8(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr)

        #filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        #filteredImg = np.uint8(filteredImg)

        np.save(disparity_dir + '/' + fn, filteredImg)
        cv2.imwrite(disparity_dir + '/' + fn + ".png", filteredImg)
        print('Finish Disparity {}'.format(fn))
