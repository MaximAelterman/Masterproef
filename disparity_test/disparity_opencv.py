import numpy as np
from sklearn.preprocessing import normalize
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('l.png',0)
imgR = cv2.imread('r.png',0)

left_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=80, blockSize=11)	#v1: numDisparities=192, blocksize=21
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 80000
sigma = 1.2
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

print('computing disparity...')
displ = left_matcher.compute(imgL, imgR).astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL).astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)

#filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
#filteredImg = np.uint8(filteredImg)
cv2.imshow('Disparity Map', filteredImg)
np.save("test.npy", filteredImg)
cv2.imwrite('test_opencv.png', filteredImg)
cv2.waitKey()
cv2.destroyAllWindows()

#disparity = stereo.compute(imgL,imgR).astype(float) / 16
#plt.imshow(disparity,'gray')
#plt.show()
