import numpy as np
import cv2

data = np.load("000010.npy")
np.savetxt("test.csv", data, delimiter=",")
print(data.shape)
maxdisp = 0;
blank_image = np.zeros((data.shape[0], data.shape[1],3), np.uint8)
for x in range(len(data)):
  for y in range(len(data[x])):
    blank_image[x][y] = data[x][y]
    if maxdisp < data[x][y]:
        maxdisp = data[x][y]
print(maxdisp)
cv2.imwrite('test.png', blank_image*1.17)
np.save("test.npy", data*1.17)
