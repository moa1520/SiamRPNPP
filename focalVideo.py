import os
import cv2

root = 'E:/NonVideo4/'
path = os.listdir(root)
mat = 'focal_images'
files = []

for folder in path:
    file_name = '/focal/053.png'
    file = root + folder + file_name
    files.append(file)

cv2.namedWindow(mat, cv2.WND_PROP_FULLSCREEN)

for frame in files:
    cv2.imshow(mat, cv2.imread(frame))
    cv2.waitKey(40)
