import os
import cv2


root = 'data/result/'
path = os.listdir(root)
mat = 'focal_images'
files = []

for file_name in path:
    file_name = root + file_name
    files.append(file_name)

cv2.namedWindow(mat, cv2.WND_PROP_FULLSCREEN)

cv2.waitKey(5000)
for frame in files:
    img = cv2.imread(frame)
    cv2.imshow(mat, img)
    cv2.waitKey(500)
