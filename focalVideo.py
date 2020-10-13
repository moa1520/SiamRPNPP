import os
import cv2


root = 'E:/NonVideo4/'
path = os.listdir(root)
mat = 'focal_images'
files = []

f = open('ground_truth/new_record.txt', 'r')

for file_name in path:
    file_name = root + file_name + '/images/005.png'
    files.append(file_name)

cv2.namedWindow(mat, cv2.WND_PROP_FULLSCREEN)

for frame in files:
    img = cv2.imread(frame)

    line = f.readline()
    bbox = line.split(',')
    bbox = list(map(int, bbox))

    cv2.rectangle(img,
                  (bbox[0], bbox[1]),
                  (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                  (0, 255, 0), 3)
    cv2.imshow(mat, img)
    cv2.waitKey(40)
