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
for frame in files[:120]:
    img = cv2.imread(frame)
    cv2.imshow(mat, img)
    cv2.waitKey(100)

# f = open('ground_truth/new_record.txt', 'r')

# for file_name in path:
#     file_name = root + file_name + '/images/005.png'
#     files.append(file_name)

# cv2.namedWindow(mat, cv2.WND_PROP_FULLSCREEN)

# cv2.waitKey(3000)
# for frame in files[:120]:
#     img = cv2.imread(frame)

#     line = f.readline()
#     bbox = line.split(',')
#     bbox = list(map(int, bbox))

#     cv2.rectangle(img,
#                   (bbox[0], bbox[1]),
#                   (bbox[0]+bbox[2], bbox[1]+bbox[3]),
#                   (0, 255, 0), 3)
#     cv2.imshow(mat, img)
#     cv2.waitKey(40)
