from plenoptic_dataloader import PlenopticDataLoader
import cv2
from glob import glob
import os


def get_frames(video_name, start_num, last_num):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
            video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name == "2D":
        dataLoader_focal = PlenopticDataLoader(
            root='E:/NonVideo4', img2d_ref='images/005.png', focal_range=(start_num, last_num))
        img2d_files = dataLoader_focal.dataLoader_2d()
        # for i in range(len(img2d_files)):
        for img2d_file in img2d_files:
            frame=cv2.imread(img2d_file)
            yield frame
    elif video_name == "3D":
        dataLoader_focal=PlenopticDataLoader(
            root='E:/NonVideo4', img2d_ref='images/005.png', focal_range=(start_num, last_num))
        img2d_files, focal_files=dataLoader_focal.dataLoader_focal()
        # for i in range(150):
        for i in range(len(img2d_files)):
            frame=cv2.imread(img2d_files[i])
            yield frame, focal_files[i]
    else:
        images=glob(os.path.join(video_name, '*.jp*'))
        images=sorted(images,
                        key=lambda x: x.split('/')[-1].split('.')[0])
        # key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame=cv2.imread(img)
            yield frame
