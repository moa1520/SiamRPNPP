import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from model import ModelBuilder
from tracker import build_tracker
from data_loader import dataLoader
from data_loader_focal_test import dataLoader_focal

# parser = argparse.ArgumentParser(description="tracking demo")
# parser.add_argument('--video_name', default='', type=str,
#                     help='videos or image files')
# args = parser.parse_args()


def get_frames(video_name):
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
    elif video_name == "test":
        files, focal = dataLoader_focal()
        for i in range(len(files)):
            print("frame: ", i)
            frame = cv2.imread(files[i])
            focal_frame = cv2.imread(focal[i])
            yield frame, focal_frame
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: x.split('/')[-1].split('.')[0])
        # key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    # create model
    model = ModelBuilder()

    # load model
    checkpoint = torch.load("pretrained_model/model.pth",
                            map_location=lambda storage, loc: storage.cpu())

    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    root = "test"
    video_name = root.split('/')[-1].split('.')[0]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    a = 0
    for frame, focal in get_frames(root):
        a += 1
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(focal)

            save_img = outputs['x_crop'].data.cpu().squeeze(
                0).numpy().transpose((1, 2, 0)).astype(np.int32)
            #s = ret['detection_cropped_resized']
            save_path = os.path.join(
                'data/x_crop', '{:03d}_detection_input.jpg'.format(a))
            cv2.imwrite(save_path, save_img)

            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            save_path = os.path.join(
                'data/output', '{:03d}_detection_input.jpg'.format(a))
            cv2.imwrite(save_path, frame)
            cv2.waitKey(40)


if __name__ == "__main__":
    main()
