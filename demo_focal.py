import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from model import ModelBuilder
from tracker import build_tracker
from data_loader import dataLoader
from data_loader_focal import dataLoader_focal

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
            frame = cv2.imread(files[i])
#            focal_img = cv2.imread(focal[i][69])
            yield frame, focal
        # for img in files:
        #     frame = cv2.imread(img)
        #     yield frame
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
    checkpoint = torch.load("pretrained_model/SiamRPNres50.pth",
                            map_location=lambda storage, loc: storage.cpu())

    model.load_state_dict(checkpoint)
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    root = "test"
    video_name = root.split('/')[-1].split('.')[0]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    for frame, focal in get_frames(root):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = []
            max_score = 0
            max_index = 0
            for f in focal:
                outputs.append(tracker.track(cv2.imread(f[68])))
            for i in range(len(outputs)):
                if max_score < outputs[i]['best_score']:
                    max_score = outputs[i]['best_score']
                    max_index = i

            # outputs = tracker.track(focal)
#            bbox = list(map(int, outputs['bbox']))
            bbox = list(map(int, outputs[i]['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == "__main__":
    main()
