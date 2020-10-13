import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from model import ModelBuilder
from tracker import build_tracker
from plenoptic_dataloader import PlenopticDataLoader

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
args = parser.parse_args()

start_num = 0
last_num = 100


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
        data_loader = PlenopticDataLoader(
            root='E:/NonVideo4', img2d_ref='images/005.png', focal_range=(start_num, last_num))
        images = data_loader.dataLoader_2d()
        for img in images[:120]:
            frame = cv2.imread(img)
            yield frame
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

    # ground truth
    f = open('ground_truth/new_record.txt', 'r')

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
    root = args.video_name
    video_name = root.split('/')[-1].split('.')[0]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    for frame in get_frames(root):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))

            # ground truth
            line = f.readline()
            bbox_label = line.split(',')
            bbox_label = list(map(int, bbox_label))
            left_top_label = (bbox_label[0], bbox_label[1])
            right_bottom_label = (
                bbox_label[0]+bbox_label[2], bbox_label[1]+bbox_label[3])

            left_top = (bbox[0], bbox[1])
            right_bottom = (bbox[0]+bbox[2], bbox[1]+bbox[3])

            center = ((left_top[0] + right_bottom[0]) / 2,
                      (left_top[1] + right_bottom[1]) / 2)
            center_label = ((left_top_label[0] + right_bottom_label[0]) / 2,
                            (left_top_label[1] + right_bottom_label[1]) / 2)

            distance = ((center[0] - center_label[0]) **
                        2 + (center[1] - center_label[1]) ** 2) ** 0.5

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.rectangle(frame, (bbox_label[0], bbox_label[1]),
                          (bbox_label[0]+bbox_label[2],
                           bbox_label[1]+bbox_label[3]),
                          (255, 255, 0), 3)
            cv2.putText(frame, "distance:" + str(distance), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255))
            cv2.imshow(video_name, frame)
            cv2.waitKey(40)


if __name__ == "__main__":
    main()
