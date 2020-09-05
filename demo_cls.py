import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model import ModelBuilder
from tracker import build_tracker
from plenoptic_dataloader import PlenopticDataLoader

# parser = argparse.ArgumentParser(description="tracking demo")
# parser.add_argument('--video_name', default='', type=str,
#                     help='videos or image files')
# args = parser.parse_args()

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
        dataLoader_focal = PlenopticDataLoader(
            root='E:/NonVideo4', img2d_ref='images/005.png', focal_range=(start_num, last_num))
        img2d_files, focal_files = dataLoader_focal.dataLoader_focal()
        for i in range(len(img2d_files)):
            frame = cv2.imread(img2d_files[i])
            yield frame, focal_files[i]
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
    first_time = True
    current_target = -1
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
            outputs = []
            max_index = -1
            sum_cls = torch.Tensor(1, 10, 25, 25)
            if first_time:
                max_index = tracker.get_cls(focal)

                first_time = False
                current_target = max_index
            else:
                max_index = tracker.get_cls(
                    focal[current_target - 3:current_target + 3])

                if max_index > 3:
                    current_target = current_target + abs(3 - max_index)
                elif max_index < 3:
                    current_target = current_target - abs(3 - max_index)

            print("Focal Image Index: ", current_target + start_num)

            '''ouput 이미지 저장'''
            # save_img = outputs[max_index]['x_crop'].data.cpu().squeeze(
            #     0).numpy().transpose((1, 2, 0)).astype(np.int32)
            #s = ret['detection_cropped_resized']
            # save_path = os.path.join(
            #     'data/x_crop', '{:03d}_detection_input.jpg'.format(a))
            # cv2.imwrite(save_path, save_img)
            ''''''
            output = tracker.track(cv2.imread(focal[max_index]))

            bbox = list(map(int, output['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.imshow(video_name, frame)

            '''output 이미지 저장'''
            save_path = os.path.join(
                'data/result', '{:03d}_detection_input.jpg'.format(a))
            cv2.imwrite(save_path, frame)
            ''''''
            cv2.waitKey(40)


if __name__ == "__main__":
    main()
