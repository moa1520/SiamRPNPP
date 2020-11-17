from get_frame import get_frames
from IOU import IOU
import os
import argparse

import cv2
import torch

from model import ModelBuilder
from tracker import build_tracker

parser = argparse.ArgumentParser(description="tracking demo")
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--gt_on', default=True, type=bool, help='Estimate IoU')
parser.add_argument('--record', default=True, type=bool,
                    help='Save images and IoU accuracy')
parser.add_argument('--start_num', default=20, type=int,
                    help='First focal image number')
parser.add_argument('--last_num', default=50, type=int,
                    help='Last focal image number')
args = parser.parse_args()

# start_num = 70
# last_num = 85
start_num = args.start_num
last_num = args.last_num


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device :", device)

    # ground truth
    gt_on = args.gt_on  # IoU 정확도를 측정할 것인지
    f = open('ground_truth/Non_video4_GT.txt', 'r')  # GT 파일
    record = args.record  # IoU 정확도, 이미지를 저장할 것인지

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

    frame_num = 0
    first_time = True
    current_target = -1
    for frame, focal in get_frames(root, start_num, last_num):
        frame_num += 1
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            # max_index = -1
            # max_val = 0
            # outputs = [tracker.track(cv2.imread(f)) for f in focal]

            # for i in range(len(outputs)):
            #     if outputs[i]['best_score'] >= max_val:
            #         max_val = outputs[i]['best_score']
            #         max_index = i
            # current_target = max_index

            ###########################################
            max_index = -1
            max_val = 0
            if first_time:
                outputs = [tracker.track(cv2.imread(f)) for f in focal]

                for i in range(len(outputs)):
                    if outputs[i]['best_score'] >= max_val:
                        max_val = outputs[i]['best_score']
                        max_index = i
                first_time = False
                current_target = max_index
            else:
                outputs = [tracker.track(cv2.imread(focal[i])) for i in range(
                    current_target - 3, current_target + 3)]

                for i in range(len(outputs)):
                    if outputs[i]['best_score'] >= max_val:
                        max_val = outputs[i]['best_score']
                        max_index = i
                if max_index > 3:
                    current_target = current_target + abs(3 - max_index)
                elif max_index < 3:
                    current_target = current_target - abs(3 - max_index)

            print("Focal Image Index: ", current_target + start_num)

            ground_truth(outputs[max_index]['bbox'][:2],
                         outputs[max_index]['bbox'][2:])

            ########################################################################

            bbox = list(map(int, outputs[max_index]['bbox']))

            # ground truth
            if gt_on:
                line = f.readline()
                bbox_label = line.split(',')
                bbox_label = list(map(int, bbox_label))

                iou = IOU(bbox, bbox_label)

                if record:
                    result_iou = open('ground_truth/result_iou.txt', 'a')
                    result_iou.write(str(iou) + ',')
                    result_iou.close()
                cv2.rectangle(frame, (bbox_label[0], bbox_label[1]),
                              (bbox_label[0]+bbox_label[2],
                               bbox_label[1]+bbox_label[3]),
                              (0, 0, 255), 3)
                cv2.putText(frame, "IoU: " + str(round(iou, 4) * 100) + "%", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 3)
            cv2.putText(frame, "focal: " + str(current_target + start_num), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
            cv2.putText(frame, "frame: " + str(frame_num), (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            cv2.imshow(video_name, frame)

            if record:
                save_image(frame_num, frame)
            cv2.waitKey(40)


def save_image(frame_num, frame):
    '''output 이미지 저장'''
    save_path = os.path.join(
        'data/result', '{:03d}.jpg'.format(frame_num))
    cv2.imwrite(save_path, frame)
    ''''''


def ground_truth(center, size):
    f = open("ground_truth/Video3.txt", 'a')
    data = "%d,%d,%d,%d\n" % (center[0], center[1], size[0], size[1])
    f.write(data)
    f.close()


if __name__ == "__main__":
    main()
