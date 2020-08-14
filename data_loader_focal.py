import os
import numpy as np


def dataLoader_focal():
    files = []
    focal = []
    focal_count = 15  # 사용할 focal 이미지 개수
    root = 'data/Video3_tiny/'
    path = os.listdir(root)
    for k, folder in enumerate(path):
        if k > 50:  # 0~99 focal 이미지 중 50 이상만 사용
            focals = []
            file_name = '/images/010.png'
            file = root + folder + file_name
            focal_name = os.listdir(root+folder+'/focal')
            for i in range(focal_count):
                focals.append(root + folder + '/focal/' +
                              focal_name[i+65])  # focal 65 ~ 80만 사용
            files.append(file)
            focal.append(focals)
    return files, focal


if __name__ == "__main__":
    dataLoader_focal()
