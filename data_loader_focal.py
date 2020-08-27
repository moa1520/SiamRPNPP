import os
import numpy as np


def dataLoader_focal(start, end):
    files = []
    focal = []
    # root = 'data/Video3_tiny/'
    root = 'E:/NonVideo4/'
    path = os.listdir(root)
    for folder in path:
        focals = []
        file_name = '/images/005.png'
        file = root + folder + file_name
        focal_name = os.listdir(root+folder+'/focal')
        # for i in range(len(focal_name)):
        for f in focal_name[start:end]:
            focals.append(root + folder + '/focal/' + f)
        files.append(file)
        focal.append(focals)
    return files, focal


if __name__ == "__main__":
    dataLoader_focal(20, 40)
