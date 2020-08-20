import os
import numpy as np


def dataLoader_focal():
    files = []
    focal = []
    # root = 'data/Video3_tiny/'
    root = 'E:/NonVideo4/'
    path = os.listdir(root)
    for i, folder in enumerate(path):
        focal_count = 28
        if i >= 50:
            focal_count = 29
        if i >= 100:
            focal_count = 31
        file_name = '/images/005.png'
        file = root + folder + file_name
        focal_name = os.listdir(root+folder+'/focal')
        focal.append(root + folder + '/focal/' +
                     focal_name[focal_count])
        files.append(file)
    return files, focal


if __name__ == "__main__":
    dataLoader_focal()
