import os
import numpy as np


def dataLoader_focal():
    files = []
    focal = []
    root = 'data/Video3_tiny/'
    path = os.listdir(root)
    for folder in path:
        focals = []
        file_name = '/images/004.png'
        file = root + folder + file_name
        focal_name = os.listdir(root+folder+'/focal')
        for img in focal_name:
            focals.append(root + folder + '/focal/' + img)
        files.append(file)
        focal.append(focals)
    return files, focal


if __name__ == "__main__":
    dataLoader_focal()
