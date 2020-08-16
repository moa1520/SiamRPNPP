import os
import numpy as np


def dataLoader():
    files = []
    # root = 'data/Video3_tiny/'
    root = 'E:/NonVideo4/'
    path = os.listdir(root)
    for folder in path:
        file_name = '/images/005.png'
        file = root + folder + file_name
        files.append(file)
    return files


if __name__ == "__main__":
    dataLoader()
