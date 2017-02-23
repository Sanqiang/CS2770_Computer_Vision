from os import listdir
from scipy.io import loadmat
from scipy.ndimage import convolve, imread
from scipy.misc import imresize, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter

def extract_keypoints(img_path, window_size = 5, thresh=-1, k=0.05):
    im = imread(img_path, flatten=True)
    # im = imresize(im, (200, 200))
    dy, dx = np.gradient(im)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2
    height = im.shape[0]
    width = im.shape[1]

    cornerList = []
    offset = int(window_size / 2)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            if r > thresh and thresh != -1:
                cornerList.append([x, y, r])
                im[y, x] = 0
                im[y - 1, x - 1] = 0
                im[y + 1, x - 1] = 0
                im[y - 1, x + 1] = 0
                im[y + 1, x + 1] = 0
            elif thresh == -1:
                cornerList.append([x, y, r])

    if thresh == -1:
        cornerList = sorted(cornerList, reverse=True, key=lambda x:x[2])
        for i in range(1000):
            corner = cornerList[i]
            x, y, r = corner[0], corner[1], corner[2]
            im[y, x] = 0
            im[y - 1, x - 1] = 0
            im[y + 1, x - 1] = 0
            im[y - 1, x + 1] = 0
            im[y + 1, x + 1] = 0


    imsave("xxx.jpg", im)


if __name__ == '__main__':
    path = "metal.jpg"
    extract_keypoints(path)