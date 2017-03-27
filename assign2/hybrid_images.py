from os import listdir
from scipy.io import loadmat
from scipy.ndimage import convolve, imread
from scipy.misc import imresize, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter



def hybrid(folder = "img2"):
        origin_path1 = "".join([folder, "/", "baby_happy.jpg"])
        im1 = imread(origin_path1, flatten=True)
        im1 = imresize(im1, (100, 100))
        im1 = gaussian_filter(im1, sigma=1)
        imsave("img2_hybrid/im1_blur.jpg", im1)

        origin_path2 = "".join([folder, "/", "baby_weird.jpg"])
        im2 = imread(origin_path2, flatten=True)
        im2 = imresize(im2, (100, 100))
        im2_filter = gaussian_filter(im2, sigma=1)
        imsave("img2_hybrid/im2_blur.jpg", im2_filter)

        im2_detail = im2 - im2_filter
        imsave("img2_hybrid/im2_detail.jpg", im2_detail)

        im2_add = im1 + im2_detail
        imsave("img2_hybrid/im2_add.jpg", im2_add)


if __name__ == '__main__':
        hybrid()