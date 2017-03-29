from scipy.ndimage import imread
from scipy.misc import imsave
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

import numpy as np

def quantizeRGB(origImg, k):
    w, h, n_channel = tuple(origImg.shape)
    data = np.reshape(origImg, (w*h, n_channel))
    kmeans = KMeans(n_clusters=k).fit(data)
    clusterIds = kmeans.labels_
    clusterIds = np.reshape(clusterIds, (w,h))
    meanColors = kmeans.cluster_centers_
    outputImg = [kmeans.cluster_centers_[l] for l in kmeans.labels_]
    outputImg = np.reshape(outputImg, (w, h, n_channel))
    return outputImg, meanColors, clusterIds

def computeQuantizationError(origImg, quantizedImg):
    w, h, n_channel = tuple(origImg.shape)
    origImg = np.reshape(origImg, (w * h * n_channel))
    quantizedImg = np.reshape(quantizedImg, (w * h * n_channel))
    error = euclidean(origImg, quantizedImg)
    return error

def colorQuantizeMain():
    for k in range(1, 15):
        img = imread("fish.jpg")
        outputImg, meanColors, clusterIds = quantizeRGB(img, k)
        file_name = "".join(["fish", str(k),".jpg"])
        imsave("part1/" + file_name, outputImg)
        print(computeQuantizationError(img, outputImg))

if __name__ == '__main__':
    colorQuantizeMain()
