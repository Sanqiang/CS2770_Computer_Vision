from cyvlfeat.sift import sift
from sklearn.cluster import KMeans
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
from sklearn import svm
import os
import random as rd

K = 100
home = os.environ["HOME"]

def computeSPMHistogram(im, means):
    result = sift(im, compute_descriptor=True)
    features = means.predict(result[1])
    pyramid = [0] * K
    for k in features:
        pyramid[k] += 1
    return pyramid

def SPMMain():
    y = []
    x = []
    xc = {}

    base_path = "/".join([home, "data/scene_categories/"])
    labels = os.listdir(base_path)
    # labels.remove(".DS_Store")  # for avoid mac file
    label2idx = {}
    for label in labels:
        if label not in label2idx:
            label2idx[label] = len(label2idx)

    for label in labels:
        label_path = "".join((base_path, "/", label))
        for image_file_name in os.listdir(label_path):
            if image_file_name[-3:] == "jpg":
                image_path = "".join((base_path, "/", label, "/", image_file_name))
                x.append(image_path)

                label_idx = label2idx[label]
                y.append(label_idx)

                if label_idx not in xc:
                    xc[label_idx] = []
                xc[label_idx].append(image_path)

    if False:
        # random split into train / test
        data_len = len(y)
        rd_idxs = rd.sample(range(data_len), data_len)
        split = round(0.8 * len(rd_idxs))
        train_idx = rd_idxs[:split]
        test_idx = rd_idxs[split:]
        x = np.array(x)
        y = np.array(y)
        train_x = x[train_idx]
        train_y = y[train_idx]
        test_x = x[test_idx]
        test_y = y[test_idx]
    if True:
        # split by category
        train_x = []
        test_x = []
        train_y = []
        test_y = []
        for label_idx in xc:
            data_len = len(xc[label_idx])
            rd_idxs = rd.sample(range(data_len), data_len)
            split = round(0.5 * len(rd_idxs))
            train_idx = rd_idxs[:split]
            test_idx = rd_idxs[split:]
            x_temp = np.array(xc[label_idx])
            train_x.extend(x_temp[train_idx])
            test_x.extend(x_temp[test_idx])
            train_y.extend([label_idx] * len(x_temp[train_idx]))
            test_y.extend([label_idx] * len(x_temp[test_idx]))

    # run kmeans
    descriptors = []
    for image_path in train_x:
        img = imread(image_path, flatten=True)
        # img = imresize(img, (200, 200))
        result = sift(img, compute_descriptor=True)
        [descriptors.append(l) for l in result[1]]
    kmeans = KMeans(n_clusters=K, max_iter=10000, n_jobs=-1, n_init=50).fit(descriptors)
    print("finished keams.")

    # caculate features
    train_x_features = []
    test_x_features = []
    for image_path in train_x:
        img = imread(image_path, flatten=True)
        # img = imresize(img, (200, 200))
        pyramid = computeSPMHistogram(img, kmeans)
        train_x_features.append(pyramid)
    for image_path in test_x:
        img = imread(image_path, flatten=True)
        # img = imresize(img, (200, 200))
        pyramid = computeSPMHistogram(img, kmeans)
        test_x_features.append(pyramid)
    # train_x_features = np.array(train_x_features)
    # test_x_features = np.array(test_x_features)

    # svm
    clf = svm.SVC()
    clf.fit(train_x_features, train_y)
    print(clf.score(test_x_features, test_y))

if __name__ == '__main__':
    SPMMain()


