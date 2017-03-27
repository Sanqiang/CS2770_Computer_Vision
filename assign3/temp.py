from cyvlfeat.sift import sift
from sklearn.cluster import KMeans
from scipy.ndimage import imread
from scipy.misc import imresize
import numpy as np
from sklearn import svm
import os
import random as rd

K = 100
y = []
x = []

home = os.environ["HOME"]
base_path = "/".join([home, "data/scene_categories/"])
labels = os.listdir(base_path)
labels.remove(".DS_Store") # for avoid mac file
label2idx = {}
for label in labels:
    if label not in label2idx:
        label2idx[label] = len(label2idx)

for label in labels:
    label_path = "".join((base_path, "/", label))
    for image_file_name in os.listdir(label_path):
        if image_file_name[-3:] == "jpg":
            image_path = "".join((base_path, "/", label,"/", image_file_name))
            x.append(image_path)
            y.append(label2idx[label])

#split into train / test
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

# run kmeans
descriptors = []
for image_path in train_x:
    img = imread(image_path, flatten=True)
    # img = imresize(img, (200, 200))
    result = sift(img, compute_descriptor=True, n_levels=2)
    [descriptors.append(l) for l in result[1]]
kmeans = KMeans(n_clusters=K, random_state=0).fit(descriptors)

# caculate features
train_x_features = []
test_x_features = []
for image_path in train_x:
    img = imread(image_path, flatten=True)
    # img = imresize(img, (200, 200))
    result = sift(img, compute_descriptor=True, n_levels=2)
    features = kmeans.predict(result[1])
    features_hist = [0] * K
    for k in features:
        features_hist[k] += 1
    train_x_features.append(features_hist)
for image_path in test_x:
    img = imread(image_path, flatten=True)
    # img = imresize(img, (200, 200))
    result = sift(img, compute_descriptor=True, n_levels=2)
    features_hist = [0] * K
    for k in features:
        features_hist[k] += 1
    test_x_features.append(features_hist)
train_x_features = np.array(train_x_features)
test_x_features = np.array(test_x_features)

#svm
clf = svm.SVC()
clf.fit(train_x_features, train_y)
print(clf.score(test_x_features, test_y))
