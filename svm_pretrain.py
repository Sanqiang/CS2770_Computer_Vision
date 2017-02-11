import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from sklearn.svm import LinearSVC
import os
from sklearn.metrics import confusion_matrix

# caffe base

caffe.set_device(3)
caffe.set_mode_gpu()

net = caffe.Net('/tmp/caffe/models/deploy.prototxt',
        '/tmp/caffe/models/weights.caffemodel', caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


train_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_xfeatures.npy")))
train_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_y.npy")))
valid_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_xfeatures.npy")))
valid_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_y.npy")))
test_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_xfeatures.npy")))
test_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y.npy")))

model = LinearSVC(max_iter=10000)
model.fit(train_x, train_y)
score = model.score(test_x, test_y)
test_y_pred = model.predict(test_x)
print(confusion_matrix(test_y, test_y_pred))
print(score)

