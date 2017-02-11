import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from sklearn.svm import LinearSVC
import os
from sklearn.metrics import confusion_matrix
random.seed(1)
# caffe base

caffe.set_device(3)
caffe.set_mode_gpu()

deploy_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/deploy.prototxt"))
weight_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/model_save1/iterx_3"))

net = caffe.Net(deploy_path, weight_path, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)


files = {}
base_path = "/tmp/caffe/data"

labels = os.listdir(base_path)
for label in labels:
        label_path = "".join((base_path, "/", label))
        files[label] =  ["".join((base_path, "/", label,"/", p)) for p in os.listdir(label_path)]


#x = []
y = []
x_features = []
y_i = 0
for label in labels:
        for file in files[label]:
                img = caffe.io.load_image(file)
                img = transformer.preprocess('data', img)
                net.blobs['data'].data[...] = img
                output = net.forward()
		#print(net.blobs['nfc8'].data.shape)		
                x_features.append(np.reshape(copy.deepcopy(net.blobs['nfc8'].data), (160,)))
		#x.append(img)
                y.append(y_i)
        y_i += 1


idx = random.sample(range(len(y)), len(y))
#x = np.array(x)
x_features = np.array(x_features)
y = np.array(y)
#x = x[idx]
y = y[idx]
x_features = x_features[idx]

split_1 = round(len(y) * 0.8)
split_2 = round(len(y) * 0.9)

#train_x = x[0:split_1]
train_xfeatures = x_features[0:split_1]
train_y = y[0:split_1]
#valid_x = x[split_1:split_2]
valid_xfeatures = x_features[split_1:split_2]
valid_y = y[split_1:split_2]
#test_x = x[split_2:]
test_xfeatures = x_features[split_2:]
test_y = y[split_2:]



model = LinearSVC(max_iter=10000)
model.fit(train_xfeatures, train_y)
score = model.score(test_xfeatures, test_y)
test_y_pred = model.predict(test_xfeatures)
print(confusion_matrix(test_y, test_y_pred))
print(score)

