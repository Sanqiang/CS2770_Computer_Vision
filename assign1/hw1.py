import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from sklearn.svm import LinearSVC
import os

# caffe base

caffe.set_device(2)
caffe.set_mode_gpu()

net = caffe.Net('/tmp/caffe/models/deploy.prototxt', 
	'/tmp/caffe/models/weights.caffemodel', caffe.TEST) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# file base
files = {}
base_path = "/tmp/caffe/data"

labels = os.listdir(base_path)
for label in labels:
	label_path = "".join((base_path, "/", label))
	files[label] =  ["".join((base_path, "/", label,"/", p)) for p in os.listdir(label_path)]


# prepare data
y_i = 0
x = []
y = []
for label in labels:
	for file in files[label]:
		img = caffe.io.load_image(file)
		img = transformer.preprocess('data', img)
		net.blobs['data'].data[...] = img
		net.forward()
		x.append(net.blobs['fc8'].data[0])
		cur_y = [0] * len(labels)
		cur_y[y_i] = 1
		#y.append(cur_y)
		y.append(y_i)
	y_i += 1
	
	
train_size = round(len(x) * 0.9)

# shuffle data
idx = random.sample(range(len(x)), len(x))
x = np.array(x)
y = np.array(y)
x = x[idx]
y = y[idx]

#seperate train, test
train_x = x[0:train_size]
train_y = y[0:train_size]
test_x = x[train_size:]
test_y = y[train_size:]


# svm model
model = LinearSVC()
model.fit(train_x, train_y)
score = model.score(test_x, test_y)

print("score:", score)
