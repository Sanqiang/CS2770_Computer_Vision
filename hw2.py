import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import caffe
from sklearn.svm import LinearSVC
import os

caffe.set_device(1)
caffe.set_mode_gpu()

# for solver

solver_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/solver.prototxt"))
weight_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/weights.caffemodel"))


solver = caffe.SGDSolver(solver_path)
solver.net.copy_from(weight_path)

#transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
#transformer.set_mean('data', np.load('/tmp/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
#transformer.set_transpose('data', (2,0,1))
#transformer.set_channel_swap('data', (2,1,0))
#transformer.set_raw_scale('data', 255.0)

# file base
#files = {}
#base_path = "/tmp/caffe/data"

#labels = os.listdir(base_path)
#for label in labels:
#        label_path = "".join((base_path, "/", label))
#        files[label] =  ["".join((base_path, "/", label,"/", p)) for p in os.listdir(label_path)]


# prepare data
#x = []
#y = []
#min_batch_x = []
#min_batch_y = []
#y_i = 0
#for label in labels:
#	for file in files[label]:        
#		img = caffe.io.load_image(file)
#                img = transformer.preprocess('data', img)
#		min_batch_x.append(img)
#		min_batch_y.append(y_i)
#		if len(min_batch_y) == 8:
#			x.append(min_batch_x)
#			y.append(min_batch_y)
#			min_batch_x = []
#			min_batch_y = []
#	y_i += 1


# shuffle data
#idx = random.sample(range(len(x)), len(x))
#x = np.array(x)
#y = np.array(y)
#x = x[idx]
#y = y[idx]

#train_size = round(len(x) * 0.9)
#train_x = x[0:train_size]
#train_y = y[0:train_size]
#test_x = x[train_size:]
#test_y = y[train_size:]

#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_x")), train_x)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_y")), train_y)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_x")), test_x)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y")), test_y)

train_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_x.npy")))
train_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_y.npy")))
test_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_x.npy")))
test_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y.npy")))

print("Training!!!!!!!!!!")

idx = 0
loop = 100
while True:
	solver.net.blobs['data'].data[...] = train_x[idx]
	solver.net.blobs['label'].data[...] = train_y[idx]
	solver.step(1)
	
	idx = (idx + 1) % len(train_x)

	if idx == 0:
		path_snap = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/iter_", str(loop)))
		solver.net.save(path_snap)
		loop += 1
