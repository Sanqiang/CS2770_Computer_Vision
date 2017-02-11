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

solver_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models_adam/solver.prototxt"))
weight_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models_adam/weights.caffemodel"))


solver = caffe.AdamSolver(solver_path)
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
#       files[label] =  ["".join((base_path, "/", label,"/", p)) for p in os.listdir(label_path)]


# prepare data
#x = []
#y = []
#y_i = 0
#for label in labels:
#	for file in files[label]:        
#		img = caffe.io.load_image(file)
#                img = transformer.preprocess('data', img)		
#		x.append(img)
#		y.append(y_i)
#	y_i += 1


# shuffle data
#idx = random.sample(range(len(x)), len(x))
#x = np.array(x)
#y = np.array(y)
#x = x[idx]
#y = y[idx]

#batch = 64
# reduce
#temp_x = []
#temp_y = []
#min_batch_x = []
#min_batch_y = []
#for i in range(len(x)):
#	min_batch_x.append(x[i])
#	min_batch_y.append(y[i])
#	if batch == len(min_batch_x):
#		temp_x.append(min_batch_x)
#		temp_y.append(min_batch_y)
#		min_batch_x = []
#		min_batch_y = []

#x = temp_x
#y = temp_y
#x = np.array(x)
#y = np.array(y)


#split_1 = round(len(x) * 0.8)
#split_2 = round(len(x) * 0.9)

#train_x = x[0:split_1]
#train_y = y[0:split_1]
#valid_x = x[split_1:split_2]
#valid_y = y[split_1:split_2]
#test_x = x[split_2:]
#test_y = y[split_2:]

#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_x")), train_x)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_y")), train_y)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_x")), valid_x)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_y")), valid_y)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_x")), test_x)
#np.save("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y")), test_y)

train_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_x.npy")))
train_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_y.npy")))
valid_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_x.npy")))
valid_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_y.npy")))
test_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_x.npy")))
test_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y.npy")))

print("Training!!!!!!!!!!")

path_deploy = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models_adam/deploy.prototxt"))

path_acc = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models_adam/acc.txt"))
path_loss = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models_adam/loss.txt"))

idx = 0
loop = 100
while True:
	solver.net.blobs['data'].data[...] = train_x[idx]
	solver.net.blobs['label'].data[...] = train_y[idx]
	solver.step(1)
	
	idx = (idx + 1) % len(train_x)

	if idx == 0:
		f_acc = open(path_acc, "a")
		f_loss = open(path_loss, "a")
				
		path_snap = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models_adam/iter_", str(loop)))
		solver.net.save(path_snap)
		acc = 0
		loss = 0
		for i in range(len(valid_x)):
			solver.net.blobs['data'].data[...] = valid_x[i] 
			solver.net.blobs['label'].data[...] = valid_y[i]
			result = solver.net.forward()
			acc += result['accuracy']
			loss += result['loss']
		acc /= float(len(valid_x))
		loss /= float(len(valid_x))

		f_acc.write(str(acc))
		f_acc.write("\n")
		f_loss.write(str(loss))
		f_loss.write("\n")

		f_acc.close()
		f_loss.close()

		loop += 1
		
