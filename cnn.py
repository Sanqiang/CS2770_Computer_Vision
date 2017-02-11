import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os
import copy

# caffe base

caffe.set_device(1)
caffe.set_mode_gpu()


# for solver

solver_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/solver.prototxt"))
weight_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/weights.caffemodel"))


solver = caffe.SGDSolver(solver_path)
solver.net.copy_from(weight_path)


train_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_x.npy")))
train_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/train_y.npy")))
valid_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_x.npy")))
valid_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/valid_y.npy")))
test_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_x.npy")))
test_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y.npy")))

#batch up

batch = 157

temp_x = []
temp_y = []
min_batch_x = []
min_batch_y = []
for i in range(len(train_x)):
	min_batch_x.append(train_x[i])
	min_batch_y.append(train_y[i])
	if batch == len(min_batch_x):
               temp_x.append(np.array(min_batch_x))
               temp_y.append(np.array(min_batch_y))
               min_batch_x = []
               min_batch_y = []
train_x = temp_x
train_y = temp_y

temp_x = []
temp_y = []
min_batch_x = []
min_batch_y = []
for i in range(len(test_x)):
        min_batch_x.append(test_x[i])
       	min_batch_y.append(test_y[i])
        if batch == len(min_batch_x):
               temp_x.append(np.array(min_batch_x))
               temp_y.append(np.array(min_batch_y))
               min_batch_x = []
               min_batch_y = []
test_x = temp_x
test_y = temp_y


temp_x = []
temp_y = []
min_batch_x = []
min_batch_y = []
for i in range(len(valid_x)):
        min_batch_x.append(valid_x[i])
       	min_batch_y.append(valid_y[i])
        if batch == len(min_batch_x):
               temp_x.append(np.array(min_batch_x))
               temp_y.append(np.array(min_batch_y))
               min_batch_x = []
               min_batch_y = []
valid_x	= temp_x
valid_y	= temp_y

# train


path_acc = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/acc.txt"))
path_loss = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/loss.txt"))

loss = []
acc_train = []
idx = 0
min_batch_x = []
min_batch_y = []

epoch = 0
idx = 0
while epoch < 25000:	
	solver.net.blobs['data'].data[...] = train_x[idx]
        solver.net.blobs['label'].data[...] = train_y[idx]
        solver.step(1)
	
	loss.append(copy.deepcopy(solver.net.blobs['loss'].data))
	acc_train.append(copy.deepcopy(solver.net.blobs['accuracy'].data))	
	
	
	#print(train_x[idx].shape)
	#print(train_y[idx].shape)
	
        idx = (idx + 1) % len(train_x)
	
        if idx == 0:
                f_acc = open(path_acc, "a")
                f_loss = open(path_loss, "a")
		
                #path_snap = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/iterx_", str(epoch)))
                path_snap = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/iter"))

		solver.net.save(path_snap)
                acc = 0
                
                for i in range(len(valid_x)):
                        solver.net.blobs['data'].data[...] = valid_x[i]
                        solver.net.blobs['label'].data[...] = valid_y[i]
                        solver.net.forward()
                        acc += copy.deepcopy(solver.net.blobs['accuracy'].data)
			#print(str(acc))
                acc /= float(len(valid_x))
                
		
		
                f_acc.write(str(acc))
                f_acc.write("\t")
		f_acc.write(str(np.mean(acc_train)))
		f_acc.write("\n")
                f_loss.write(str(np.mean(loss)))
                f_loss.write("\n")
		
		print(len(loss))
				

		loss = []
		acc_train = []
                f_acc.close()
                f_loss.close()

                epoch += 1


for i in range(len(test_x)):
    solver.net.blobs['data'].data[...] = test_x[i]
    solver.net.blobs['label'].data[...] = test_y[i]
    result = solver.net.forward()
    acc += result['accuracy']
acc /= float(len(test_x))
f_acc = open(path_acc, "a")
f_acc.write("test")
f_acc.write(str(acc))
f_acc.close()
