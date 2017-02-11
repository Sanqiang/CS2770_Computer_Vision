import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import caffe
import os

# caffe base
caffe.set_device(3)
caffe.set_mode_gpu()

solver_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/models/solver.prototxt"))
weight_path = "".join((os.environ['HOME'], "/private/cs_2770_assign1/model_save1/iterx_3"))

solver = caffe.SGDSolver(solver_path)
solver.net.copy_from(weight_path)


test_x = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_x.npy")))
test_y = np.load("".join((os.environ['HOME'], "/private/cs_2770_assign1/data/test_y.npy")))

acc = 0
for i in range(len(test_x)):
    solver.net.blobs['data'].data[...] = test_x[i]
    solver.net.blobs['label'].data[...] = test_y[i]
    result = solver.net.forward()
    acc += result['accuracy']
acc /= float(len(test_x))
print(acc)

