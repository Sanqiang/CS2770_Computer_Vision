# import PIL
# from PIL import Image
from os import listdir
from scipy.io import loadmat
from scipy.ndimage import convolve, imread
from scipy.misc import imresize, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

def preprocess(folder = "img"):
    for file in listdir(folder):
        if file[-3:] == "jpg":
            origin_path = "".join([folder, "/", file])
            target_path = "".join([folder, "_processed/", file])

            # scipy implementation
            im = imread(origin_path, flatten=True)
            im = imresize(im, (100, 100))
            imsave(target_path, im)

            # PIL implementation
            # im = Image.open(origin_path).convert('L')
            # im = im.resize((100, 100))
            # im.save(target_path)

def filter_img(folder = "img_processed", num_filter = 48):
    mat = loadmat("filters.mat")
    for file in listdir(folder):
        if file[-3:] == "jpg":
            origin_path = "".join([folder, "/", file])
            im = imread(origin_path)
            im_origin = imread("".join(["img", "/", file]))

            for i in range(num_filter):
                filter = mat['F'][:, :, i]
                c_im = np.ndarray((im.shape))
                convolve(im, filter, c_im)

                f, (ax0, ax1, ax2) = plt.subplots(1, 3)
                ax0.imshow(im_origin)
                ax1.imshow(filter)
                ax2.imshow(c_im, cmap=plt.cm.Greys_r)

                plt.savefig("".join(["img_filter/", file, "_", str(i), ".jpg"]))
                plt.close()

                # break

def calculate_dist(folder = "img_processed", num_filter = 48, use_mean = False, use_norm = False):
    mat = loadmat("filters.mat")
    data = {}
    for file in listdir(folder):
        if file[-3:] == "jpg":
            data[file] = np.ndarray((100,100,48), dtype=np.float)
            origin_path = "".join([folder, "/", file])
            im = imread(origin_path)

            for i in range(num_filter):
                filter = mat['F'][:, :, i]
                c_im = np.ndarray((im.shape))
                convolve(im, filter, c_im)
                c_im = convolve(im, filter, mode="constant", cval=0)

                data[file][:,:,i] = c_im

    # within class
    dists_inclass = []
    for file in data:
        for file2 in data:
            if file[:4] == file2[:4] and file != file2:
                data1 = data[file].flatten()
                data2 = data[file2].flatten()
                if use_norm:
                    data1 /= np.sum(data1)
                    data2 /= np.sum(data2)
                if use_mean:
                    dist = euclidean(np.mean(data1), np.mean(data2))
                else:
                    dist = euclidean(data1, data2)
                    dist /= len(data1)
                dist = np.mean(dist)
                dists_inclass.append(dist)
    print("mean of distance between same class", np.mean(dists_inclass))

    dists_outclass = []
    for file in data:
        for file2 in data:
            if file[:4] != file2[:4]:
                data1 = data[file].flatten()
                data2 = data[file2].flatten()
                if use_norm:
                    data1 /= np.sum(data1)
                    data2 /= np.sum(data2)
                if use_mean:
                    dist = euclidean(np.mean(data1), np.mean(data2))
                else:
                    dist = euclidean(data1, data2)
                    dist /= len(data1)
                dist = np.mean(dist)
                dists_outclass.append(dist)
    print("mean of distance between different classes", np.mean(dists_outclass))




if __name__ == '__main__':
    # preprocess()
    # filter_img()
    calculate_dist(use_mean=False, use_norm=False)
    calculate_dist(use_mean=True, use_norm=False)
    # calculate_dist(use_mean=False, use_norm=True)
    # calculate_dist(use_mean=True, use_norm=True)