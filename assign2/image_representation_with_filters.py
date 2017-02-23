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
            im = imresize(im, (200, 200))
            imsave(target_path, im)

            # PIL implementation
            # im = Image.open(origin_path).convert('L')
            # im = im.resize((200, 200))
            # im.save(target_path)

def filter_img(folder = "img_processed", num_filter = 48):
    mat = loadmat("img/filters.mat")
    for file in listdir(folder):
        if file[-3:] == "jpg":
            origin_path = "".join([folder, "/", file])
            im = imread(origin_path, flatten=False)

            for i in range(num_filter):
                filter = mat['F'][i]
                c_im = convolve(im, filter, mode="constant", cval=0)

                f, (ax0, ax1, ax2) = plt.subplots(1, 3)
                ax0.imshow(im, cmap = plt.cm.Greys_r)
                ax1.imshow(filter, cmap = plt.cm.Greys_r)
                ax2.imshow(c_im, cmap = plt.cm.Greys_r)

                plt.savefig("".join(["img_filter/", file, "_", str(i), ".jpg"]))
                plt.close()

def dist(folder = "img_processed", num_filter = 48, use_mean = False):
    mat = loadmat("img/filters.mat")
    data = {}
    for file in listdir(folder):
        if file[-3:] == "jpg":
            data[file] = np.ndarray((200,200,48), dtype=np.float)
            origin_path = "".join([folder, "/", file])
            im = imread(origin_path, flatten=False)

            for i in range(num_filter):
                filter = mat['F'][i]
                c_im = convolve(im, filter, mode="constant", cval=0)
                if use_mean:
                    c_im = np.mean(c_im.flatten())

                data[file][:,:,i] = c_im

    # within class
    dists_inclass = []
    for file in data:
        for file2 in data:
            if file[:4] == file2[:4] and file != file2:
                dists_inclass.append(euclidean(data[file].flatten(), data[file2].flatten()))
    print("mean of inclass dist", np.mean(dists_inclass))

    dists_outclass = []
    for file in data:
        for file2 in data:
            if file[:4] != file2[:4]:
                dists_outclass.append(euclidean(data[file].flatten(), data[file2].flatten()))
    print("mean of outclass dist", np.mean(dists_outclass))




if __name__ == '__main__':
    # preprocess()
    # filter_img()
    dist(use_mean=False)
    dist(use_mean=True)