from os import listdir
from scipy.io import loadmat
from scipy.ndimage import convolve, imread
from scipy.misc import imresize, imsave, imshow
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter

def extract_keypoints(img_path, window_size = 11, k=0.05):
    im = imread(img_path, flatten=True)
    height = im.shape[0]
    width = im.shape[1]
    # im = imresize(im, (200, 200))
    dy, dx = np.gradient(im)
    i_x = dx ** 2
    i_xy = dy * dx
    i_y = dy ** 2

    xs = []
    ys = []
    rs = []
    dxs = []
    dys = []
    offset = int(window_size / 2)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            ixx = i_x[y - offset:y + offset + 1, x - offset:x + offset + 1]
            ixy = i_xy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            iyy = i_y[y - offset:y + offset + 1, x - offset:x + offset + 1]
            sxx = ixx.sum()
            sxy = ixy.sum()
            syy = iyy.sum()

            det = (sxx * syy) - (sxy**2)
            trace = sxx + syy
            r = det - k*(trace**2)

            xs.append(x)
            ys.append(y)
            rs.append(r)
            dxs.append(dx[y, x])
            dys.append(dy[y, x])

    output = (xs, ys, rs, dxs, dys)
    return output

def find_corner(img_path, output, oimg_path, thres = -1, cornner_cnt = 10000):
    im = imread(img_path, flatten=True)
    im_origin = imread(img_path, flatten=False)
    if thres == -1:
        s_idxs = sorted(range(len(output[2])), key=lambda k: output[2][k], reverse=True)
        for i in range(cornner_cnt):
            x, y, r = output[0][s_idxs[i]], output[1][s_idxs[i]], output[2][s_idxs[i]]
            im[y, x] = 0
            im[y - 1, x - 1] = 0
            im[y + 1, x - 1] = 0
            im[y - 1, x + 1] = 0
            im[y + 1, x + 1] = 0
    else:
        for i in range(len(output[0])):
            x, y, r = output[0][i], output[1][i], output[2][i]
            if r > thres:
                im[y, x] = 0
                im[y - 1, x - 1] = 0
                im[y + 1, x - 1] = 0
                im[y - 1, x + 1] = 0
                im[y + 1, x + 1] = 0
    # imsave(oimg_path, im)
    f, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(im_origin)
    ax1.imshow(im, cmap=plt.cm.Greys_r)

    plt.savefig(oimg_path)
    plt.close()


def process(input_folder="img3", output_folder="img3_result"):
    for file in listdir(input_folder):
        if file[:3] == "img":
            input_path = "".join([input_folder, "/", file])
            output = extract_keypoints(input_path)
            output_path = "".join([output_folder, "/", file])
            find_corner(input_path, output, output_path)

if __name__ == '__main__':
    process()

