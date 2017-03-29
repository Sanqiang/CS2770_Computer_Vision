from scipy.ndimage import imread
from scipy.misc import imsave, imresize
import numpy as np
from skimage.draw import circle_perimeter
from matplotlib_utils import impixelinfo
import matplotlib.pyplot as plt


def detectEdges(im, threshold):
    edges = []
    dx, dy = np.gradient(im)
    m_xy = np.sqrt(dx**2 + dy**2)
    oriens = np.arctan2(dy, dx)
    len_y, len_x = np.shape(im)
    for y in range(len_y):
        for x in range(len_x):
            if m_xy[y][x] > threshold:
                edges.append((x, y, m_xy[y][x], oriens[y][x]))
    return edges

def detectEdgesMain():
    img = imread("sample.jpg", flatten=True)
    outputs = detectEdges(img, 90)
    for output in outputs:
        x, y, grad, orien = output
        img[y][x] = 0

    imsave("sample_line.jpg", img)

quantization_value = 5
def detectCircles(im, edges, radius, top_k):
    len_y, len_x = np.shape(im)
    hough_space = {}
    # r = radius
    for r in radius:
        for edge in edges:
            x, y, grad, orien = edge
            orien = np.arctan2(y, x)
            a = y - r * np.sin(orien)
            b = x - r * np.cos(orien)

            if a < 0 or a >= len_x or b < 0 and b >= len_y:
                continue

            bin_a = int(np.ceil(a / quantization_value))
            bin_b = int(np.ceil(b / quantization_value))

            if (bin_a, bin_b, r) not in hough_space:
                hough_space[(bin_a, bin_b, r)] = 0
            hough_space[(bin_a, bin_b, r)] += 1

    centers = sorted(hough_space.items(), key=(lambda x: x[1]), reverse=True)[:top_k]
    return centers

def detectCirclesMain():
    pic = "jupiter"
    img = imread(pic + ".jpg", flatten=True)

    len_y, len_x = np.shape(img)

    edges = detectEdges(img, 100)

    radius = range(33, 34)
    # radius = 6
    centers = detectCircles(img, edges, radius, 10)

    for k, v in centers:
        rr, cc = circle_perimeter(k[0] * quantization_value, k[1] * quantization_value, k[2])
        # rr, cc = circle_perimeter(k[0] * quantization_value, k[1] * quantization_value, radius)
        rr_filter = []
        cc_filter = []
        for r,c in zip(rr,cc):
            if r >= 0 and c >= 0 and r < len_y and c < len_x:
                rr_filter.append(r)
                cc_filter.append(c)
        img[rr_filter, cc_filter] = 0
    file_name = "part2/" + pic + "_r_" + str(radius[0]) +  "_" + str(radius[-1]) + ".jpg"
    imsave(file_name, img)

if __name__ == '__main__':
    # detectEdgesMain()
    detectCirclesMain()