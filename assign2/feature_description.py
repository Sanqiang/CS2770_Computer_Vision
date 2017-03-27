from scipy.ndimage import imread
import numpy as np
from feature_detection import extract_keypoints

def compute_features(img_path, window_size = 11, k=0.05):
    im = imread(img_path, flatten=True)
    height = im.shape[0]
    width = im.shape[1]

    output = extract_keypoints(img_path, window_size, k)
    dxs = output[3]
    dys = output[4]

    hist_xys = {}
    offset = int(window_size / 2)
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            m_xys = [np.sqrt(iy**2+ix**2) for iy, ix
                in zip(dys[y - offset:y + offset + 1, x - offset:x + offset + 1], dxs[y - offset:y + offset + 1, x - offset:x + offset + 1])]
            theta_xys = [np.arctan2(iy, ix) * 180 / np.pi for iy, ix
                in zip(dys[y - offset:y + offset + 1, x - offset:x + offset + 1], dxs[y - offset:y + offset + 1, x - offset:x + offset + 1])]

            hist_xy = np.zeros((8,), dtype=np.float)
            for m, theta in zip(m_xys, theta_xys):
                while theta < 0:
                    theta += 360
                while theta > 360:
                    theta -= 360

                hist_xy[int(theta / 45)] = m

            # normalize
            hist_xy = [val if val <= 0.2 else 0.2 for val in hist_xy]
            denorm = sum(hist_xy)
            hist_xy /= denorm
            hist_xy = [val if val <= 0.2 else 0.2 for val in hist_xy]

            hist_xys[(x, y)] = hist_xy
    return hist_xys
