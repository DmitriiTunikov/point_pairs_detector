import math
import time

import cv2
import numpy as np


class points_pair(object):
    def __init__(self, xl, xr):
        self.xl = xl
        self.xr = xr
        self.center = (xr + xl) / 2
        self.alive = False


def cross_detect(edge_image):
    tic = time.perf_counter()

    height = edge_image.shape[0]
    width = edge_image.shape[1]
    min_dist_between_rails = 50
    max_dist_between_rails = 100

    pairs = [0] * height
    tracks = [0] * height

    depth = 3
    x_eps = 3
    min_track_len = 50
    for y in reversed(range(height)):
        pairs[y] = []
        tracks[y] = []

        for yy in range(y + 1, min(y + depth + 1, height)):
            for pair in pairs[yy]:
                pair.alive = False

        #make correct min/max dist between rails function = f(y)

        for x_l in range(width - 1 - min_dist_between_rails):
            if int(edge_image[y, x_l]) == 0:
                continue

            min_idx = min(x_l + min_dist_between_rails + max_dist_between_rails, width)
            for x_r in range(x_l + min_dist_between_rails, min_idx):
                if int(edge_image[y, x_r]) == 0:
                    continue

                cur_pair = points_pair(x_l, x_r)
                cur_track = [cur_pair]
                pairs[y].append(cur_pair)

                for yy in range(y + 1, min(y + depth + 1, height)):
                    for i, pair in enumerate(pairs[yy]):
                        if math.fabs(pair.center - cur_pair.center) < x_eps:
                            pair.alive = True
                            cur_track.extend(tracks[yy][i])

                tracks[y].append(cur_track)

        #delete old tracks
        if y + depth < height:
            for old_tracks in tracks[y + depth]:
                old_tracks_len = len(old_tracks)
                i = 0
                #while i < old_tracks_len:
                #    if len(old_tracks[i]) < min_track_len:
                #        del old_tracks[i]
                #        old_tracks_len = old_tracks_len - 1
                #    else:
                #        i = i + 1


    for elem in pairs:
        print(len(elem))

    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")

if __name__ == '__main__':
    # загружаем изображение и отображаем его
    image = cv2.imread("test.png", 0)

    image_blur = cv2.GaussianBlur(image, (0, 0), 1)
    cv2.imshow("Gauss blur", image_blur)

    #high_thresh, thresh_im = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #lowThresh = 0.5*high_thresh

    v = np.median(image)
    sigma = 0.6
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image_blur, lower, upper)
    cv2.imshow("Canny edges", edges)

    res = cross_detect(edges)
    #dilation example
    #kernel = np.ones((5, 5), np.uint8)
    #dilation = cv2.dilate(edges, kernel, iterations=1)
    #cv2.imshow("Canny dilation", dilation)

    cv2.waitKey(0)
