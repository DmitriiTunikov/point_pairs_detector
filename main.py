import math
import random
import time
import cv2
import numpy as np


class points_pair(object):
    def __init__(self, xl, xr):
        self.xl = xl
        self.xr = xr
        self.center = (xr + xl) / 2
        self.alive = False


def has_same_pair_for_list(pairs, new_pair):
    rail_eps = 11
    center_eps = 11

    for pair in pairs:
        if math.fabs(pair.center - new_pair.center) < center_eps and \
                math.fabs(pair.xl - new_pair.xl) < rail_eps and \
                math.fabs(pair.xr - new_pair.xr) < rail_eps:
            return True

    return False


def has_same_pair(pairs, new_pair, new_pair_y):
    center_eps = 10
    if new_pair_y not in pairs:
        return False
    elif math.fabs(pairs[new_pair_y].center - new_pair.center) < center_eps:
        return True
    else:
        return False


def abs(x, y):
    if x - y > 0:
        return x - y
    else:
        return -x + y


def contains_list(lists, check_list):
    for list_elem in lists:
        if abs(len(list_elem), len(check_list)) > 1:
            continue

        if list_elem[len(list_elem) - 1] == check_list[len(check_list) - 1]:
            return True

    return False


def cross_track_with_track(track1, track2):
    eps = 10

    res = []
    for pair1 in track1:
        for pair2 in track2:
            if abs(pair1[0], pair2[0]) < eps:
                y = int((pair1[0] + pair2[0]) / 2)
                x = -1

                if abs(pair1[1].xl, pair2[1].xl) < eps:
                    x = int((pair1[1].xl + pair2[1].xl) / 2)
                if abs(pair1[1].xr, pair2[1].xr) < eps:
                    x = int((pair1[1].xr + pair2[1].xr) / 2)

                if x != -1:
                    flag = False
                    for old_res_elem in res:
                        if abs(old_res_elem[0], x) < eps and abs(old_res_elem[1], y) < eps:
                            flag = True
                            break
                    if not flag:
                        res.append((x, y))

    return res


def find_cross_by_tracks(tracks):
    res = []

    all_tracks_as_lists = []
    for track_as_lists in tracks:
        all_tracks_as_lists.extend(track_as_lists)

    for i, track1 in enumerate(all_tracks_as_lists):
        for j in range(i + 1, len(all_tracks_as_lists)):
            track2 = all_tracks_as_lists[j]
            res.extend(cross_track_with_track(track1, track2))

    return res


def make_accum_arr(width, height, cell_size):
    h_size = int(height / cell_size)
    w_size = int(width / cell_size)

    res = [0] * h_size

    for i in range(h_size):
        res[i] = [0] * w_size

    #res = [[0] * w_size for i in range(h_size)] * h_size

    return res


def find_cross_by_accum(tracks, width, height, cell_size):
    accum_arr = make_accum_arr(width, height, cell_size)

    for track_as_lists in tracks:
        for track_as_pairs_list in track_as_lists:
            for pair in track_as_pairs_list:
                h_idx = int(pair[0] / cell_size)
                w_idxs = [int(pair[1].xl / cell_size), int(pair[1].xr / cell_size)]
                for w_idx in w_idxs:
                    accum_arr[h_idx][w_idx] = accum_arr[h_idx][w_idx] + 1

    maxs = []
    max_count = 4

    for h_idx, accum_str in enumerate(accum_arr):
        for w_idx, accum_elem in enumerate(accum_str):
            if len(maxs) < max_count and accum_elem > 0:
                maxs.append((accum_elem, (h_idx, w_idx)))
            else:
                for max_idx, max_elem in enumerate(maxs):
                    if accum_elem > max_elem[0]:
                        maxs[max_idx] = (accum_elem, (h_idx, w_idx))
                        break

    res = []

    for max_elem in maxs:
        res.append((int(max_elem[1][1] * cell_size + cell_size / 2), int(max_elem[1][0] * cell_size + cell_size / 2)))

    return res

def cross_detect_list(edge_image):
    tic = time.perf_counter()

    height = edge_image.shape[0]
    width = edge_image.shape[1]
    min_dist_between_rails = 50
    max_dist_between_rails = 150

    pairs = [0] * height
    tracks = [0] * height

    depth = 3
    x_eps = 30
    min_track_len = 200
    it_heigh = height
    #int(0.3 * height)
    for y in reversed(range(height)):
        pairs[y] = []
        tracks[y] = []

        for yy in range(y + 1, min(y + depth + 1, it_heigh)):
            for pair in pairs[yy]:
                pair.alive = False

        # make correct min/max dist between rails function = f(y)

        for x_l in range(width - 1 - min_dist_between_rails):
            if int(edge_image[y, x_l]) == 0:
                continue

            min_idx = min(x_l + min_dist_between_rails + max_dist_between_rails, width)
            for x_r in range(x_l + min_dist_between_rails, min_idx):
                if int(edge_image[y, x_r]) == 0:
                    continue

                cur_pair = points_pair(x_l, x_r)
                if has_same_pair_for_list(pairs[y], cur_pair):
                    continue
                #KEK
                new_track_point = (y, cur_pair)
                cur_tracks_as_list = []

                pairs[y].append(cur_pair)
                for yy in range(y + 1, min(y + depth + 1, it_heigh)):
                    size = len(tracks[yy])
                    i = 0
                    while i < size:
                        track_yy_list = tracks[yy][i]

                        if contains_list(cur_tracks_as_list, track_yy_list):
                            size = size - 1
                            del tracks[yy][i]
                            continue

                        i = i + 1
                        track_yy_first_pair = track_yy_list[len(track_yy_list) - 1]

                        #if track_yy_first_pair[1].xr - track_yy_first_pair[1].xl > \
                        #        (cur_pair.xr - cur_pair.xl) * 1.5:
                        #    continue

                        track_yy_last_pair = track_yy_list[len(track_yy_list) - 1]

                        y_diff = abs(track_yy_last_pair[0], y)
                        xl_diff = math.fabs(track_yy_last_pair[1].xl - cur_pair.xl)
                        xr_diff = math.fabs(track_yy_last_pair[1].xr - cur_pair.xr)
                        c_diff = math.fabs(track_yy_last_pair[1].center - cur_pair.center)

                        if y_diff <= depth and xl_diff < x_eps and xr_diff < x_eps and c_diff < x_eps:
                            new_track = track_yy_list
                            new_track.append(new_track_point)
                            cur_tracks_as_list.append(new_track)

                if len(cur_tracks_as_list) == 0:
                    cur_tracks_as_list.append([new_track_point])
                tracks[y].extend(cur_tracks_as_list)
        print(y)

    for track_as_lists in tracks:
        #if isinstance(track_as_lists, list):
        if len(track_as_lists) > 0:
            size = len(track_as_lists)
            i = 0
            while i < size:
                if len(track_as_lists[i]) < min_track_len:
                    del track_as_lists[i]
                    size = size - 1
                    continue
                low_width = track_as_lists[i][0][1].xr - track_as_lists[i][0][1].xl
                high_width = track_as_lists[i][len(track_as_lists[i]) - 1][1].xr - \
                             track_as_lists[i][len(track_as_lists[i]) - 1][1].xl

                #if high_width < low_width:
                #   del track_as_lists[i]
                #    size = size - 1
                #    continue
                #else:
                i = i + 1

    toc = time.perf_counter()
    print(f"Time: {toc - tic:0.4f} seconds")
    return tracks


def drow_tracks(tracks, image_color, image_blur, edges):
    tic = time.perf_counter()
    crosses = []
    cell_size = 30
    res = find_cross_by_accum(tracks, edges.shape[1], edges.shape[0], 40)
    for cross_point in res:
        cv2.circle(image_color, cross_point, 10, (255, 0, 0), -1)
    #crosses = find_cross_by_tracks(tracks)
    toc = time.perf_counter()
    print(f"Time find crosses: {toc - tic:0.4f} seconds")
    for cross in crosses:
        cv2.circle(image_color, cross, 1, (0, 0, 255), -1)

    for track_as_lists in tracks:
        if isinstance(track_as_lists, list):
            if len(track_as_lists) > 0:
                for track_as_pairs_list in track_as_lists:
                    track_color = (random.randrange(255), random.randrange(255), random.randrange(255))
                    rail_color = track_color#(random.randrange(255), random.randrange(255), random.randrange(255))
                    for pair in track_as_pairs_list:
                        cv2.circle(image_color, (int(pair[1].center), pair[0]), 1, track_color, -1)
                        cv2.circle(image_color, (int(pair[1].xl), pair[0]), 1, rail_color, -1)
                        cv2.circle(image_color, (int(pair[1].xr), pair[0]), 1, rail_color, -1)
                        cv2.circle(image_blur, (int(pair[1].center), pair[0]), 1, (255, 0, 0), -1)
                        cv2.circle(edges, (int(pair[1].center), pair[0]), 1, (255, 0, 0), -1)


    cv2.imshow("Image color", image_color)
    cv2.imshow("Gauss blur", image_blur)
    cv2.imshow("Canny edges", edges)


if __name__ == '__main__':
    # загружаем изображение и отображаем его
    im_name = "img0002.jpg"
    image = cv2.imread(im_name, 0)
    image_color = cv2.imread(im_name)

    image_blur = cv2.GaussianBlur(image, (0, 0), 1.5)
    # high_thresh, thresh_im = cv2.threshold(image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh

    v = np.median(image_blur)
    sigma = 0.6
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image_blur, lower, upper)
    #cv2.imshow("original", image)
    #cv2.imshow("edges", edges)
    tracks = cross_detect_list(edges)
    drow_tracks(tracks, image_color, image_blur, edges)

    # dilation example
    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(edges, kernel, iterations=1)
    # cv2.imshow("Canny dilation", dilation)

    cv2.waitKey(0)
