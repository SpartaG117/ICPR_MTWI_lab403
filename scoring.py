import torch
import math
import cv2
import numpy as np

#def scoring(candidates, score_map):

    #for boxes in candidates:



def generate_bin(box):
    """
    :param box: [coordinates(8)]
    :return:  bin, rect_bins
    """
    bins = []
    rec_bins = []
    g = 2
    x_l = (box[0] + box[2])/2
    y_l = (box[1] + box[3])/2
    x_t = (box[0] + box[6])/2
    y_t = (box[1] + box[7])/2
    x_r = (box[4] + box[6])/2
    y_r = (box[5] + box[7])/2
    x_b = (box[2] + box[4])/2
    y_b = (box[3] + box[5])/2
    x_m = (x_l + x_r)/2
    y_m = (y_t + y_b)/2

    bins.append([box[0], box[1], x_l, y_l, x_m, y_m, x_t, y_t])
    bins.append([x_t, y_t, x_m, y_m, x_r, y_r, box[6], box[7]])
    bins.append([x_l, y_l, box[2], box[3], x_b, y_b, x_m, y_m])
    bins.append([x_m, y_m, x_b, y_b, box[4], box[5], x_r, y_r])

    bin0 = np.array([bins[0][0].floor])
    for bin in bins:
        rec_bins.append([[bin[0].floor(), bin[1].floor()],
                         [bin[2].floor(), bin[3].ceil()],
                         [bin[4].ceil(), bin[5].ceil()],
                         [bin[6].ceil(), bin[7].floor()]])

    rec_bins = np.array(rec_bins)
    rect_bins = []
    for rec in rec_bins:
        rect = cv2.minAreaRect(rec)
        rect_points = cv2.boxPoints(rect)
        rect_bin = []
        for i in range(rect_points.shape[0]):
            for j in range(2):
                rect_bin.append(rect_points[i][j])
        rect_bins.append(rect_bin)

    return np.array(bins), np.array(rect_bins).round().astype(np.int32)


if __name__ == '__main__':
    a = [torch.tensor(10.2),
         torch.tensor(10.2),
         torch.tensor(18.3),
         torch.tensor(18.3),
         torch.tensor(23.4),
         torch.tensor(13.2),
         torch.tensor(15.),
         torch.tensor(4.8)]
    print(a)
    b,c = generate_bin(a)
    print(b)
    print(c)