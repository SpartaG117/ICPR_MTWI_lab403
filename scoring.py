import torch
import numpy as np

def scoring(candidates, score_map, threshold=0.6):

    batch_size = len(candidates)
    final_boxes = []
    for i in range(batch_size):
        final_box = []
        for box in candidates[i]:
            bins = generate_bin(box)
            sum_score = 0
            for j, bin in enumerate(bins):
                trans_bin, rotate_mat = trans_box_csys(bin)
                score = roi_score(trans_bin, rotate_mat, score_map[j])
                sum_score += score
            sum_score = sum_score/4
            if sum_score > threshold:
                final_box.append(box)
        final_boxes.append(final_box)
    return final_boxes




def roi_score(trans_bin, rotate_mat, score_map):
    num_pixel = 0
    score = 0
    x_min = int(trans_bin[0][0])
    x_max = int(trans_bin[3][0] + 1)
    y_min = int(trans_bin[0][1])
    y_max = int(trans_bin[1][1] +1)
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            coord = torch.tensor([i,j], dtype=torch.float)
            coord_t = torch.matmul(coord,rotate_mat).round()
            x = int(coord_t[0].item())
            y = int(coord_t[0].item())
            score += score_map[y][x]
            num_pixel += 1
    return score/num_pixel


def trans_box_csys(box):

    top_l = torch.tensor([box[0], box[1]])
    bottom_l = torch.tensor([box[2], box[3]])
    bottom_r = torch.tensor([box[4], box[5]])
    top_r = torch.tensor([box[6], box[7]])

    unit_v = torch.tensor([1., 0.])
    top_v = top_r - top_l
    cos = (top_v * unit_v).sum() / top_v.pow(2).sum().sqrt() / unit_v.pow(2).sum().sqrt()
    sin = (1 - cos.pow(2)).sqrt()
    rotate_mat = torch.tensor([[cos, sin], [-sin, cos]])

    top_lt = torch.matmul(top_l, rotate_mat)
    top_rt = torch.matmul(top_r, rotate_mat)
    bottom_lt = torch.matmul(bottom_l, rotate_mat)
    bottom_rt = torch.matmul(bottom_r, rotate_mat)
    box_t = [top_lt.round(), bottom_lt.round(), bottom_rt.round(), top_rt.round()]
    return box_t, rotate_mat.inverse()




def generate_bin(box):
    """
    :param box: [coordinates(8)] coordinates is torch.tensor and always scalar
    :return:  bin: numpy.array[4,8], rect_bins: numpy.array[4,8]
    """
    bins = []
    rec_bins = []

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

    return torch.from_numpy(np.array(bins).round())


"""
    for bin in bins:
        rec_bins.append([[bin[0].round(), bin[1].round()],
                         [bin[2].round(), bin[3].round()],
                         [bin[4].round(), bin[5].round()],
                         [bin[6].round(), bin[7].round()]])

    rec_bins = np.array(rec_bins)
    rect_bins = []
    for rec in rec_bins:
        rect = cv2.minAreaRect(rec)
        rect_points = cv2.boxPoints(rect)
        rect_points = to_std_seq(rect_points)
        rect_bin = []
        for i in range(rect_points.shape[0]):
            for j in range(2):
                rect_bin.append(rect_points[i][j])
        rect_bins.append(rect_bin)
"""

def to_std_seq(rect_points):
    for i in range(4):
        top_l = rect_points[i]

        index = i + 1
        if index > 3:
            index = 0
        top_r = rect_points[index]

        index += 1
        if index > 3:
            index = 0
        bottom_r = rect_points[index]

        index += 1
        if index > 3:
            index = 0
        bottom_l = rect_points[index]

        if top_l[0] < top_r[0] and bottom_l[0] < bottom_r[0] \
                and top_l[1] < bottom_l[1] and top_r[1] < bottom_r[1] :
            return np.array([top_l, bottom_l, bottom_r, top_r])
    print(rect_points)
    raise RuntimeError("rect_points must belong to a rectangle")

if __name__ == '__main__':

    a = [torch.tensor(10.2),
         torch.tensor(10.2),
         torch.tensor(18.3),
         torch.tensor(18.3),
         torch.tensor(23.4),
         torch.tensor(13.2),
         torch.tensor(15.),
         torch.tensor(4.8)]
    box= generate_bin(a)
    box = box[0]
    print(box)
    box_t, rotate_mat = trans_box_csys(box)
    print(box_t)
    score_map = torch.randint(0,2,(300,300))
    print(roi_score(box_t, rotate_mat, score_map))


    """
    d = np.array([[10.2,10.2],[15,4.8],[23.4,13.2],[18.3,18.3]])
    print(to_std_seq(d))
    """