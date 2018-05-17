import torch
import itertools


def Sampling(corners, corner_type, ss_threshold = 5, width=300, height=300):
    """
    :param corners: [batch_size, corner_id, coordinates(x,y,ss,ss)]
    :param corner_type: [batch_size, corner_id, corner_type]
                        corner_type => 1(top-left),2(bottom-left)
                                       3(bottom-right),4(top-right)
    :return: candidates: [batch_size, num_boxes, coordinates(8)]
    """
    assert corners.size(0) == corner_type.size(0)

    candidates=[]
    for i in range(corners.size(0)):
        corner_1 = []
        corner_2 = []
        corner_3 = []
        corner_4 = []
        pairs_1=[]
        pairs_2=[]
        pairs_3=[]
        pairs_4=[]
        for corner_id in range(corners.size(1)):
            if corner_type[i,corner_id,0] ==1:
                corner_1.append(corners[i,corner_id])
            elif corner_type[i,corner_id,0] ==2:
                corner_2.append(corners[i,corner_id])
            elif corner_type[i,corner_id,0] ==3:
                corner_3.append(corners[i,corner_id])
            elif corner_type[i,corner_id,0] ==4:
                corner_4.append(corners[i,corner_id])
            else:
                raise ValueError("The type of corner must be 1 or 2 or 3 or 4")

        # (top-left,top-right)
        for pair in itertools.product(corner_1,corner_4):
            pairs_1.append(pair)
        # (top-right, bottom-right)
        for pair in itertools.product(corner_4,corner_3):
            pairs_2.append(pair)
        # (bottom-left, bottom-right)
        for pair in itertools.product(corner_2,corner_3):
            pairs_3.append(pair)
        # (top-left, bottom-left)
        for pair in itertools.product(corner_1,corner_2):
            pairs_4.appned(pair)

        boxes=[]

        for pair in pairs_1:
            top_l, top_r = pair
            if top_l[2] > top_r[2]:
                ss_min = top_r[2]
                ss_max = top_l[2]
            else:
                ss_min = top_l[2]
                ss_max = top_r[2]
            # three rules
            if top_l[0] >= top_r[0]:
                continue
            elif ss_max <= ss_threshold:
                continue
            elif ss_max/ss_min <= 1.5:
                continue
            # construct rotated rectangle
            unit_v = torch.tensor([1,0],dtype=torch.float)
            v = top_r[0:2] - top_l[0:2]
            cos = torch.sum(v * unit_v) / v.pow(2).sum().sqrt()
            sin = (1 - cos.pow(2)).sqrt()
            rotate_mat = torch.tensor([[cos, sin],[-sin, cos]])
            top_l_r = torch.matmul(top_l,rotate_mat)
            top_r_r = torch.matmul(top_r,rotate_mat)
            top_l_r[1] += ss_min
            top_r_r[1] += ss_min
            bottom_l = torch.matmul(top_l_r,rotate_mat.inverse())
            bottom_r = torch.matmul(top_r_r,rotate_mat.inverse())

            box =[top_l[0], top_l[1], bottom_l[0], bottom_l[1],
                  bottom_r[0], bottom_r[1], top_r[0], top_r[1]]

            # TODO: adjust algorithm
            flag = 0
            for coord in box:
                if coord > width or coord > height or coord < 0:
                    flag = 1
            if flag == 1:
                continue
            boxes.append(box)

        for pair in pairs_2:
            top_r, bottom_r = pair
            if bottom_r[2] > top_r[2]:
                ss_min = top_r[2]
                ss_max = bottom_r[2]
            else:
                ss_min = bottom_r[2]
                ss_max = top_r[2]
            # three rules
            if top_r[1] >= bottom_r[1]:
                continue
            elif ss_max <= ss_threshold:
                continue
            elif ss_max/ss_min <= 1.5:
                continue
            # construct rotated rectangle
            unit_v = torch.tensor([0,1],dtype=torch.float)
            v = bottom_r[0:2] - top_r[0:2]
            cos = torch.sum(v * unit_v) / v.pow(2).sum().sqrt()
            sin = (1 - cos.pow(2)).sqrt()
            rotate_mat = torch.tensor([[cos, sin],[-sin, cos]])
            bottom_r_r = torch.matmul(bottom_r,rotate_mat)
            top_r_r = torch.matmul(top_r,rotate_mat)
            bottom_r_r[0] -= ss_min
            top_r_r[0] -= ss_min
            bottom_l = torch.matmul(bottom_r_r,rotate_mat.inverse())
            top_l = torch.matmul(top_r_r,rotate_mat.inverse())

            box =[top_l[0], top_l[1], bottom_l[0], bottom_l[1],
                  bottom_r[0], bottom_r[1], top_r[0], top_r[1]]

            # TODO: adjust algorithm
            flag = 0
            for coord in box:
                if coord > width or coord > height or coord < 0:
                    flag = 1
            if flag == 1:
                continue
            boxes.append(box)

        for pair in pairs_3:
            bottom_l, bottom_r = pair
            if bottom_l[2] > bottom_r[2]:
                ss_min = bottom_r[2]
                ss_max = bottom_l[2]
            else:
                ss_min = bottom_l[2]
                ss_max = bottom_r[2]
            # three rules
            if bottom_l[0] >= bottom_r[0]:
                continue
            elif ss_max <= ss_threshold:
                continue
            elif ss_max/ss_min <= 1.5:
                continue
            # construct rotated rectangle
            unit_v = torch.tensor([1,0],dtype=torch.float)
            v = bottom_r[0:2] - bottom_l[0:2]
            cos = torch.sum(v * unit_v) / v.pow(2).sum().sqrt()
            sin = (1 - cos.pow(2)).sqrt()
            rotate_mat = torch.tensor([[cos, sin],[-sin, cos]])
            bottom_l_r = torch.matmul(bottom_l,rotate_mat)
            bottom_r_r = torch.matmul(bottom_r,rotate_mat)
            bottom_l_r[1] -= ss_min
            bottom_r_r[1] -= ss_min
            top_l = torch.matmul(bottom_l_r,rotate_mat.inverse())
            top_r = torch.matmul(bottom_r_r,rotate_mat.inverse())

            #TODO: inspect corners out of bounds

            box =[top_l[0], top_l[1], bottom_l[0], bottom_l[1],
                  bottom_r[0], bottom_r[1], top_r[0], top_r[1]]

            # TODO: adjust algorithm
            flag = 0
            for coord in box:
                if coord > width or coord > height or coord < 0:
                    flag = 1
            if flag == 1:
                continue
            boxes.append(box)

        for pair in pairs_4:
            top_l, bottom_l = pair
            if top_l[2] > bottom_l[2]:
                ss_min = bottom_l[2]
                ss_max = top_l[2]
            else:
                ss_min = top_l[2]
                ss_max = bottom_l[2]
            # three rules
            if top_l[1] >= bottom_l[1]:
                continue
            elif ss_max <= ss_threshold:
                continue
            elif ss_max/ss_min <= 1.5:
                continue
            # construct rotated rectangle
            unit_v = torch.tensor([0,1],dtype=torch.float)
            v = bottom_l[0:2] - top_l[0:2]
            cos = torch.sum(v * unit_v) / v.pow(2).sum().sqrt() / unit_v.pow(2).sum().sqrt()
            sin = (1 - cos.pow(2)).sqrt()
            rotate_mat = torch.tensor([[cos, sin],[-sin, cos]])
            bottom_l_r = torch.matmul(bottom_l,rotate_mat)
            top_l_r = torch.matmul(top_l,rotate_mat)
            bottom_l_r[0] += ss_min
            top_l_r[0] += ss_min
            bottom_r = torch.matmul(bottom_l_r,rotate_mat.inverse())
            top_r = torch.matmul(top_l_r,rotate_mat.inverse())

            #TODO: inspect corners out of bounds

            box =[top_l[0], top_l[1], bottom_l[0], bottom_l[1],
                  bottom_r[0], bottom_r[1], top_r[0], top_r[1]]

            # TODO: adjust algorithm
            flag = 0
            for coord in box:
                if coord > width or coord > height or coord < 0:
                    flag = 1
            if flag == 1:
                continue
            boxes.append(box)

        candidates.append(boxes)

    return candidates


