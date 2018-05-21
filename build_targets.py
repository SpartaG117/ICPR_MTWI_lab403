import torch
import cv2
import numpy as np


def build_score_offset_targets(default_boxes, gt_boxes, gt_corner_type_ids, config):
    """
    为单张图片生成相应的targets

    Params:
        default_boxes: FloatTensor, shape [num_db_boxes, (y1, x1, y2, x2)] 所有的boxes,输入是在图像域坐标
        gt_boxes: FloatTensor, shape [num_gt_boxes, (y1, x1, y2, x2)]真实的点的标记，每个点都是一个正方形的box,
            并且附带这个点的类别信息,假设是在图像域的坐标
        gt_type_ids: LongTensor, shape [num_gt_boxes], 1，2，3，4 top_left, top_right, bottom_right, bottom_left

    Returns: 都是 tensor
        target_scores: LongTensor, shape [num_db_boxes, 8], 每2位必须有一个是1，意味这该box必须是某个corner type box的前景
            或者背景
        target_offsets: FloatTensor, shape, [num_db_boxes, 16], 如果该box是某个corner type box的前景，则按公式计算offset，
            如果是背景，则设置为0
        match: LongTensor, shape [num_db_boxes, 4], 如果与某一个box匹配（iou大于0.7），则为1,否则为-1
    """

    assert len(gt_boxes.size()) == 2, 'Ground truth boxes must have shape [N ,4]'
    assert len(gt_corner_type_ids.size()) == 1, 'Corner type ids must have shape [N]'
    assert gt_boxes.size()[0] == gt_corner_type_ids.size()[0], '{:d} boxes but {:d} corner type ids'.format(
        gt_boxes.size()[0], gt_corner_type_ids.size()[0])

    num_default_boxes = default_boxes.size()[0]
    # 定义我们需要的targets, 这里score默认背景设置为1
    target_scores = torch.zeros((num_default_boxes, config.NUM_CORNER_TYPES * 2), requires_grad=False).long()
    target_scores[:, 0:8:2] = 1
    target_offsets = torch.zeros((num_default_boxes, config.NUM_CORNER_TYPES * 4), requires_grad=False).float()

    # positive negative match
    match = -torch.ones((num_default_boxes, config.NUM_CORNER_TYPES), requires_grad=False).long()

    # todo 检查一下计算overlaps这个函数正常吗
    # 两者的输入都是在图象域坐标，都是矩形
    overlaps = compute_overlaps(default_boxes, gt_boxes)
    # 计算default box 只有与gt box相交iou大于一定的阈值的设置为1
    keep_bool = (overlaps >= config.IOU_THRESHOLD_FOR_DEFAULT_AND_GT_BOX)
    # for debug
    # print('default box 与 gt box iou大于阈值的框的数量')
    # print(torch.nonzero(keep_bool).size()[0])

    for i in range(num_default_boxes):
        row_keep_bool = keep_bool[i]
        ixs = torch.nonzero(row_keep_bool)

        # 这里应该考虑ixs为空的情况，这种情况说明该default box与任何gt_box的iou小于0.7
        # 则对应的scores和offsets和match都分别取了默认值，0，0，-1
        if ixs.size()[0] == 0:
            continue

        # 这个ixs就是一个gt_boxes中的一个编号
        ixs = ixs.squeeze(dim=1)
        row_keep_overlaps = overlaps[i, ixs]
        row_keep_corner_type_ids = gt_corner_type_ids[ixs]

        for corner_type_id in config.CORNER_TYPE_IDS:
            iys = torch.nonzero(row_keep_corner_type_ids == corner_type_id)
            # 这个判断表明该box没有这个corner_id的gt与之匹配
            # 这不执行此次循环，所有的为默认值
            if iys.size()[0] == 0:
                continue

            # 表明该box可以得到此corner_id的box，得到相应的gt_box
            elif iys.size()[0] == 1:
                iys = iys.squeeze(dim=1)
                gt_box_id = ixs[iys[0]]

            # 表明该box与两个及以上同corner_id的box都匹配，这里只能选择一个，那就选择overlap最大的那一个
            # 并且得到其相应的gt_box_id
            else:
                iys = iys.squeeze(dim=1)
                row_keep_specific_overlaps = row_keep_overlaps[iys]
                _, order = torch.sort(row_keep_specific_overlaps, descending=True)
                gt_box_id = ixs[iys[order[0]]]

            assert gt_corner_type_ids[gt_box_id].item() == corner_type_id, 'Something wrong with the code'

            # 编码方式【，，，，，，】一共有8位，每两位代表着某个corner的类型，前一位是不是的概率，后面一位是存在的概率
            # offsets 每四位对应相应的corner_id 的box的系数
            target_scores[i, (corner_type_id - 1) * 2] = 0
            target_scores[i, (corner_type_id - 1) * 2 + 1] = 1
            offsets = compute_offsets(default_boxes[i], gt_boxes[gt_box_id])
            target_offsets[i, (corner_type_id - 1) * 4:corner_type_id * 4] = offsets
            match[i, corner_type_id - 1] = 1

    return target_scores, target_offsets, match


def build_seg_targets(rects, config):
    """
    为单张图片生成相应的mask
    所以这部分有个关键函数，利用fillConvexPoly画多边形机及其内部的点，注意其输入必须是整数坐标，而且是[x, y]形式

    Params:
        rects: list of lists, [[y1, x1, y2, x2, y3, x3, y4, x4], [], [], []]
        这里希望输入的坐标是在图像域，连续的4个坐标刚好代表了一个四边形，第一个点为左上角，然后是顺时针方向，依此右上，右下，左下

    Returns: tensor
        target_segs: LongTensor, shape, [4, H, W], 第一个channel是左上的mask，第二个是右上的mask，第三个是右下，第四个左下
    """

    assert len(rects) > 0
    num_rects = len(rects)
    rects = torch.tensor(rects).float()

    assert len(rects.size()) == 2
    assert rects.size()[0] == num_rects
    assert rects.size()[1] == 8

    target_segs = np.zeros((4, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]))

    def swap_xy(point_yx):
        point_xy = torch.zeros(2, requires_grad=False).float()
        point_xy[0] = point_yx[1]
        point_xy[1] = point_yx[0]
        return point_xy

    for i_rect in range(num_rects):
        # 得到四个点的坐标
        a = swap_xy(rects[i_rect, 0:2])
        b = swap_xy(rects[i_rect, 2:4])
        c = swap_xy(rects[i_rect, 4:6])
        d = swap_xy(rects[i_rect, 6:8])

        # 把这四个点进行切分，得到中心坐标，即把这个多半行划分为4份
        mid_ab = (a + b) / 2
        mid_bc = (b + c) / 2
        mid_cd = (c + d) / 2
        mid_da = (d + a) / 2
        # mid_rect = (mid_bc + mid_da) / 2
        mid_rect = (mid_ab + mid_cd) / 2

        part_top_left = torch.stack((a, mid_ab, mid_rect, mid_da), dim=0)
        part_top_right = torch.stack((mid_ab, b, mid_bc, mid_rect), dim=0)
        part_bottom_right = torch.stack((mid_rect, mid_bc, c, mid_cd), dim=0)
        part_bottom_left = torch.stack((mid_da, mid_rect, mid_cd, d), dim=0)
        parts = [part_top_left, part_top_right, part_bottom_right, part_bottom_left]
        for j_part in range(len(parts)):
            cv2.fillConvexPoly(target_segs[j_part], torch.round(parts[j_part]).int().numpy(), 1)
    return torch.from_numpy(target_segs).long().requires_grad_(False)


def compute_overlaps(boxes1, boxes2):
    """
    两个输入都是 floattensor，不需要梯度， boxes1: shape [N1 ,4],  boxes2: shape [N2 ,4],
    这里输入的坐标是在图像域里面的

    返回值: overlaps floattensor, 不需要梯度， shape [N1, N2]
    """
    assert isinstance(boxes1, torch.FloatTensor)
    assert isinstance(boxes2, torch.FloatTensor)

    y1 = boxes1[:, 0]
    x1 = boxes1[:, 1]
    y2 = boxes1[:, 2]
    x2 = boxes1[:, 3]
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    area = h * w
    overlaps = torch.zeros((boxes1.size()[0], boxes2.size()[0]), requires_grad=False).float()

    for i in range(boxes2.size()[0]):
        # 计算某个boxes2与boxes1中所有box的交集
        h_overlap = torch.max(torch.min(y2, boxes2[i, 2]) - torch.max(y1, boxes2[i, 0]) + 1, torch.tensor(0.0).float())
        w_overlap = torch.max(torch.min(x2, boxes2[i, 3]) - torch.max(x1, boxes2[i, 1]) + 1, torch.tensor(0.0).float())
        area_overlap = h_overlap * w_overlap
        iou = area_overlap / \
              (area + (boxes2[i, 2] - boxes2[i, 0] + 1) * (boxes2[i, 3] - boxes2[i, 1] + 1) - area_overlap)
        overlaps[:, i] = iou

    return overlaps


def compute_offsets(box, gt_box):
    """
    计算box到gt_box的一个offset
    Params:
        box: FloatTensor, shape [4]
        gt_box: FloatTensor, shape [4]

    Returns:
        offsets: FloatTensor, shape [4]
    """

    assert len(box.size()) == 1 and len(gt_box.size()) == 1
    assert box.size()[0] == 4 and gt_box.size()[0] == 4
    assert isinstance(box, torch.FloatTensor)
    assert isinstance(gt_box, torch.FloatTensor)

    offsets = torch.zeros(4, requires_grad=False).float()
    box_y_ctr, box_x_ctr, box_h, box_w = get_ctrs_h_w(box)
    gt_box_y_ctr, gt_box_x_ctr, gt_box_h, gt_box_w = get_ctrs_h_w(gt_box)

    offsets[0] = (gt_box_y_ctr - box_y_ctr) / box_h
    offsets[1] = (gt_box_x_ctr - box_x_ctr) / box_w
    offsets[2] = torch.log(gt_box_h / box_h)
    offsets[3] = torch.log(gt_box_w / box_w)
    return offsets


def get_ctrs_h_w(box):
    """
    给定box, [y1, x1, y2, x2]得到中心坐标和高宽

    Params:
        box: FloatTensor, shape [4]

    Returns:
        [y_ctr, x_ctr, h, w]
    """
    assert len(box.size()) == 1 and box.size()[0] == 4

    y1 = box[0]
    x1 = box[1]
    y2 = box[2]
    x2 = box[3]
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    y_ctr = y1 + 0.5 * (h - 1)
    x_ctr = x1 + 0.5 * (w - 1)
    return y_ctr, x_ctr, h, w
