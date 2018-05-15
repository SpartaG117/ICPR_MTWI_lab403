import torch


def build_score_offset_targets(default_boxes, gt_boxes, gt_corner_type_ids, config):
    """
        default_boxes: 所有的boxes,假设是输入是在图像域坐标
        gt_boxes: 真实的点的标记，每个点都是一个正方形的box, 并且附带这个点的类别信息,假设是在图像域的坐标 这里的两个应该都要是tensor
        gt_type_ids: 1，2，3，4 top_left, top_right, bottom_right, bottom_left
    """
    # todo 这里应该确保num_default_boxes是有值的
    num_default_boxes = default_boxes.size()[0]
    # 定义我们需要的targets
    target_scores = torch.zeros((num_default_boxes, config.NUM_CORNER_TYPES * 2), requires_grad=False).long()
    target_offsets = torch.zeros((num_default_boxes, config.NUM_CORNER_TYPES * 4), requires_grad=False)

    # positive negative match
    match = -torch.ones((num_default_boxes, 4), requires_grad=False).long()

    overlaps = compute_overlaps(default_boxes, gt_boxes)
    # 计算default box 只有与gt box相交iou大于一定的阈值的设置为1
    keep_bool = (overlaps >= config.IOU_THRESHOLD)

    # todo 这里应该可以减少时间复杂度
    for i in range(num_default_boxes):
        row_keep_bool = keep_bool[i]
        ixs = torch.nonzero(row_keep_bool)

        # 这里应该考虑ixs为空的情况，这种情况说明该default box不属于任何gtbox
        if ixs.size()[0] == 0:
            continue
        ixs = ixs.squeeze(dim=1)
        row_keep_overlaps = overlaps[i, ixs]
        row_keep_corner_type_ids = gt_corner_type_ids[ixs]

        for corner_type_id in config.CORNER_TYPE_IDS:
            iys = torch.nonzero(row_keep_corner_type_ids == corner_type_id)
            # 这个判断表明该box没有corner_id的gt与之匹配
            if iys.size()[0] == 0:
                continue

            elif iys.size()[0] == 1:
                iys = iys.squeeze(dim=1)
                gt_box_id = ixs[iys[0]]

            # 》= 2
            else:
                row_keep_specific_overlaps = row_keep_overlaps[iys]
                _, order = torch.sort(row_keep_specific_overlaps, descending=True)
                gt_box_id = ixs[iys[order[0]]]

            # 编码方式【，，，，，，】一共有8位，每两位代表着某个corner的类型，前一位是不是的概率，后面一位是存在的概率
            target_scores[i, (corner_type_id - 1) * 2 + 1] = 1
            offsets = compute_offsets(default_boxes[i], gt_boxes[gt_box_id])
            target_offsets[i, (corner_type_id - 1) * 4:corner_type_id * 4] = offsets
            match[i, corner_type_id - 1] = 1

    return target_scores, target_offsets, match


def build_seg_targets(zuobiao, config):
    """
    这里希望输入的坐标是整数类型，连续的4个刚好代表了一个四边形，返回值是一个4*h*w的mask，0，1取值
    所以这部分有个关键函数，用来判断一个点是否在一个凸4边形内部
    """
    target_segs = np.zeros(4, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    assert len(zuobiao) % 4 == 0
    num_boxes = int(len(zuobiao) / 4)

    for x in range(config.IMAGE_SHAPE[0]):
        for y in range(config.IMAGE_SHAPE[1]):
            for i_box in range(num_boxes):
                # 得到四个点的坐标
                convex_quadrilateral_points = torch.FloatTensor([zuobiao[i_box * 4:(i_box + 1) * 4]])
                # 把这四个点进行切分，得到中心坐标，即把这个多半行划分为4份
                a, b, c, d = convex_quadrilateral_points[0:4]
                mid_ab = (a + b) / 2
                mid_bc = (b + c) / 2
                mid_cd = (c + d) / 2
                mid_da = (d + a) / 2
                # mid_convex_quadrilateral = (mid_bc + mid_da) / 2
                mid_convex_quadrilateral = (mid_ab + mid_cd) / 2

                part_top_left = torch.stack([a, mid_ab, mid_convex_quadrilateral, mid_da], dim=0)
                part_top_right = torch.stack([mid_ab, b, mid_bc, mid_convex_quadrilateral], dim=0)
                part_bottom_right = torch.stack([mid_convex_quadrilateral, mid_bc, c, mid_cd], dim=0)
                part_bottom_left = torch.stack([mid_da, mid_convex_quadrilateral, mid_cd, d], dim=0)
                point = torch.FloatTensor([x, y])

                parts = [part_top_left, part_top_right, part_bottom_right, part_bottom_left]
                for j_part in range(len(parts)):
                    if (is_in_convex_quadrilateral(parts[j_part], point)):
                        target_segs[j_part, x, y] = 1
        return target_segs.long()


def is_in_convex_quadrilateral(convex_quadrilateral_points, point):
    """
    用来判断一个点是否在凸四边形内部，仅仅只内部，不包括边界上的点，如过在内部，返回true,不在内部，则返回false
    convex_quadrilateral_points是一个4*2的floattensor
    point是一个2大小的tensor
    """
    # 先得到凸四边形的四个顶点, 这里的定点是在图像域中的坐标
    assert convex_quadrilateral_points.size()[0] == 4
    a, b, c, d = convex_quadrilateral_points[0:4]
    x = point

    # 计算我们需要的一些向量
    ab = b - a
    ax = x - a
    bc = c - b
    bx = x - b
    cd = d - c
    cx = x - c
    da = a - d
    dx = x - d

    return ((ab[0] * ax[1] - ax[0] * ab[1]) > 0) and ((bc[0] * bx[1] - bx[0] * bc[1]) > 0) \
           and ((cd[0] * cx[1] - cx[0] * cd[1]) > 0) and ((da[0] * dx[1] - dx[0] * da[1]) > 0)


def compute_overlaps(boxes1, boxes2):
    """
    两个输入都是tensor，boxes1: shape [N1 ,4],  boxes2: shape [N2 ,4],
    这里输入的坐标是在图像域里面的
    """
    y1 = boxes1[:, 0]
    x1 = boxes1[:, 1]
    y2 = boxes1[:, 2]
    x2 = boxes1[:, 3]
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    area = h * w
    overlaps = torch.zeros(boxes1.size()[0], boxes2.size()[0])

    for i in range(boxes2.size()[0]):
        # 计算某个boxes2与boxes1中所有box的交集
        h_overlap = torch.max(torch.min(y2, boxes2[i, 2]) - torch.max(y1, boxes2[i, 0]) + 1, torch.FloatTensor([0]))
        w_overlap = torch.max(torch.min(x2, boxes2[i, 3]) - torch.max(x1, boxes2[i, 1]) + 1, torch.FloatTensor([0]))
        area_overlap = h_overlap * w_overlap
        iou = area_overlap / (
                    area + (boxes2[i, 2] - boxes2[i, 0] + 1) * (boxes2[i, 3] - boxes2[i, 1] + 1) - area_overlap)
        overlaps[:, i] = iou

    return overlaps


def compute_offsets(box, gt_box):
    """
    这里的两个输入都是一个box, tensor, shape [4]
    """
    offsets = torch.zeros(4)
    box_y_ctr, box_x_ctr, box_h, box_w = get_ctrs_h_w(box)
    gt_box_y_ctr, gt_box_x_ctr, gt_box_h, gt_box_w = get_ctrs_h_w(gt_box)

    offsets[0] = (gt_box_y_ctr - box_y_ctr) / box_h
    offsets[1] = (gt_box_x_ctr - box_x_ctr) / box_w
    offsets[2] = torch.log(gt_box_h / box_h)
    offsets[3] = torch.log(gt_box_w / box_w)
    return offsets


def get_ctrs_h_w(box):
    # 确保只有一个box，而且这个box是在图像域里面
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
