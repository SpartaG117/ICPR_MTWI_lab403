import torch
import torch.nn.functional as F


def online_hard_example_mining(logits, targets, default_boxes, match, nms_threshold, config):
    """
    这个代码还要负责target_offsets的生成, 这里应当直接生成[N ,4], N是batch中所有图片的的positive

    logits, shape [N, num_all_db, 8]
    targets, shape [N, num_all_db, 8]
    default_boxes: 注意这里的box是在图像域图标里面的
    match: shape [N, num_all_db， 4] 是+1 和 -1标签， 表明一个样本是正样本还是负样本,这部分应该由generate_targets生成
    nms_threshold: 0.7

    config: TRAIN_BOXES_PER_IMAGE, POSITIVE_BOX_RATIO
    这里用到了两个属性值， 一个是每张图片训练的box的数量，一个属性是一张图片中positive box的比例

    返回值应当是一个train_match，shape [batch_size, num_default_boxes，4],
    每个batch_size的match可能不一样，这里主要是说的数量，因为某些可能数量不够，但我们保持一个比例,
    假设返回正样本为1，负样本为-1， 不计算损失函数的为0,利用+1和-1来计算损失函数

    """
    assert logits.size() == targets.size()
    batch_size = logits.size()[0]
    num_default_boxes = logits.size()[1]

    # 这个值不需要计算梯度
    train_match = torch.zeros((batch_size, num_default_boxes * 4), requires_grad=False).long()

    # 把某些维度进行融合，可以方便计算损失函数
    logits = logits.view(-1, 2)
    targets = targets.view(-1, 2)
    targets = torch.nonzero(targets)[:, 1].long()

    # 计算损失函数，这里的值应该是一个 [N * num_all_db * 4]的tensor
    loss = F.cross_entropy(input=logits, target=targets, reduce=False)
    # 这里将损失函数值还原成原来的形状
    # 注意这个每个default box 定义了4个独立的损失函数，代表了4个位置的预测
    # shape [N, num_default_boxes * 4]
    loss = loss.view(batch_size, -1)

    # 对default box进行4倍扩充，四个相邻的box是相同的，代表四个不同位置的
    default_boxes_x4 = torch.zeros((num_default_boxes * 4, 4), requires_grad=False)
    for i in range(4):
        default_boxes_x4[i:num_default_boxes * 4:4] = default_boxes

    # 这里利用极大值抑制进行抑制，返回值是bool,1代表没有倍抑制的，0代表被抑制的，注意这个数量可能随着image变化
    keep_bool_after_nms = []
    for i in range(batch_size):
        loss_scores = loss[i, :]
        keep_bool_after_nms.append(nms(torch.cat((default_boxes_x4, loss_scores.squeeze(dim=1)), dim=1), nms_threshold))
    keep_bool_after_nms = torch.stack(keep_bool_after_nms, dim=0)

    # 这里得到的值是nms之后的正负样本，0表示不参与损失函数计算的box, +1是正样本，-1是负样本
    keep = keep_bool_after_nms * match.view(batch_size, -1)

    for i in range(batch_size):
        # 得到batch中某张image的一个default box keep 1=positive, 0=nertal, -1=negative
        row_keep = keep[i, :]
        # todo 这里可以判断下，加入row_keep全为0的情况
        ixs = torch.nonzero(row_keep != 0).squeeze(dim=1)
        # pn = positive negative
        num_pn_boxes = ixs.size()[0]

        # 得到经过nms之后每张图片中fg和bg的数量，注意这可能少于我们事先设定的阈值，所以这里要进行相关处理
        fg_count = torch.nonzero(row_keep == 1).size()[0]
        bg_count = torch.nonzero(row_keep == -1).size()[0]

        # todo 这部分计算实际数量的代码可能会出现问题，需要检查一下
        train_positive_boxes_per_image = int(config.TRAIN_BOXES_PER_IMAGE * config.POSITIVE_BOX_RATIO)
        actual_fg_count = min(min(fg_count, int(bg_count * config.POSITIVE_BOX_RATIO)), train_positive_boxes_per_image)
        if actual_fg_count == train_positive_boxes_per_image:
            actual_bg_count = config.TRAIN_BOXES_PER_IMAGE - actual_fg_count
        else:
            actual_bg_count = int(actual_fg_count / config.POSITIVE_BOX_RATIO)

        assert (actual_fg_count + actual_bg_count) <= config.TRAIN_BOXES_PER_IMAGE

        loss_scores = loss[i, ixs]
        loss_scores, order = torch.sort(loss_scores, descending=True)

        fc = 0
        bc = 0
        j = 0
        while (fc < actual_fg_count or bc < actual_bg_count) and j < num_pn_boxes:
            index = ixs[order[j]]
            if row_keep[index] == 1 and fc < actual_fg_count:
                train_match[i, index] = 1
                fc += 1

            elif row_keep[index] == -1 and bc < actual_bg_count:
                train_match[i, index] = -1
                bc += 1
            else:
                raise RuntimeError('There must be something wrong')
            j += 1

    return train_match


def nms(boxes_with_scores, nms_threshold):
    """
    非极大值抑制
    """
    num_boxes = boxes_with_scores.size()[0]

    y1 = boxes_with_scores[:, 0]
    x1 = boxes_with_scores[:, 1]
    y2 = boxes_with_scores[:, 2]
    x2 = boxes_with_scores[:, 3]
    scores = boxes_with_scores[:, 4]

    h = y2 - y1 + 1
    w = x2 - x1 + 1
    area = h * w

    scores, order = torch.sort(scores, descending=True)

    keep_bool = torch.ones(num_boxes, requires_grad=False)

    for i in range(num_boxes - 1):
        inds = order[i + 1:num_boxes].data
        h_overlap = torch.max(torch.min(y2[order[i].data], y2[inds]) - torch.max(y1[order[i].data], y1[inds]) + 1, 0)
        w_overlap = torch.max(torch.min(x2[order[i].data], x2[inds]) - torch.max(x1[order[i].data], x1[inds]) + 1, 0)
        area_overlap = h_overlap * w_overlap
        iou = area_overlap / (
                    area[order[i].data] + (y2[inds] - y1[inds] + 1) * (x2[inds] - x1[inds] + 1) - area_overlap)
        supressed_inds = torch.nonzero(iou >= nms_threshold).squeeze(dim=1) + i + 1
        keep_bool[supressed_inds.data] = 0

    return keep_bool
