import os
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import cv2

from sure_generate_boxes import generate_default_boxes
from build_targets import build_score_offset_targets, build_seg_targets


class IcprDataset(Dataset):
    def __init__(self, root_dir, config, phase='train', image_transform=None, desired_image_shape=(512, 512)):
        self.root_dir = root_dir
        self.config = config
        self.phase = phase
        self.image_transform = image_transform
        assert len(desired_image_shape) == 2
        self.desired_img_height = desired_image_shape[0]
        self.desired_img_width = desired_image_shape[1]

        self.default_boxes = torch.from_numpy(np.array(generate_default_boxes(self.config))).float()

        img_dir = 'image_' + self.phase
        txt_dir = 'txt_' + self.phase
        img_dir = os.path.join(self.root_dir, img_dir)
        txt_dir = os.path.join(self.root_dir, txt_dir)
        assert os.path.exists(txt_dir)
        assert os.path.exists(img_dir)

        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.img_list = os.listdir(img_dir)

    def __getitem__(self, idx):
        # 得到图片文件及相应的文本文件的路径
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        assert os.path.isfile(img_path)

        txt_path = os.path.join(self.txt_dir, img_name[:-4] + '.txt')
        assert os.path.isfile(txt_path)

        # 这里读入图爿，并且对其进行reszie
        img = Image.open(img_path).convert('RGB')
        img_width, img_height = img.size
        img = img.resize((self.desired_img_width, self.desired_img_height), resample=Image.BILINEAR)
        hscale = self.desired_img_height / img_height
        wscale = self.desired_img_width / img_width

        # 读取信息
        with open(txt_path, 'rb') as f:
            lines = f.readlines()

        # [[x1, y1, x2, y2, x3, y3, x4, y4], [], [], []] 这样形式的列表
        # 注意这里的数据集有问题
        quads = []
        for line in lines:
            line = line.decode('utf-8').split(',')[:-1]
            quads.append([float(coord) for coord in line])

        # 需要注意的是，minAreaRect的输入必须是int型
        rects= []
        for quad in quads:
            cnt = np.round(np.reshape(np.array(quad), (4, 2))).astype(np.int)
            rect = cv2.minAreaRect(cnt)
            rect_points = cv2.boxPoints(rect)
            # todo 这里需要对points的顺序做个调整
            # 这里的矩阵需要满足一定的定义 x2 > x1, x3 > x4, y3 > y2, y4 > y1
            j = 0
            while not is_correct_rect(rect_points):
                rect_points = shift_rect_points(rect_points)
                j += 1
                # 如果j==4,说明绕了一圈了，说明这个点的坐标是存在问题的
                if j == 4:
                    print(rect_points)
                    break

            # j == 4说明这个矩阵是有问题的，我们需要把它删除掉，可以简单利用面积先确定下,10是随便确定的
            if j == 4:
                assert compute_area(rect_points) < 10
                continue

            # 再次确定下是正确的矩形
            assert is_correct_rect(rect_points)

            # 把np.array的坐标[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]转换成我们需要的格式
            # [y1, x1, y2, x2, y3, x3, y4, x4]
            # 原来的坐标是整数类型，现在转换成浮点数
            temp = [rect_points[0, 1] * hscale, rect_points[0, 0] * wscale,
                    rect_points[1, 1] * hscale, rect_points[1, 0] * wscale,
                    rect_points[2, 1] * hscale, rect_points[2, 0] * wscale,
                    rect_points[3, 1] * hscale, rect_points[3, 0] * wscale]
            rects.append(temp)

        # for box in truth_other:
        #     for i, coord in enumerate(box):
        #         if i % 2 == 0:
        #             if coord < 0:
        #                 box[i] = 0
        #             elif coord > h-1:
        #                 box[i] = h-1
        #         else:
        #             if coord < 0:
        #                 box[i] = 0
        #             elif coord > w-1:
        #                 box[i] = w-1

        # 生成的这部分应该取决于一张图片中真是的rect的数量的4倍（原图中存在错误的四边形，应该去掉）
        gt_boxes = torch.zeros((len(rects) * 4, 4), requires_grad=False).float()
        gt_corner_type_ids = torch.zeros(len(rects) * 4, requires_grad=False).long()

        #
        for i, rect in enumerate(rects):
            ss1 = math.sqrt(math.pow(rect[1] - rect[3], 2) + math.pow(rect[0] - rect[2], 2))
            ss2 = math.sqrt(math.pow(rect[3] - rect[5], 2) + math.pow(rect[2] - rect[4], 2))
            ss = min(ss1, ss2)

            for j in range(4):
                gt_boxes[i*4+j][0] = rect[j*2] - 0.5 * (ss - 1)
                gt_boxes[i*4+j][1] = rect[j*2 + 1] - 0.5 * (ss - 1)
                gt_boxes[i*4+j][2] = rect[j*2] + 0.5 * (ss - 1)
                gt_boxes[i*4+j][3] = rect[j*2 + 1] + 0.5 * (ss - 1)
                gt_corner_type_ids[i*4+j] = j+1

        print(self.default_boxes.type())
        target_scores, target_offsets, match = \
            build_score_offset_targets(self.default_boxes, gt_boxes, gt_corner_type_ids, self.config)
        target_segs = build_seg_targets(rects, self.config)

        if self.image_transform:
            img = self.image_transform(img)

        return {'image': img, 'target_scores': target_scores, 'target_offsets': target_offsets,
                'target_segs': target_segs, 'match': match}

        # return {'image': img, 'target_scores': target_scores, 'target_offsets': target_offsets, 'match': match}

    def __len__(self):
        return len(self.img_list)


def detection_collate(batch):
    images = []
    target_scores = []
    target_offsets = []
    target_segs = []
    match = []

    for sample in batch:
        print(sample.keys())
        images.append(sample['image'])
        target_scores.append(sample['target_scores'])
        target_offsets.append(sample['target_offsets'])
        target_segs.append(sample['target_segs'])
        match.append(sample['match'])

    #         torch.stack(target_segs, dim=0)
    return torch.stack(images, dim=0), torch.stack(target_scores, dim=0), torch.stack(target_offsets, dim=0), \
           torch.stack(target_segs, dim=0), torch.stack(match, dim=0)


def is_correct_rect(rect):
    """
    :param rect_points: np.array shape [4, 2] [x1, y1], [x2, y2]
    :return:
    """
    return rect[1, 0] > rect[0, 0] and rect[2, 0] > rect[3, 0] and rect[2, 1] > rect[1, 1] and rect[3, 1] > rect[0, 1]


def shift_rect_points(rect):
    temp1 = rect[0]
    temp1 = temp1[np.newaxis, :]
    temp2 = rect[1:4]
    return np.concatenate((temp2, temp1), axis=0)


def compute_area(rect_points):
    vector1 = rect_points[1] - rect_points[0]
    s1 = math.sqrt(math.pow(vector1[0], 2) + math.pow(vector1[1], 2))
    vector2 = rect_points[2] - rect_points[1]
    s2 = math.sqrt(math.pow(vector2[0], 2) + math.pow(vector2[1], 2))
    return s1 * s2


class Config():
    def __init__(self):
        self.IMAGE_SHAPE = (512, 512) # desired height and width
        self.NUM_CORNER_TYPES = 4 # top left, top right, bottom right, bottom left
        self.IOU_THRESHOLD = 0.7 # 在build match中使用
        self.CORNER_TYPE_IDS = (1, 2, 3, 4) # label for top left, top right, bottom right, bottom left
        self.ALL_SCALES = ((4, 8, 6, 10, 12, 16), (20, 24, 28, 32), (36, 40, 44, 48), (56, 64, 72, 80),
                           (88, 96, 104, 112), (124, 136, 148, 160), (184, 208, 232, 256))


if __name__ =='__main__':
    config = Config()
    root_dir = 'd:/dataset/data/'
    phase = 'train'
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = IcprDataset(root_dir, config, phase, transform)
    # data = dataset[0]
    # print(data['truth'])
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=1,
                            collate_fn=detection_collate)
    print(len(dataloader))
    for i,batch in enumerate(dataloader):
        if i == 1:
            break
        imgs, target_scores, target_offsets, target_segs, match = batch
        print(imgs.size(), target_scores.size(), target_offsets.size(), target_segs.size(), match.size())
        
