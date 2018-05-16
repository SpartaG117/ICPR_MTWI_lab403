import os
import math

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class IcprDataset(Dataset):

    def __init__(self, root_dir, phase='train', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform

        img_dir = 'image_' + self.phase
        txt_dir = 'txt_' + self.phase
        img_dir = os.path.join(self.root_dir, img_dir)
        txt_dir = os.path.join(self.root_dir, txt_dir)
        assert os.path.exists(txt_dir)
        assert os.path.exists(img_dir)

        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.txt_list = os.listdir(txt_dir)

    def __getitem__(self, idx):
        txt_name = self.txt_list[idx]
        txt_path = os.path.join(self.txt_dir, txt_name)
        assert os.path.isfile(txt_path)

        img_path = os.path.join(self.img_dir, txt_name[:-4] + '.jpg')
        assert os.path.isfile(img_path)

        with open(txt_path, 'rb') as f:
            lines = f.readlines()
        num_txt_lines = len(lines)

        truth = []
        for i in range(num_txt_lines):
            line = lines[i].decode('utf-8').split(',')[:-1]
            truth.append([float(coord) for coord in line])

        print(img_path)
        print(truth)


        # 这里吧truth[]存储的是原来的坐标格式[x1, y1, x2, y2, x3, y3]，而且是以逆时针存储的，
        # 现在我们改一改[y1, x1, y2, x2, y3, x3, y4, x4]以顺时针形式存储
        truth_other = []
        for i in range(len(truth)):
            temp = [truth[i][1], truth[i][0], truth[i][7], truth[i][6], truth[i][5], truth[i][4], truth[i][3], truth[i][2]]
            truth_other.append(temp)

        # 这里的多边形需要满足一定的定义 x2 > x1, x3 > x4, y3 > y2, y4 > y1
        for i in range(len(truth_other)):
            box = truth_other[i]
            assert box[3] > box[1] and box[5] > box[7] and box[6] > box[0] and box[4] > box[2]
        print(truth_other)

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        for box in truth_other:
            for i, coord in enumerate(box):
                if i % 2 == 0:
                    if coord < 0:
                        box[i] = 0
                    elif coord > h-1:
                        box[i] = h-1
                else:
                    if coord < 0:
                        box[i] = 0
                    elif coord > w-1:
                        box[i] = w-1

        gt_corners = torch.zeros([num_txt_lines * 4, 5], dtype=torch.float)

        # 注意这里的box可能不是矩阵，是4边形，我们选取边最短的那一个
        for i, box in enumerate(truth_other):
            ss1 = math.sqrt(math.pow((box[1] - box[3]), 2) + math.pow((box[0] - box[2]), 2))
            ss2 = math.sqrt(math.pow((box[3] - box[5]), 2) + math.pow((box[2] - box[4]), 2))
            ss3 = math.sqrt(math.pow((box[5] - box[7]), 2) + math.pow((box[4] - box[6]), 2))
            ss4 = math.sqrt(math.pow((box[7] - box[1]), 2) + math.pow((box[6] - box[0]), 2))
            ss = min(min(min(ss1, ss2), ss3), ss4)

            for j in range(4):
                gt_corners[i*4+j][0] = box[j*2] - 0.5 * (ss - 1)
                gt_corners[i*4+j][1] = box[j*2 + 1] - 0.5 * (ss - 1)
                gt_corners[i*4+j][2] = box[j*2] + 0.5 * (ss - 1)
                gt_corners[i*4+j][3] = box[j*2 + 1] + 0.5 * (ss - 1)
                gt_corners[i*4+j][4] = j+1

            # if ss1 > ss2:
            #     ss = ss2
            # else:
            #     ss = ss1

            # for j in range(4):
            #     gt_corners[i*4+j][0] = box[j*2+1] - ss/2
            #     gt_corners[i*4+j][1] = box[j*2] - ss/2
            #     gt_corners[i*4+j][2] = box[j*2+1] + ss/2
            #     gt_corners[i*4+j][3] = box[j*2] + ss/2
            #     gt_corners[i*4+j][4] = j+1

        if self.transform:
            img = self.transform(img)
        else:
            totensor = transforms.ToTensor()
            img = totensor(img)
        return {"image":img, "truth":gt_corners}

    def __len__(self):
        return len(self.txt_list)


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample['image'])
        targets.append(sample['truth'])
    return torch.stack(imgs, 0), targets


if __name__ =='__main__':
    root_dir = 'd:/dataset/data/'
    phase = 'train'
    transform = transforms.Compose([transforms.Resize((300,300)),
                                    transforms.ToTensor()])
    dataset = IcprDataset(root_dir, phase, transform)
    # data = dataset[0]
    # print(data['truth'])
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=detection_collate)
    for i,batch in enumerate(dataloader):
        imgs, targets = batch
        print(i)
