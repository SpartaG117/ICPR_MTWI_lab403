import torch
import math
import os
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms

class IcprDataset(Dataset):

    def __init__(self, root_dir, phase = 'train', transform = None):
        self.data_dir = root_dir
        self.phase = phase
        self.transform = transform

        txt_dir = 'txt_' + self.phase
        img_dir = 'image_' + self.phase
        txt_dir = os.path.join(root_dir, txt_dir)
        img_dir = os.path.join(root_dir, img_dir)
        assert os.path.exists(txt_dir) == True
        assert os.path.exists(img_dir) == True

        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.txt_list = os.listdir(txt_dir)

    def __getitem__(self, idx):
        txt_name = self.txt_list[idx]
        txt_path = os.path.join(self.txt_dir, txt_name)
        assert os.path.isfile(txt_path) == True
        img_path = os.path.join(self.img_dir, txt_name[:-4] + '.jpg')
        assert os.path.isfile(img_path) == True

        with open(txt_path,'rb') as f:
            lines = f.readlines()
        num_txt = len(lines)
        truth = []
        for i in range(num_txt):
            line = lines[i].decode('utf-8').split(',')[:-1]
            truth.append([float(coord) for coord in line ])

        img = Image.open(img_path).convert('RGB')
        w,h = img.size
        for box in truth:
            for i,coord in enumerate(box):
                if i%2 == 0:
                    if coord < 0:
                        box[i] = 0
                    elif coord > w-1:
                        box[i] = w-1
                else:
                    if coord < 0:
                        box[i] = 0
                    elif coord > h-1:
                        box[i] = h-1

        gt_corners = torch.zeros([num_txt*4,5], dtype=torch.float)
        for i,box in enumerate(truth):
            ss1 = math.sqrt(math.pow((box[0] - box[2]),2) + math.pow((box[1] - box[3]),2))
            ss2 = math.sqrt(math.pow((box[0] - box[4]),2) + math.pow((box[1] - box[5]),2))
            if ss1 > ss2:
                ss = ss2
            else:
                ss = ss1
            for j in range(4):
                gt_corners[i*4+j][0] = box[j*2+1] - ss/2
                gt_corners[i*4+j][1] = box[j*2] - ss/2
                gt_corners[i*4+j][2] = box[j*2+1] + ss/2
                gt_corners[i*4+j][3] = box[j*2] + ss/2
                gt_corners[i*4+j][4] = j+1

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
    root_dir = './data/'
    phase = 'train'
    transform = transforms.Compose([transforms.Resize((300,300)),
                                    transforms.ToTensor()])
    dataset = IcprDataset(root_dir, phase, transform)
    # data = dataset[0]
    # print(data['truth'])
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=4,
                            collate_fn=detection_collate)
    for i,batch in enumerate(dataloader):
        imgs, targets = batch
        print(imgs.size(), len(targets[0]))

