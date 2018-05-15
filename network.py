import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        
        self.config = config
        
        # conv1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # conv4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # conv5
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        # conv6
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.ReLU(inplace=True)
        )
        # conv7
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        # conv8
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        )
        # conv9
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        )
        # conv10
        self.conv10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        # conv11
        self.conv11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        )
        
        # DeconvolutionLayer的第一个参数是需要转置卷积（即2倍上采样）的输入，第二个参数是不需要上采样的（即从网络前层获取的特征）
        # deconv
        # in_channels(f11, conv10) = (256, 256)
        self.deconv1 = DeconvLayer(256, 256) # feature from conv10
        # in_channels(f10, conv9) = (256, 256)
        self.deconv2 = DeconvLayer(256, 256) # feature from conv9
        # in_channels(f9, conv8) = (256, 512)
        self.deconv3 = DeconvLayer(256, 512) # feature from conv8
        # in_channels(f8, conv7) = (256, 1024)
        self.deconv4 = DeconvLayer(256, 1024) # feature from conv7
        # in_channels(f7, conv4) = (256, 512)
        self.deconv5 = DeconvLayer(256, 512) # feature from conv4
        # in_channels(f4, conv3) = (256, 256)
        self.deconv6 = DeconvLayer(256, 256) # feature from conv3
        
        self.pps = PositionSensitiveSegmentationLayer()
        
        # 第一个参数是输入的feature map in_hannels
        self.pred_f11 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION, self.config.NUMBER_OF_CORNER_TYPE)
        self.pred_f10 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION, self.config.NUMBER_OF_CORNER_TYPE)
        self.pred_f9 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION, self.config.NUMBER_OF_CORNER_TYPE)
        self.pred_f8 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION, self.config.NUMBER_OF_CORNER_TYPE)
        self.pred_f7 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION, self.config.NUMBER_OF_CORNER_TYPE)
        self.pred_f4 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION, self.config.NUMBER_OF_CORNER_TYPE)
        self.pred_f3 = PredScoreAndOffsetLayer(256, self.config.BOXES_PER_LOCATION_FOR_F3, self.config.NUMBER_OF_CORNER_TYPE)
        
        print('finish')
        
    def forward(self, x):
        c1 = self.conv1(x)
        p1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.conv2(p1)
        p2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        c3 = self.conv3(p2)
        p3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        c4 = self.conv4(p3)
        p4 = F.max_pool2d(c4, kernel_size=2, stride=2)
        c5 = self.conv5(p4)
        p5 = F.max_pool2d(c5, kernel_size=3, stride=1, padding=1)
        
        c6 = self.conv6(p5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)
        c9 = self.conv9(c8)
        c10 = self.conv10(c9)
        
        f11 = self.conv11(c10)
        
        f10 = self.deconv1(f11, c10)
        f9 = self.deconv2(f10, c9)
        f8 = self.deconv3(f9, c8)
        f7 = self.deconv4(f8, c7)
        f4 = self.deconv5(f7, c4)
        f3 = self.deconv6(f4, c3)
        
        
        pred_segs = self.pps(f3, f4, f7, f8, f9)
        
        pred_f11_scores, pred_f11_offsets = self.pred_f11(f11)
        pred_f10_scores, pred_f10_offsets = self.pred_f10(f10)
        pred_f9_scores, pred_f9_offsets = self.pred_f9(f9)
        pred_f8_scores, pred_f8_offsets = self.pred_f8(f8)
        pred_f7_scores, pred_f7_offsets = self.pred_f7(f7)
        pred_f4_scores, pred_f4_offsets = self.pred_f4(f4)
        pred_f3_scores, pred_f3_offsets = self.pred_f3(f3)
        
        pred_scores = self.gather(pred_f3_scores, pred_f4_scores, pred_f7_scores, pred_f8_scores, pred_f9_scores,
                                  pred_f10_scores, pred_f11_scores)
        pred_offsets = self.gather(pred_f3_offsets, pred_f4_offsets, pred_f7_offsets, pred_f8_offsets, pred_f9_offsets,
                                  pred_f10_offsets, pred_f11_offsets)
    
        return pred_scores, pred_offsets, pred_segs
        
    def gather(self, *preds):
        assert len(preds) > 0
        
        boxes_per_location_for_f3 = self.config.BOXES_PER_LOCATION_FOR_F3
        boxes_per_location = self.config.BOXES_PER_LOCATION
        
        batch_size = preds[0].size()[0]
        channels = preds[0].size()[1]
        
        outputs = []
        
        for i in range(len(preds)):
            feature_map = preds[i]
            height = feature_map.size()[2]
            width = feature_map.size()[3]
            feature_map = feature_map.permute(0, 2, 3, 1).contiguous()
            if i == 0:
                feature_map = feature_map.view(batch_size, height * width * boxes_per_location_for_f3, -1)
            else:
                feature_map = feature_map.view(batch_size, height * width * boxes_per_location, -1)
                
            outputs.append(feature_map)
        
        return torch.cat(outputs, dim=1)
        
        
    
    
class DeconvLayer(nn.Module): 
    def __init__(self, in_channels1, in_channels2):
        super(DeconvLayer, self).__init__()
        
        self.process1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels1, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True)
        )
        
        self.process2 = nn.Sequential(
            nn.Conv2d(in_channels2, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, affine=True)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x1, x2):
        out_1 = self.process1(x1)
        out_2 = self.process2(x2)
        print(out_1.size())
        print(out_2.size())
        out = out_1 + out_2
        out = self.relu(out)
        
        return out

class PredScoreAndOffsetLayer(nn.Module):
    def __init__(self, in_channels, boxes_per_location, number_of_corner_type):
        super(PredScoreAndOffsetLayer, self).__init__()
        
        self.in_channels = in_channels
        self.boxes_per_location = boxes_per_location
        self.number_of_corner_type = number_of_corner_type
        
        self.shortcut = nn.Conv2d(in_channels, 1024, kernel_size=1, stride=1)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, stride=1),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1),
        )
        
        self.predict_scores = nn.Conv2d(1024, self.boxes_per_location * self.number_of_corner_type * 2, kernel_size=1, stride=1)
        
        self.predict_offsets = nn.Conv2d(1024, self.boxes_per_location * self.number_of_corner_type * 4, kernel_size=1, stride=1)
    
    def forward(self, x):
        out = self.layer(x)
        feature_shared = self.shortcut(x) + out
        
        pred_scores = self.predict_scores(feature_shared)
        pred_offsets = self.predict_offsets(feature_shared)
        
        return pred_scores, pred_offsets

    
class PositionSensitiveSegmentationLayer(nn.Module):
    def __init__(self):
        super(PositionSensitiveSegmentationLayer, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 4, kernel_size=2, stride=2)
        )
        
    def forward(self, f3, f4, f7, f8, f9):
        
        print(f3.size())
        print(f4.size())
        print(f7.size())
        print(f8.size())
        print(f9.size())
        out = f3 + F.upsample(f4, scale_factor=2, mode='bilinear') + F.upsample(f7, scale_factor=4, mode='bilinear') + \
            F.upsample(f8, scale_factor=8, mode='bilinear') + F.upsample(f9, scale_factor=16, mode='bilinear')
        out = self.layer(out)
        return out
