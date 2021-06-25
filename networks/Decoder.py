import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F

class decoder(nn.Module):
    def __init__(self, dropout_rate = 0.1):
        super(decoder, self).__init__()
        self.dr = dropout_rate
        self.convx4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, groups=1024, bias=False),
                                   nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())
        self.convx3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=512, bias=False),
                                   nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())

        self.convx2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
                                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                              bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        self.convx1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False),
                                    nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        ##################################
        self.conv1 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                   nn.BatchNorm2d(512),
                                   nn.Dropout(self.dr),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   #nn.Dropout(self.dr),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   #nn.Dropout(self.dr),
                                   nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1, groups=1,  bias=False),
                                   nn.BatchNorm2d(64),
                                   # nn.Dropout(self.dr),
                                   nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1,  bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),
                                   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(),nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, groups=1,  bias=False),
                                   nn.BatchNorm2d(1),
                                   )
        self.sigmoid = nn.Sigmoid()
        self.conv_midresult = nn.Conv2d(256, 1,kernel_size=1, stride=1, bias=False)




    def forward(self, x, x4,x3,x2,x1):
        x4 = self.convx4(x4)
        x3 = self.convx3(x3)
        x2 = self.convx2(x2)
        x1 = self.convx1(x1)

        x = torch.cat((x, x4), 1)
        x = self.conv1(x)
        x = F.upsample(x, size=(60, 107), mode='bilinear')

        x = torch.cat((x, x3), 1)
        x = self.conv2(x)
        x = F.upsample(x, size=(120, 214), mode='bilinear')

        x = torch.cat((x, x2), 1)
        x = self.conv3(x)
        mid_attention = self.conv_midresult(x)
        mid_attention = self.sigmoid(mid_attention)
        x = mid_attention * x
        x = F.upsample(x, size=(240, 427), mode='bilinear')

        x = torch.cat((x, x1), 1)
        x = self.conv4(x)
        x = F.upsample(x, size=(480, 854), mode='bilinear')
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x, mid_attention