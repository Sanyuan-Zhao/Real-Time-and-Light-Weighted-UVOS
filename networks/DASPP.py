import torch
from torch import nn
import torch.nn.functional as F

class DASPP(nn.Module):
    def __init__(self, features, H, W, M, G, r, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(DASPP, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.H = H
        self.W = W
        self.conv1 = nn.Sequential(nn.Conv2d(features, features, kernel_size=1),nn.BatchNorm2d(features))

        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=6*(2+i), groups=G, dilation=6*(2+i)),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(features, features, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(features)

        self.conv_1x1_3 = nn.Conv2d(3*features, 128, kernel_size=1)  # (128 = 3*32)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(128)
        self.conv_1x1_4 = nn.Conv2d(128, features*2, kernel_size=1)
        #self.show = nn.Conv2d()
        
    def forward(self, x):

        out_img = self.avg_pool(x)  # (shape: (batch_size, 512, 1, 1))
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (shape: (batch_size, 256, 1, 1))
        out_img = F.upsample(out_img, size=(self.H, self.W), mode="bilinear")  # (shape: (batch_size, 256, h/16, w/16))

        conv1 = self.conv1(x)


        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)

        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        output = torch.cat([conv1, fea_v, out_img], dim=1)
        output = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(output)))
        output = self.conv_1x1_4(output)
        return output


class DynamicUnit(nn.Module):
    def __init__(self, in_features, out_features, H, W, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(DUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)  #mid features = 32
            #print('mid_features',mid_features)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            DASPP(mid_features, H, W, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(out_features),
            nn.Conv2d(out_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)

if __name__=='__main__':
    x = torch.rand(5, 512, 30, 54)
    net = DynamicUnit(in_features=512, out_features=512, H=30, W=54, M=3, G=8, r=2, stride=1)
    out = net(x)
    criterion = nn.L1Loss()
    loss = criterion(out, x)
    loss.backward()
    print('out shape : {}'.format(out.shape))
    print('loss value : {}'.format(loss))