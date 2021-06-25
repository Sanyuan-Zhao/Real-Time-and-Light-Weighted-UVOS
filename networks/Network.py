
import torch.nn as nn
from torch.autograd import Variable
import torch
import os

from networks.resnet import ResNet101_OS16
from networks.DASPP import DynamicUnit
from networks.APNB import APNB
from networks.RNN_Conv import RNN_Conv
from networks.Decoder import decoder


class Network(nn.Module):
    def __init__(self, model_id=1, project_dir='./test1/', dropout_rate=0.1):
        super(Network, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        self.feature_H = 30
        self.feature_W = 54
        self.DASPP_in_features = 512
        self.DSAPP_out_features = 512
        self.dropout_rate = dropout_rate

        self.ResNet = ResNet101_OS16()  # NOTE! specify the type of ResNet here
        self.conv1 = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1, dilation=1, groups=2048, bias=False),
            nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(self.dropout_rate),
            nn.ReLU())
        self.DASPP = DynamicUnit(in_features=self.DASPP_in_features, out_features=self.DSAPP_out_features, H=self.feature_H,
                                 W=self.feature_W, M=4, G=8, r=2, stride=1)
        self.APNB = APNB(in_channels=512, out_channels=256, key_channels=256, value_channels=512,
                         dropout=0.05, sizes=([1]))
        self.conv2 = nn.Conv2d(768,256,3,1,1,1)
        self.rnn_conv = RNN_Conv(input_size=(30, 54), input_dim=256, hidden_dim=[256,256,256,256], kernel_size=(5, 5),
		                          num_layers=4, p_TD=0.1, batch_first=True, bias=True, return_all_layers=True)
        self.decoder = decoder(self.dropout_rate)
    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        x1, x2, x3, x4, feature_map = self.ResNet(x)
        feature_map = self.conv1(feature_map)
        #the spatial part1: Dynamic ASPP
        aspp_output = self.DASPP(feature_map)
        # the spatial part2: APNB
        apnb_output = self.APNB(feature_map)

        # concat two feature maps
        spatial_output = torch.cat((aspp_output,apnb_output), 1)
        spatial_output = self.conv2(spatial_output)

        # the temporal part: RNN-Conv
        rnn_input = torch.unsqueeze(spatial_output, 0)
        rnn_output, hidden_state_output = self.rnn_conv(rnn_input)
        rnn_output = torch.squeeze(rnn_output[0], dim=0)

        output, mid_output = self.decoder(rnn_output, x4,x3,x2,x1)
        output1, output2 = torch.split(output, 1, dim=0)
        mid_output1, mid_output2= torch.split(mid_output, 1, dim=0)
        return output1, output2, mid_output1, mid_output2

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)

if __name__ == '__main__':
    print('start')
    net = Network()
    net.cuda()
    print(net)
    batch_image = Variable(torch.randn(2, 3, 480, 854)).cuda()
    x1, x2 = batch_image.split(1, 0)
    y1, y2, mid1, mid2= net.forward(batch_image)

    print('Network parameters:', sum(param.numel() for param in net.parameters()))
