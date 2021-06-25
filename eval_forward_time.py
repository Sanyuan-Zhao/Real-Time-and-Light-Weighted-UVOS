import os
import numpy as np
import torch
import time

from PIL import Image
from argparse import ArgumentParser

from thop import profile




from torch.autograd import Variable
from networks.Network import Network
from networks.resnet import ResNet101_OS16
from networks.non_local import NLBlockND
from networks.DASPP import SKUnit
from networks.ASPP import ASPP
from networks.APNB import APNB

import torchvision.models as models
import torch


import torch.backends.cudnn as cudnn
cudnn.benchmark = True
torch.cuda.set_device(0)
def main(args):

    net = Network()


    if (not args.cpu):
        model = net.cuda()

    model.eval()


    images = torch.randn(args.batch_size, args.num_channels, args.height, args.width)
    inputs = Variable(images)
    if (not args.cpu):
        images = inputs.cuda()#.half()

    time_train = []

    i=0
    for _ in range(100):
    #for step, (images, labels, filename, filenameGt) in enumerate(loader):
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            outputs = net.forward(images)#, True)

        if (i == 10):#not args.cpu):
            torch.cuda.synchronize()
                #wait for cuda to finish (cuda is asynchronous!)

        if i!=0:    #first run always takes some time for setup
            fwt = time.time() - start_time
            time_train.append(fwt)
            print ("Forward time per img (b=%d): %.5f (Mean: %.5f)" % (args.batch_size, fwt/args.batch_size, sum(time_train) / len(time_train) / args.batch_size))

        time.sleep(1)   #to avoid overheating the GPU too much
        i+=1


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--width', type=int, default=854)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())
