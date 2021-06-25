import argparse
import os

import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn
from torch import optim
from dataloaders.helpers import *

import cv2

from config_tools import get_config
from data import load_data
from networks.network_seqlen3vn5_train2 import Network
from util import tools


def get_average(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)

def train(config, pretrained=None):
    devices = config['gpu_id']
    batch_size = config['batch_size']
    dataset = config['dataset']
    lr = config['lr']
    log_dir = config['log_dir']
    prefix = config['prefix']
    seq_len = batch_size
    meanval = (104.00699, 116.66877, 122.67892)
    print('Using gpus: {}'.format(devices))
    torch.cuda.set_device(devices[0])

    if dataset == 'DAVIS':
        Dataloader = load_data('DAVIS','train', seq_len, batch_size)
    elif dataset == 'SegTrackv2':
        Dataloader = load_data('SegTrackv2','train', seq_len, batch_size)

    len_dl = len(Dataloader)
    print(dataset)
    print(len_dl)

    net = Network()

    if pretrained:
        net.load_state_dict(torch.load(pretrained))
    net.cuda()
    net.train()
    print('Pretrained Model Loading completed!')

    loss_function = nn.BCELoss()
    print('learning rate =',lr)
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[4000, 100000],
                                               gamma=0.1)
    writer = SummaryWriter(log_dir)

    dtype = torch.cuda.FloatTensor
    dtype_t = torch.cuda.LongTensor

    epoch_r = int(300000 / len_dl)
    for epoch in range(1, epoch_r):
        for step, data in enumerate(Dataloader):
            scheduler.step()

            img1, img2, img3 = data['image'].split(1, 0)
            gt1, gt2, gt3 = data['gt'].split(1, 0)
            gt_21, gt_22, gt_23 = data['gt_2'].split(1, 0)


            if(data['seq_name'][0]!=data['seq_name'][-1]):
                continue
            input = data['image'].permute(0, 3, 1, 2)   #shape:[3, 3, 480, 854]

            optimizer.zero_grad()
            input.requires_grad_()

            r1, r2, r3, x1, x2, x3 = net.forward(input.cuda())

            loss1 = loss_function(r1, gt1.type(dtype))
            loss2 = loss_function(r2, gt2.type(dtype))
            loss3 = loss_function(r3, gt3.type(dtype))

            loss_21 = loss_function(x1, gt_21.type(dtype))
            loss_22 = loss_function(x2, gt_22.type(dtype))
            loss_23 = loss_function(x3, gt_23.type(dtype))

            loss = loss1+loss2+loss3+(loss_21+loss_22+loss_23)

            loss.backward()

            mae1 = tools.eval_mae(r1, gt1.cuda())
            f1 = tools.eval_Fscore(r1, gt1.cuda())
            mae2 = tools.eval_mae(r2, gt2.cuda())
            f2 = tools.eval_Fscore(r2, gt2.cuda())
            mae3 = tools.eval_mae(r3, gt3.cuda())
            f3 = tools.eval_Fscore(r3, gt3.cuda())
            mae = (mae1 + mae2 + mae3) / 3
            f = (f1 + f2 + f3) / 3

            image = img3.cpu().clone().detach().numpy()
            result =r3.cpu().clone().detach().permute(2,3,0,1).contiguous().numpy()
            mid_result = x3.cpu().clone().detach().permute(2,3,0,1).contiguous().numpy()
            result = result[:,:,0,:]
            mid_result = mid_result[:,:,0,:]
            image = image[0,:,:,:]

            image = (image - image.min()) / max((image.max() - image.min()), 1e-8)
            cv2.imshow('image', overlay_mask(image,tens2image(gt3)))
            cv2.imshow('gt',tens2image(gt3))
            cv2.imshow('result', result)
            cv2.imshow('mid_result',mid_result)
            cv2.imshow('gt_2', tens2image(gt_23))
            cv2.waitKey(5)

            optimizer.step()

            writer.add_scalar('train/loss-step', loss, epoch * len_dl + step)
            writer.add_scalar('train/MAE-step', mae, epoch * len_dl + step)
            writer.add_scalar('train/Fscore-step', f, epoch * len_dl + step)
            print('epoch{} step{}, loss:{}, MAE:{}, F-score:{}'.format(epoch, step, loss, mae, f))
            #print('loss in epoch{} step{}:{}'.format(epoch, step, loss))

            if step % 1000 == 0:
                torch.save(net.state_dict(),
                           prefix + '.pth')

        if epoch % 2 == 0 and len_dl > 200:
            torch.save(net.state_dict(), prefix + str(epoch) + '.pth')
            print(str(epoch), 'checkpoint saved.')

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-l', dest='log_dir', type=str,
                        help='Location of Logs')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset')
    parser.add_argument('--gpu_id', '-g', type=int, nargs='+', help='GPU Id')
    parser.add_argument('--batch_size', '-b', type=int, help='Batch Size')
    parser.add_argument('--lr', type=float, help='Learning Rate')
    parser.add_argument('--prefix', type=str, help='Model Prefix')
    parser.add_argument('--pretrained', '-p', type=str, help='Pretrained '
                                                             'Model Location')
    parser.add_argument('--config', dest='config_file', help='Config File')
    args = parser.parse_args()
    config_from_args = args.__dict__
    config_file = config_from_args.pop('config_file')
    config = get_config('train', config_from_args, config_file)
    checkpointdir = config['prefix']
    print(checkpointdir)
    if not os.path.exists(checkpointdir):
        os.makedirs(checkpointdir)
    train(config, config['pretrained'])
