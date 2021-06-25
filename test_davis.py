import argparse
import os
from pathlib import Path as P
from matplotlib import pyplot as plt

import torch
import torch.utils.data

from dataloaders.helpers import *
from config_tools import get_config
import util.tools as tools

from networks.network_seqlen3vn5_train2 import Network

db_root_dir = '/home/zhao/datasets/DAVIS'
seq_list = []
fname_list = []
img_list = []
labels = []

def load_davis_test_data():

    meanval = (104.00699, 116.66877, 122.67892)
    path_db_root = P(db_root_dir)
    path_sequences = path_db_root / 'ImageSets' / '480p'
    file_extension = '.txt'
    seq_name = None #'blackswan'
    mode = 'test'
    mode_fname_mapping = {
        'train': 'train',
        'test': 'val',
    }
    if mode in mode_fname_mapping:  # if mode is 'train' or 'test', and doesn't named any sequence, then, fname = train or val
        if seq_name is None:
            fname = mode_fname_mapping[mode]
        else:
            fname = 'trainval'

    else:
        raise Exception('Mode {} does not exist. Must be one of [\'train\', \'val\', \'test\']')

    sequences_file = path_sequences / (fname + file_extension)
    with open(str(sequences_file)) as f:
        sequences = f.readlines()
        sequences = [s.split() for s in sequences]
        img_list, labels = zip(*sequences)
        path_db_root.joinpath(*img_list[0].split('/'))
        tmp_list = [i.split('/') for i in img_list]

        seq_list = [i[-2] for i in tmp_list]
        fname_list = [i[-1].split('.')[0] for i in tmp_list]
        img_list = [str(path_db_root.joinpath(*i.split('/')))
                    for i in img_list]
        labels = [str(P(*l.split('/')))
                  for l in labels]

    if seq_name is not None:
        tmp = [(s, f, i, l)
               for s, f, i, l in zip(seq_list, fname_list, img_list, labels)
               if s == seq_name]
        tmp = [(s, f, i, l if index == 0 else None)
               for index, (s, f, i, l) in enumerate(tmp)]
        seq_list, fname_list, img_list, labels = list(zip(*tmp))
        if mode == 'train':
            seq_list = [seq_list[0]]
            fname_list = [fname_list[0]]
            img_list = [img_list[0]]
            labels = [labels[0]]

    print('seq_list:',seq_list)
    print('fname:',fname)
    print('img_list:', img_list)
    print('labels:',labels)
    assert (len(labels) == len(img_list))



def test(net, seq_len, dataset):
    '''
    Test on validation dataset
    :param net: net to evaluate
    :param mode: full image or cropped image
    :param Dataloader: data to evaluate
    :return:
    '''

    dtype = torch.cuda.FloatTensor
    dtype_t = torch.cuda.LongTensor
    meanval = (104.00699, 116.66877, 122.67892)
    savedir_name = config['savedir_name']
    if not os.path.exists(savedir_name):
        os.makedirs(savedir_name)

    DAVIS_classes = ['blackswan','bmx-trees','breakdance','camel','car-roundabout','car-shadow','cows','dance-twirl','dog','drift-chicane','drift-straight',
               'goat','horsejump-high','kite-surf','libby','motocross-jump','paragliding-launch','parkour','scooter-black','soapbox']

    img_number = {'blackswan': 50,
                  'bmx-trees': 80,
                  'breakdance': 84,
                  'camel': 90,
                  'car-roundabout': 75,
                  'car-shadow': 40,
                  'cows': 104,
                  'dance-twirl': 90,
                  'dog': 60,
                  'drift-chicane': 52,
                  'drift-straight': 50,
                  'goat': 90,
                  'horsejump-high': 50,
                  'kite-surf': 50,
                  'libby': 49,
                  'motocross-jump': 40,
                  'paragliding-launch': 80,
                  'parkour': 100,
                  'scooter-black': 43,
                  'soapbox': 99}

    precision = {}
    recall = {}
    Fsocre = {}
    mae = {}  #Mean Absolute Error
    for cls in DAVIS_classes:
        precision[cls] = 0.0
        recall[cls]    = 0.0
        mae[cls]       = 0.0
        Fsocre[cls]         = 0.0


    for cls in DAVIS_classes:
        cls_result = os.path.join(savedir_name,cls)
        if not os.path.exists(cls_result):
            os.makedirs(cls_result)
        cls_mae = []
        cls_prec = []
        cls_recall = []
        for i in range(0, img_number[cls]-(seq_len-1), 1):
            filename = []
            labelname = []
            img = []
            label = []

            for j in range(seq_len):

                filename.append(os.path.join('/home/zhao/datasets/DAVIS/JPEGImages/480p/',cls,'%05d.jpg' %(i+j)))
                labelname.append(os.path.join('/home/zhao/datasets/DAVIS/Annotations/480p/', cls, '%05d.png' %(i+j)))

                img.append(cv2.imread(filename[j], 1))
                img[j] = np.subtract(np.array(img[j], dtype=np.float32), np.array(meanval, dtype=np.float32))
                img[j] = torch.tensor(img[j]).permute(2, 0, 1)
                label.append(cv2.imread(labelname[j], 0))
                label[j] = np.array(label[j], dtype=np.float32)
                label[j] = label[j] / np.max([label[j].max(), 1e-8])
                label[j] = torch.tensor(label[j])
                result = []

                if j==0:
                    input = torch.unsqueeze(img[j], 0)

                else:
                    inputimg = torch.unsqueeze(img[j], 0)
                    input = torch.cat((input, inputimg), 0)

                if j==seq_len-1:
                    net.zero_grad()
                    with torch.no_grad():
                        r = net(input.cuda())
                    for ri in range(seq_len):

                        result.append(r[ri])        #size: 1*1*480*854

                        cls_mae.append(tools.eval_mae(result[ri], label[ri].cuda()).cpu().detach().numpy())
                        prec, rec = tools.eval_pr(result[ri], label[ri].cuda())
                        cls_prec.append(prec)
                        cls_recall.append(rec)

                        result[ri] = result[ri].cpu().clone().detach().permute(2, 3, 0, 1).contiguous().numpy()
                        result[ri] = result[ri][:, :, 0, :]
                        result[ri] = result[ri] * 255
                        result[ri] = result[ri].astype(np.uint8)

                        label[ri] = label[ri].cpu().clone().detach().contiguous().numpy()
                        label[ri] = label[ri] * 255
                        label[ri] = label[ri].astype(np.uint8)

                        cv2.imshow('result'+ str(ri), result[ri])
                        cv2.imwrite(os.path.join(cls_result, str(i+ri).zfill(5) + '.png'), result[ri])
                        cv2.imshow('lable' + str(ri), label[ri])
                        cv2.waitKey(5)

        beta = 0.3
        mae[cls] = tools.get_average(cls_mae)
        precision[cls] = tools.get_average(cls_prec)
        recall[cls] = tools.get_average(cls_recall)
        if precision[cls] == None:
            precision[cls] = 0
        Fsocre[cls] = (1 + beta ** 2) * precision[cls] * recall[cls] / (beta ** 2 * precision[cls] + recall[cls])
        print(cls,'mae is', mae[cls],'|| prec is', precision[cls],'|| recall is', recall[cls],'|| Fsocre is', Fsocre[cls])

    print('-----------------------------------------------')
    temp = []
    for k,v in mae.items():
        temp.append(v)
    Mae = tools.get_average(temp)
    print('Mae is',Mae)
    temp = []
    for k,v in Fsocre.items():
        temp.append(v)
    Mean_Fsocre = tools.get_average(temp)
    Max_Fsocre = max(temp)
    print('Mean Fscore is', Mean_Fsocre)
    print('Max Fsocre is', Max_Fsocre)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpu_id', nargs='+', type=int)
    parser.add_argument('--num', type=int, )
    parser.add_argument('--model', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--savedir_name', type=str)
    parser.add_argument('--config', dest='config_file', help='Config File')
    args = parser.parse_args()

    config_from_args = args.__dict__
    config_file = config_from_args.pop('config_file')
    config = get_config('test', config_from_args, config_file)

    devices = config['gpu_id']
    num = config['num']
    dataset = config['dataset']
    model = config['model']
    seq_len = config['batchsize']

    print('gpus: {}'.format(devices))
    torch.cuda.set_device(devices[0])

    load_davis_test_data()

    net = Network()

    #net = nn.DataParallel(net, device_ids=devices)
    net.load_state_dict(torch.load(config['model']))
    net.cuda()
    print('Loading completed!')
    net.eval()
    test(net, seq_len, dataset)

