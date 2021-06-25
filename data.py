

from matplotlib import pyplot as plt
import torch
import cv2
import os
from PIL import Image
from torch import utils
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from pathlib import Path as P
from scipy.misc import imresize
from dataloaders.helpers import *


class davis(Dataset):
    def __init__(self, mode='train',
                 db_root_dir='/home/zhao/datasets/DAVIS',
                 meanval=(104.00699, 116.66877, 122.67892),
                 inputRes=None,
                 seq_len=7,
                 seq_name=None,
                 transform=None):
        """
        mode: 'train','test'
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        seq_len: lenght of input video frames, default by 3
        seq_name: which sequence
        """
        self.mode = mode.lower()
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.meanval = meanval
        self.seq_len = seq_len
        self.seq_name = seq_name
        self.transform = transform


        mode_fname_mapping = {
            'train': 'train_aug',
            'test': 'val',
        }

        if self.mode in mode_fname_mapping:     #if mode is 'train' or 'test', and doesn't named any sequence, then, fname = train or val
            if self.seq_name is None:
                fname = mode_fname_mapping[self.mode]
            else:
                fname = 'trainval'

        else:
            raise Exception('Mode {} does not exist. Must be one of [\'train\', \'val\', \'test\']')
        # stuff
        path_db_root = P(db_root_dir)
        path_sequences = path_db_root / 'ImageSets' / '480p'
        file_extension = '.txt'

        sequences_file = path_sequences / (fname + file_extension)
        with open(str(sequences_file)) as f:
            sequences = f.readlines()
            # sequences[0] == '/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png '
            sequences = [s.split() for s in sequences]
            img_list, labels = zip(*sequences)
            path_db_root.joinpath(*img_list[0].split('/'))
            tmp_list = [i.split('/') for i in img_list]
            dir = [i[-3] for i in tmp_list]
            seq_list = [i[-2] for i in tmp_list]  # seq_list[0] = bear
            fname_list = [i[-1].split('.')[0] for i in tmp_list]  # fname_list[0] = 00000
            img_list = [str(path_db_root.joinpath(*i.split('/')))
                        for i in img_list]
            labels = [str(P(*l.split('/')))
                      for l in labels]


        if self.seq_name is not None:
            tmp = [(d, s, f, i, l)
                   for d, s, f, i, l in zip(dir, seq_list, fname_list, img_list, labels)
                   if s == self.seq_name]
            tmp = [(d, s, f, i, l if index == 0 else None)
                   for index, (d, s, f, i, l) in enumerate(tmp)]
            dir, seq_list, fname_list, img_list, labels = list(zip(*tmp))
            if self.mode == 'train':
                dir = [dir[0]]
                seq_list = [seq_list[0]]
                fname_list = [fname_list[0]]
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        labels_2 = []
        for i in range(len(seq_list)):
            if dir[i]=='mirror':
                labels_2.append(os.path.join('Annotations', '120pmirror', seq_list[i], (fname_list[i] + '.png')))
            else:
                labels_2.append(os.path.join('Annotations', '120p', seq_list[i], (fname_list[i] + '.png')))
        self.seq_list = seq_list
        self.fname_list = fname_list
        self.img_list = img_list
        self.labels = labels
        self.labels_2 = labels_2

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img, gt, gt_2 = self.make_img_gt_pair(idx)
        sample = {
            'image': img,
            'gt': gt,
            'gt_2': gt_2,
            'seq_name': self.seq_list[idx],
            'fname': self.fname_list[idx]            # image = r3.cpu().clone().detach().numpy()
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


    def make_img_gt_pair(self, idx):

        index = idx
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[index]))
        if self.labels[index] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[index]), 0)
            #print(os.path.join(self.db_root_dir,self.labels_2[idx]))
            label_2 = cv2.imread(os.path.join(self.db_root_dir,self.labels_2[index]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            gt_2 = np.zeros((120,214), dtype=np.uint8)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[index] is not None:
                label = imresize(label, self.inputRes, interp='nearest')
                label_2 = imresize(label_2, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[index] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt / np.max([gt.max(), 1e-8])

            gt_2 = np.array(label_2, dtype=np.float32)
            gt_2 = gt_2 / np.max([gt_2.max(), 1e-8])
        return img, gt, gt_2

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

class SegTrackv2(Dataset):
    def __init__(self, mode='train',
                 db_root_dir='/home/zhao/datasets/SegTrackv2',
                 meanval=(104.00699, 116.66877, 122.67892),
                 inputRes=(854,480),
                 seq_len=7,
                 seq_name=None,
                 transform=None):
        """
        mode: 'train','test'
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        seq_len: lenght of input video frames, default by 3
        seq_name: which sequence
        """
        self.mode = mode.lower()
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.meanval = meanval
        self.seq_len = seq_len
        self.seq_name = seq_name
        self.transform = transform


        mode_fname_mapping = {
            # 'train': 'train_aug',
            'train': 'train1',
            'test': 'val',
        }

        if self.mode in mode_fname_mapping:     #if mode is 'train' or 'test', and doesn't named any sequence, then, fname = train or val
            if self.seq_name is None:
                fname = mode_fname_mapping[self.mode]
            else:
                fname = 'trainval'

        else:
            raise Exception('Mode {} does not exist. Must be one of [\'train\', \'val\', \'test\']')
        # stuff
        path_db_root = P(db_root_dir)
        path_sequences = path_db_root / 'ImageSets'
        file_extension = '.txt'

        sequences_file = path_sequences / (fname + file_extension)
        with open(str(sequences_file)) as f:
            sequences = f.readlines()
            # sequences[0] == '/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png '
            sequences = [s.split() for s in sequences]
            img_list, labels = zip(*sequences)
            path_db_root.joinpath(*img_list[0].split('/'))
            tmp_list = [i.split('/') for i in img_list]
            dir = [i[-3] for i in tmp_list]
            seq_list = [i[-2] for i in tmp_list]  # seq_list[0] = bear
            fname_list = [i[-1].split('.')[0] for i in tmp_list]  # fname_list[0] = 00000
            img_list = [str(path_db_root.joinpath(*i.split('/')))
                        for i in img_list]
            labels = [str(P(*l.split('/')))
                      for l in labels]


        if self.seq_name is not None:
            tmp = [(d, s, f, i, l)
                   for d, s, f, i, l in zip(dir, seq_list, fname_list, img_list, labels)
                   if s == self.seq_name]
            tmp = [(d, s, f, i, l if index == 0 else None)
                   for index, (d, s, f, i, l) in enumerate(tmp)]
            dir, seq_list, fname_list, img_list, labels = list(zip(*tmp))
            if self.mode == 'train':
                dir = [dir[0]]
                seq_list = [seq_list[0]]
                fname_list = [fname_list[0]]
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        labels_2 = []
        for i in range(len(seq_list)):
            if dir[i]=='mirror':
                labels_2.append(os.path.join('Annotations', '025gtmirror', seq_list[i], (fname_list[i] + '.png')))
            else:
                labels_2.append(os.path.join('Annotations', '025gt', seq_list[i], (fname_list[i] + '.png')))
        self.seq_list = seq_list
        self.fname_list = fname_list
        self.img_list = img_list
        self.labels = labels
        self.labels_2 = labels_2
        # log.info('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt, gt_2 = self.make_img_gt_pair(idx)
        sample = {
            'image': img,
            'gt': gt,
            'gt_2': gt_2,
            'seq_name': self.seq_list[idx],
            'fname': self.fname_list[idx]            # image = r3.cpu().clone().detach().numpy()
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


    def make_img_gt_pair(self, idx):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
            label_2 = cv2.imread(os.path.join(self.db_root_dir,self.labels_2[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)
            gt_2 = np.zeros((120,214), dtype=np.uint8)

        if self.inputRes is not None:
            img = cv2.resize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = cv2.resize(label, self.inputRes)
                label_2 = cv2.resize(label_2, (214,120))

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt / np.max([gt.max(), 1e-8])

            gt_2 = np.array(label_2, dtype=np.float32)
            gt_2 = gt_2 / np.max([gt_2.max(), 1e-8])
        return img, gt, gt_2

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

def load_data(dataset, mode, seq_len, batch_size):
    trans = transforms.ToTensor()
    if dataset == 'DAVIS':
        datas = davis(mode= mode, seq_len=seq_len, transform=None)
    else:
        datas = SegTrackv2(mode= mode, seq_len=seq_len, transform=None)
    Dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size,
                                             shuffle=False, drop_last=True)
    return Dataloader

if __name__ == '__main__':
    import dataloaders.custom_transforms as tr
    import torch
    #from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = tr.ToTensor()

    dataset = davis(mode='train', seq_len=3, transform=transforms)
    # dataset = SegTrackv2(mode='train', seq_len=3, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()

        print('i=', i)
        print(data['seq_name'],":",data['fname'])
        print(data['image'].shape)
        img1, img2, img3 = data['image'].split(1, 0)
        gt1,  gt2,  gt3 = data['gt'].split(1,0)
        gt21, gt22, gt23 = data['gt_2'].split(1,0)
        print(gt1.shape)
        plt.imshow(overlay_mask(im_normalize(tens2image(img2)), tens2image(gt2)))

        if i == 5:
            break

    plt.show(block=True)
