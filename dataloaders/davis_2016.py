from __future__ import division

from dataloaders.helpers import *
from torch.utils.data import Dataset

import os
from pathlib import Path as P

import numpy as np
from scipy.misc import imresize
from torch.utils.data import Dataset

from mypath import Path

# if Path.is_custom_pytorch():
# #     sys.path.append(Path.custom_pytorch())  # Custom PyTorch
# # if Path.is_custom_opencv():
# #     sys.path.insert(0, Path.custom_opencv())
import cv2

#log = get_logger(__file__)


class DAVIS2016(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, mode='train',
                 inputRes=None,
                 db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.mode = mode.lower()
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name

        mode_fname_mapping = {
            'train': 'train',
            'test': 'val',
        }
        if self.mode in mode_fname_mapping:
            if self.seq_name is None:
                fname = mode_fname_mapping[self.mode]
            else:
                fname = 'trainval'
        else:
            raise Exception('Mode {} does not exist. Must be one of [\'train\', \'val\', \'test\']')

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

            seq_list = [i[-2] for i in tmp_list]
            fname_list = [i[-1].split('.')[0] for i in tmp_list]
            img_list = [str(path_db_root.joinpath(*i.split('/')))
                        for i in img_list]
            labels = [str(P(*l.split('/')))
                      for l in labels]

        if self.seq_name is not None:
            tmp = [(s, f, i, l)
                   for s, f, i, l in zip(seq_list, fname_list, img_list, labels)
                   if s == self.seq_name]
            tmp = [(s, f, i, l if index == 0 else None)
                   for index, (s, f, i, l) in enumerate(tmp)]
            seq_list, fname_list, img_list, labels = list(zip(*tmp))
            if self.mode == 'train':
                seq_list = [seq_list[0]]
                fname_list = [fname_list[0]]
                img_list = [img_list[0]]
                labels = [labels[0]]

        assert (len(labels) == len(img_list))

        self.seq_list = seq_list
        self.fname_list = fname_list
        self.img_list = img_list
        self.labels = labels

        #log.info('Done initializing ' + fname + ' Dataset')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img, gt = self.make_img_gt_pair(idx)
        sample = {
            'image': img,
            'gt': gt,
            'seq_name': self.seq_list[idx],
            'fname': self.fname_list[idx]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def make_img_gt_pair(self, idx):
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]))
        if self.labels[idx] is not None:
            label = cv2.imread(os.path.join(self.db_root_dir, self.labels[idx]), 0)
        else:
            gt = np.zeros(img.shape[:-1], dtype=np.uint8)

        if self.inputRes is not None:
            img = imresize(img, self.inputRes)
            if self.labels[idx] is not None:
                label = imresize(label, self.inputRes, interp='nearest')

        img = np.array(img, dtype=np.float32)
        img = np.subtract(img, np.array(self.meanval, dtype=np.float32))

        if self.labels[idx] is not None:
            gt = np.array(label, dtype=np.float32)
            gt = gt / np.max([gt.max(), 1e-8])

        return img, gt

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))

        return list(img.shape[:2])

def load_data(data_num, data_set, len_s, batch_size):
    trans = transforms.ToTensor()
    datas = DAVIS2016(data_num, data_set, len_s, trans)
    Dataloader = torch.utils.data.DataLoader(datas, batch_size=batch_size,
                                             shuffle=True, drop_last=False)
    return Dataloader

if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    dataset = DAVIS2016(db_root_dir='/home/zhao/datasets/DAVIS',
                        mode='train', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)
