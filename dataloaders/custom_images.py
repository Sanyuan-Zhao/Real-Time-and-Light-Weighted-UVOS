import os
import sys
from pathlib import Path as P

import numpy as np
from scipy.misc import imresize
from torch.utils.data import Dataset

from util.logger import get_logger
from mypath import Path

if Path.is_custom_pytorch():
    sys.path.append(Path.custom_pytorch())  # Custom PyTorch
if Path.is_custom_opencv():
    sys.path.insert(0, Path.custom_opencv())
import cv2

log = get_logger(__file__)


class CustomImages(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, mode='train', inputRes=None,
                 db_root_dir='/usr/stud/ondrag/Me',
                 db_root_dir2='/usr/stud/ondrag/Me',
                 transform=None, meanval=(126.71216173, 119.22616378, 118.00651622)):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        db_root_dir=db_root_dir2
        self.mode = mode.lower()
        self.inputRes = inputRes
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval

        valid_modes = ['train', 'val']
        if mode not in valid_modes:
            raise Exception('Mode {} does not exist. Must be one of {}'.format(self.mode, str(valid_modes)))

        path_db_root = P(db_root_dir)
        file_extension = '.txt'
        path_file_mode = path_db_root / ('train' + file_extension)

        with open(str(path_file_mode)) as f:
            sequences = f.readlines()
            sequences = [s.split() for s in sequences]
            img_list, labels = zip(*sequences)
            path_db_root.joinpath(*img_list[0].split('/'))
            tmp_list = [i.split('/') for i in img_list]
            fname_list = [i[-1].split('.')[0] for i in tmp_list]  # seq_list[0] = 00000
            img_list = [str(path_db_root.joinpath(*i.split('/')))
                        for i in img_list]
            labels = [str(P(*l.split('/')))
                      for l in labels]

        assert (len(labels) == len(img_list))

        self.seq_list = ['Me'] * len(img_list)
        self.fname_list = fname_list
        self.img_list = img_list
        self.labels = labels

        log.info('Done initializing ' + __file__ + ' Dataset')

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


if __name__ == '__main__':
    from dataloaders.custom_transforms import RandomHorizontalFlip, Resize, ToTensor
    from dataloaders.helpers import *

    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([RandomHorizontalFlip(), Resize(scales=[0.5, 0.8, 1]), ToTensor()])

    dataset = CustomImages(db_root_dir=Path.db_root_dir(),
                           mode='train', transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, data in enumerate(dataloader):
        plt.figure()
        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
        if i == 10:
            break

    plt.show(block=True)
