import random
import cv2
import numpy as np
import torch


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """

    def __init__(self, rots=(-30, 30), scales=(.75, 1.25)):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0]) / 2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots) - 1)]
            sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for elem in sample.keys():
            if elem in ['fname', 'seq_name']:
                continue
            tmp = sample[elem]

            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if tmp.ndim == 2:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_CUBIC

            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            if tmp.min() < 0.0:
                tmp = tmp - tmp.min()

            if tmp.max() > 1.0:
                tmp = tmp / tmp.max()

            sample[elem] = tmp

        return sample


class Resize(object):
    """Randomly resize the image and the ground truth to specified scales.
    Args:
        scales (list): the list of scales
    """

    def __init__(self, scales=[0.5, 0.8, 1]):
        self.scales = scales

    def __call__(self, sample):

        # Fixed range of scales
        sc = self.scales[random.randint(0, len(self.scales) - 1)]

        for elem in sample.keys():
            if elem in ['fname', 'seq_name']:
                continue
            else:
                tmp = sample[elem]

                if tmp.ndim == 2:
                    flagval = cv2.INTER_NEAREST
                else:
                    flagval = cv2.INTER_CUBIC

                tmp = cv2.resize(tmp, None, fx=sc, fy=sc, interpolation=flagval)

                sample[elem] = tmp

        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['fname', 'seq_name']:
                    continue
                else:
                    tmp = sample[elem]
                    tmp = cv2.flip(tmp, flipCode=1)
                    sample[elem] = tmp

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if elem in ['fname', 'seq_name']:
                continue
            else:
                tmp = sample[elem]

                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]

                # swap color axis because
                # numpy image: H x W x C
                # torch image: C X H X W
                #print('tmp',tmp.shape)
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.from_numpy(tmp)

        return sample
