#!/usr/bin/env python
"""
    File Name   :   SphericalViewSynthesis-pabellon
    date        :   20/6/2020
    Author      :   wenbo
    Email       :   huwenbodut@gmail.com
    Description :
                              _     _
                             ( |---/ )
                              ) . . (
________________________,--._(___Y___)_,--._______________________
                        `--'           `--'
"""

import pdb
import random
from os.path import join

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import util
import dataset.transform as Xform


def inverse_normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.get_device(), requires_grad=False)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.get_device(), requires_grad=False)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return x * std + mean


def normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.get_device(), requires_grad=False)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.get_device(), requires_grad=False)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    return (x - mean) / std


class Pabellon(Dataset):

    def __init__(self, data_root=None, data_list=None, rescale_width=512, loop=1):
        super(Pabellon, self).__init__()

        self.frames = util.translate_datalist(data_root, data_list, number=1)
        self.loop = loop
        self.aug = Xform.Compose([
            Xform.ToTensor(),
            # Xform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.rescale_width = rescale_width

    def __len__(self):
        return len(self.frames) * self.loop

    # @staticmethod
    # def pad_image(img, basics):
    #     H, W, _ = img.shape
    #     paded_H = (H // basics + 1) * basics
    #     paded_W = (W // basics + 1) * basics
    #     front_pad_H, front_pad_W = (paded_H - H) // 2, (paded_W - W) // 2
    #
    #     img_pad = np.zeros(shape=[paded_H, paded_W, 3], dtype=img.dtype)
    #     img_pad[front_pad_H:front_pad_H + H, front_pad_W:front_pad_W + W, :] = img
    #     return img_pad, [H, W]
    #
    # @staticmethod
    # def depad_image(img_pad, shape):
    #     paded_H, paded_W = img_pad.shape
    #     H, W = shape
    #     front_pad_H, front_pad_W = (paded_H - H) // 2, (paded_W - W) // 2
    #
    #     img = img_pad[front_pad_H:front_pad_H + H, front_pad_W:front_pad_W + W, ...]
    #     return img
    #
    # @staticmethod
    # def depad_tensor(img_pad, shape):
    #     # pdb.set_trace()
    #     paded_H, paded_W = img_pad.shape[-2:]
    #     H, W = shape
    #     front_pad_H, front_pad_W = (paded_H - H) // 2, (paded_W - W) // 2
    #
    #     img = img_pad[:, :, front_pad_H:front_pad_H + H, front_pad_W:front_pad_W + W]
    #     return img

    def __getitem__(self, index_long):
        """
        :param index:
        :return: RGB, np.float32, normalized to (0.0~1.0)
        """
        index = index_long % len(self.frames)

        # left = cv2.cvtColor(cv2.imread(self.frames[index][0]), cv2.COLOR_BGR2RGB)
        # right = cv2.cvtColor(cv2.imread(self.frames[index][0].replace('_L.png', '_R.png')), cv2.COLOR_BGR2RGB)
        left = cv2.imread(self.frames[index][0])
        right = cv2.imread(self.frames[index][0].replace('_L.png', '_R.png'))
        if self.rescale_width is not None:
            left = cv2.resize(left, dsize=(self.rescale_width, self.rescale_width // 2), interpolation=cv2.INTER_CUBIC)
            right = cv2.resize(right, dsize=(self.rescale_width, self.rescale_width // 2),
                               interpolation=cv2.INTER_CUBIC)

        # aug
        left, right = self.aug(left / 255., right / 255.)
        # left, right = self.aug(left, right)
        return left, right
