# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp
import cv2

from utils import frame_utils


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.image_list = []
        self.width = None
        self.height = None
        self.image_root = None

    def __getitem__(self, index):
        img1 = frame_utils.read_gen(os.path.join(self.image_root, self.image_list[index][0] + ".jpg"))
        img2 = frame_utils.read_gen(os.path.join(self.image_root, self.image_list[index][1] + ".jpg"))

        img1 = np.array(img1).astype(np.uint8)[..., :3]
        img2 = np.array(img2).astype(np.uint8)[..., :3]
        img1 = cv2.resize(img1, dsize=(self.width, self.height),interpolation=cv2.INTER_AREA)
        img2 = cv2.resize(img2, dsize=(self.width, self.height), interpolation=cv2.INTER_AREA)
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        return img1, img2, self.image_list[index][0], self.image_list[index][1]

    def __len__(self):
        return len(self.image_list)


class OurDate(FlowDataset):
    def __init__(self, image_root='datasets/Sintel', width=1536, height=768):
        super(OurDate, self).__init__()
        self.height = height
        self.width = width
        self.image_root = image_root
        image_list = sorted(os.listdir(image_root))
        if 'background.jpg' in image_list:
            image_list.remove('background.jpg')
        for i in range(0, len(image_list) - 1, 1):
            image_name_f = image_list[i].split(".")[0]
            image_name_b = image_list[i + 1].split(".")[0]
            self.image_list += [[image_name_f, image_name_b]]
            self.image_list += [[image_name_b, image_name_f]]



