from torch.utils.data import Dataset
from PIL import Image
from skimage import io
from os.path import join
import numpy as np
import random
import glob
import torch
import torch.nn as nn
import torch.nn.functional as f
from math import fabs
import cv2
import os
from os.path import join


class Low_Light_Dataset(Dataset):
    def __init__(self, root, transform_low, transform_high):
        self.root = root
        self.transform_low = transform_low
        self.transform_high = transform_high
        self.low_folder = join(root, 'low')
        self.low_paths = os.listdir(self.low_folder)
        self.low_paths.sort()
        self.high_folder = join(root, 'high')
        self.high_paths = os.listdir(self.high_folder)
        self.high_paths.sort()
        # self.seg_folder = join(root, '')
        self.length = len(self.high_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        low_image = Image.open(join(self.low_folder, self.low_paths[i]))
        high_image = Image.open(join(self.high_folder, self.high_paths[i]))

        seed = np.random.randint(2147483647)

        torch.manual_seed(seed)
        if self.transform_low is not None:
            low_image = self.transform_low(low_image)

        torch.manual_seed(seed)
        if self.transform_high is not None:
            high_image = self.transform_high(high_image)

        return [low_image, high_image]
