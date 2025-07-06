# -*- coding: utf-8 -*-
"""
@File    : data_for_test.py
@Author  : 莫胜辉
@Email   : 18230571270@163.com
@Date    : 2025/7/4 10:51
@Version : 1.0
@Desc    : TODO
"""
import os.path

import torch

from precache_data import load_test_data
from torch.utils.data import Dataset


class SkinSegmentationTestDataset(Dataset):
    def __init__(self, file_name='cache/test_data.pt'):
        super().__init__()
        self.file_name = file_name
        if os.path.exists(self.file_name):
            self.image_list = torch.load(self.file_name)
        else:
            load_test_data()
            self.image_list = torch.load(self.file_name)
        self.mean_std = torch.load('mean_std_pth')

    def __getitem__(self, item):
        image = self.image_list[item].transpose(2, 0, 1)
        image = torch.from_numpy(image).float() / 255
        image = (image - self.mean_std['mean'][:, None, None]) / self.mean_std['std'][:, None, None]
        return image

    def __len__(self):
        return len(self.image_list)
