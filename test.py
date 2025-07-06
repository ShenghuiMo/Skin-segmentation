# -*- coding: utf-8 -*-
"""
@File    : test.py
@Author  : 莫胜辉
@Email   : 18230571270@163.com
@Date    : 2025/7/3 8:40
@Version : 1.0
@Desc    : TODO
"""
from datetime import datetime
import os.path

import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_for_test import SkinSegmentationTestDataset
from U_net_model import UNET
from tensorboardX import SummaryWriter
from convert_pre_csv import save_submission_csv
from PIL import Image
import cv2

from PIL import Image
import numpy as np


class SegmentationTest():
    def __init__(self, batch_size=1):
        self.cuda_use = torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = 'cuda' if self.cuda_use else 'cpu'
        self.model = UNET()
        self.date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.cuda_use:
            self.model.to(self.device)
        self.check_point = torch.load('model_pth/best1.pth')
        self.model.load_state_dict(self.check_point['model_state'])
        if not os.path.exists("test/runs"):
            os.makedirs('test_runs')
        self.writer = SummaryWriter(
            logdir=f"test_runs/{self.date_time}_batch_size_{self.batch_size}_cuda_use_{self.cuda_use}")

    def Init_test_loader(self):
        test_loader = DataLoader(SkinSegmentationTestDataset(), batch_size=self.batch_size, pin_memory=self.cuda_use,
                                 num_workers=4)
        return test_loader

    def main(self):
        test_loader = self.Init_test_loader()
        self.model.eval()
        pre_list = []
        with torch.no_grad():
            for batch_index, image in tqdm(enumerate(test_loader), desc="test...."):
                image = image.to(self.device)
                pre = self.model(image)
                pre = (pre > 0.5).float()
                pre_list.append((pre[0].cpu().numpy()).astype(np.uint8))
                if batch_index % 10 == 0:
                    image = image[0].cpu()
                    pre = pre[0].cpu()
                    self.writer.add_image(f'image_{batch_index}', image, global_step=batch_index)
                    self.writer.add_image(f'pre_{batch_index }', pre, global_step=batch_index)
        self.writer.close()
        if os.path.exists('pre'):
            torch.save(pre_list, 'pre/mask.pth')
        else:
            os.makedirs('pre')
            torch.save(pre_list, 'pre/mask.pth')

        # # 读取原图尺寸，假设这样：
        # original_sizes = []
        # file_name_image = 'Dataset/Test/Image'
        # image_names = sorted(os.listdir(file_name_image))  # 按名称排序保证顺序一致
        #
        # for image_path in image_names:
        #     img = np.array(Image.open(os.path.join(file_name_image, image_path))).transpose(2,0,1)
        #     original_sizes.append(img.shape[1:3])
        # for idx,image_numpy in enumerate(pre_list):
        #     print(image_numpy.shape)
        #     image_pil = Image.fromarray(image_numpy)
        #     h, w = original_sizes[idx]
        #     pre_pil = image_pil.resize((h,w),resample=Image.NEAREST)
        #     pre_list[idx] = np.array(pre_pil)
        # id_list = [f"{i + 1087:05d}" for i in range(len(pre_list))]
        # save_submission_csv(id_list, pre_list)


if __name__ == "__main__":
    Test_ = SegmentationTest()
    Test_.main()
