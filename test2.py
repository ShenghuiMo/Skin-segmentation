# -*- coding: utf-8 -*-
"""
@File    : test2.py
@Author  : 莫胜辉
@Email   : 18230571270@163.com
@Date    : 2025/7/4 14:45
@Version : 1.0
@Desc    : TODO
"""
from datetime import datetime
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from data_for_test import SkinSegmentationTestDataset
from U_net_model import UNET
from convert_pre_csv import save_submission_csv


class SegmentationTest():
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNET().to(self.device)

        # 加载模型权重
        checkpoint = torch.load("model_pth/best1.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])

        self.model.eval()

    def Init_test_loader(self):
        test_loader = DataLoader(
            SkinSegmentationTestDataset(),
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        return test_loader

    def main(self):
        test_loader = self.Init_test_loader()
        predictions = []

        with torch.no_grad():
            for images in tqdm(test_loader, desc="Predicting..."):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = (outputs > 0.5).float()  # 二值化
                outputs = outputs.squeeze(1).cpu().numpy()  # shape: [B, H, W]
                for i in range(outputs.shape[0]):
                    predictions.append(outputs[i].astype(np.uint8))

        # 获取原图大小，用于 resize
        original_sizes = []
        image_dir = 'Dataset/Test/Image'
        image_names = sorted(os.listdir(image_dir))
        for name in image_names:
            img = Image.open(os.path.join(image_dir, name))
            original_sizes.append(img.size)  # (width, height)

        # Resize 掩码到原图尺寸
        resized_masks = []
        for idx, mask in enumerate(predictions):
            w, h = original_sizes[idx]
            mask_img = Image.fromarray(mask * 255).resize((w, h), resample=Image.NEAREST)
            resized_mask = np.array(mask_img)
            resized_mask = (resized_mask > 127).astype(np.uint8)
            resized_masks.append(resized_mask)

        # 生成 ID 列表（如 00001, 00002 ...）
        id_list = [f"{i+1087:05d}" for i in range(len(resized_masks))]

        # 保存为 CSV
        save_submission_csv(id_list, resized_masks, save_path='pre/submission1.csv')


if __name__ == "__main__":
    tester = SegmentationTest(batch_size=1)
    tester.main()
