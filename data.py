# -*- coding: utf-8 -*-
"""
@File    : data.py
@Author  : 莫胜辉
@Email   : 18230571270@163.com
@Date    : 2025/7/1 17:09
@Version : 1.0
@Desc    : TODO
"""
import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from precache_data import load_train_data


# 进行模型训练时的数据增强类
class Segmentation_augmentation():
    def __init__(self, mean, std):
        self.transform = A.Compose([
            A.RandomRotate90(),  # 随机旋转90
            A.HorizontalFlip(),  # 水平翻转
            A.VerticalFlip(),  # 垂直方向翻转
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.RandomBrightnessContrast(p=0.2),  # 随机的亮度对比度改变
            A.Normalize(mean=mean, std=std),  # 归一化，只会对对应的image进行归一化，不会对标签归一化，其他增强是相同的
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']


# 构造数据集
class Skin_Dataset(Dataset):
    def __init__(self, is_val=False, val_stride=4, data_augmentation=True, data_file_name='cache/train_data.pt'):
        super().__init__()
        self.file_name = data_file_name
        if os.path.exists(self.file_name):
            self.image, self.label = torch.load(self.file_name)  # 加载以npy的形式保存的数据，防止每次训练重复加载，加快模型训练速度
        else:
            load_train_data()
            self.image, self.label = torch.load(self.file_name)
        self.is_val = is_val
        self.val_stride = val_stride
        self.data_augmentation = data_augmentation
        if not self.is_val:  # 以步长为val_stride进行测试集和验证集的划分
            self.split_image_list = [image for i, image in enumerate(self.image) if i % self.val_stride != 0 or i == 0]
            self.split_label_list = [label for i, label in enumerate(self.label) if i % self.val_stride != 0 or i == 0]
        else:
            self.split_image_list = [image for i, image in enumerate(self.image) if i % self.val_stride == 0 and i != 0]
            self.split_label_list = [label for i, label in enumerate(self.label) if i % self.val_stride == 0 and i != 0]

        if os.path.exists("mean_std_pth"):  # 读取保存在文件中的均值和方差，避免每次训练重复计算，第一次时需要调用对应函数
            mean_std = torch.load("mean_std_pth")
        else:
            mean_std = self.cal_mean_and_std()
        self.mean = mean_std['mean']
        self.std = mean_std["std"]
        if not self.is_val and self.data_augmentation:
            self.train_transforms = Segmentation_augmentation(self.mean, self.std)

    def __getitem__(self, item):
        if not self.is_val and self.data_augmentation:
            image, label = self.train_transforms(self.split_image_list[item], self.split_label_list[item])
            return image, label.float().unsqueeze(0)
        else:
            # Image: HWC -> CHW, float and normalized
            image_np = self.split_image_list[item].transpose(2, 0, 1)  # HWC → CHW
            image = torch.from_numpy(image_np).float() / 255.0
            image = (image - self.mean[:, None, None]) / self.std[:, None, None]

            # Label: HW -> 1HW
            label = torch.from_numpy(self.split_label_list[item]).float().unsqueeze(0)
            return image, label

    def __len__(self):
        return len(self.split_image_list)

    def cal_mean_and_std(self):  # 计算均值和方差的函数
        sum_mean = 0
        sum_std = 0
        n = len(self.split_image_list)

        for image in self.split_image_list:
            # image: HWC numpy → CHW tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            sum_mean += image.mean(dim=[1, 2])  # per-channel mean
            sum_std += image.std(dim=[1, 2])  # per-channel std

        mean = sum_mean / n
        std = sum_std / n

        torch.save({"mean": mean, "std": std}, "mean_std_pth")
        return {"mean": mean, "std": std}
