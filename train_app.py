# -*- coding: utf-8 -*-
"""
@File    : train_app.py
@Author  : 莫胜辉
@Email   : 18230571270@163.com
@Date    : 2025/7/2 9:17
@Version : 1.0
@Desc    : TODO
"""
import os.path
from datetime import datetime

from tensorboardX import SummaryWriter
import torch.cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Skin_Dataset
from U_net_model import UNET
import torch.optim as opt
import torch.nn as nn
import torch.nn.init as init


def Init_parameters(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.constant(m.weight, 1)
        if m.bias is not None:
            init.constant(m.bias, 0)


def Dice_loss(pre, label, epsilon=1e-6):
    pre = pre.contiguous()
    label = label.contiguous()
    label = (label > 127).float()  # 建议做二值化

    intersection = (pre * label).sum(dim=[1, 2, 3])
    union = pre.sum(dim=[1, 2, 3]) + label.sum(dim=[1, 2, 3])
    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    loss = 1 - dice_score
    return loss.mean(), dice_score.mean()  # 保证是标量，支持 backward()


def bce_loss_dice_loss(pre, label, epsilon=1e-6, weight_bce=0.5, weight_dice=0.5):
    pre = pre.contiguous()
    label = label.contiguous()
    label = (label > 127).float()
    bce_loss = nn.BCELoss()
    loss1 = bce_loss(pre, label)  # bce函数返回一个标量，代表批次的平均值
    intersection = (pre * label).sum(dim=[1, 2, 3])
    union = pre.sim(dim=[1, 2, 3]) + pre.sum(dim=[1, 2, 3])
    dice_score = (2 * intersection + epsilon) / union
    loss2 = 1 - dice_score.mean()
    total_loss = loss1 * weight_bce + loss2 * weight_dice
    return total_loss


class TrainApp():
    def __init__(self, epoch=10, val_stride=4, batch_size=16, weigh_decay=False, check_train=False):
        self.epoch = epoch
        self.val_stride = val_stride
        self.batch = batch_size
        self.weight_decay = weigh_decay
        self.use_cuda = torch.cuda.is_available()
        self.devices_count = torch.cuda.device_count()  # 该数据较小，不建议使用多卡训练，反而会拖慢训练速度
        self.model = UNET()
        if self.devices_count > 1:
            self.model = nn.DataParallel(self.model)
        if not check_train:
            self.model.apply(Init_parameters)   # 当不进行断点训练加载时，进行模型的初始化以加快模型收敛速度
        self.check_train = check_train
        if check_train:
            self.check_point = torch.load("model_pth/best1.pth")
            self.model.load_state_dict(self.check_point['model_state'])
        self.date = datetime.now().strftime("%Y%m%d,%H%M%S")  # 获取当前时间
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print(f"use device{self.device},cuda_isavailable：{self.use_cuda},devices_count:{self.devices_count}")
        self.model.to(self.device)  # 将模型置于GPU上加快训练
        self.writer = SummaryWriter(
            logdir=f"runs/{self.date}_new_data_augmentation_fuse_conv_skip_connection__epoch_{epoch}_val_stride{val_stride}_batch_size{batch_size}_weight_decay{self.weight_decay}_data_normalization")  # 使用tensorboard来可视化结果

    def Init_train_loader(self):  # 构造训练数据加载器
        train_loader = DataLoader(Skin_Dataset(is_val=False, val_stride=self.val_stride), batch_size=self.batch,
                                  shuffle=True,
                                  pin_memory=self.use_cuda, num_workers=4)
        return train_loader

    def Init_val_loader(self):   # 构造验证集加载器
        val_loader = DataLoader(Skin_Dataset(is_val=True, val_stride=self.val_stride), batch_size=self.batch,
                                shuffle=True,
                                pin_memory=self.use_cuda, num_workers=4)
        return val_loader

    def Init_optimizer(self, is_weight_decay=False):  # 构建优化器，采用adam优化器，能够自动调动学习率
        if not is_weight_decay:
            optimizer = opt.Adam(self.model.parameters())
            return optimizer
        else:
            return opt.Adam(self.model.parameters(), weight_decay=1e-5)

    def train_(self, train_loader, optimizer, e):
        average_epoch_loss = 0
        average_dice = 0
        self.model.train()
        for image, label in tqdm(train_loader, desc="training...."):
            image = image.to(self.device)
            label = label.to(self.device)
            pre = self.model(image)
            loss, dice = Dice_loss(pre, label)
            average_epoch_loss += loss.item()
            average_dice += dice.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_epoch_loss = average_epoch_loss / len(train_loader)
        average_dice = average_dice / len(train_loader)
        self.writer.add_scalar('train_average_loss', average_epoch_loss, e)
        self.writer.add_scalar("train_average_dice", average_dice, e)
        print(f'epoch: {e}, average_loss: {average_epoch_loss},dice:{average_dice}')
        return average_epoch_loss

    def val_(self, val_loader, e):
        average_val_loss = 0
        self.model.eval()
        average_dice = 0
        for batch_idx, (image, label) in enumerate(tqdm(val_loader, desc="val...")):
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.no_grad():
                pre = self.model(image)
                loss, dice = Dice_loss(pre, label)
                average_val_loss += loss.item()
                average_dice += dice.item()
                if batch_idx == 0:
                    # 转为 [0,1] 范围（视情况取sigmoid或softmax）

                    # 只取前1张图像进行展示
                    img = image[0].cpu()
                    pre = pre[0].cpu()
                    gt = label[0].cpu()

                    # 可以用灰度图（[1,H,W]）显示
                    self.writer.add_image(f'input_image/epoch_{e}', img, global_step=e)
                    self.writer.add_image(f'gt_mask/epoch_{e}', gt, global_step=e)
                    self.writer.add_image(f'pred_mask/epoch_{e}', pre, global_step=e)

        average_val_loss = average_val_loss / len(val_loader)
        average_dice = average_dice / len(val_loader)
        self.writer.add_scalar("val_train_loss", average_val_loss, global_step=e)
        self.writer.add_scalar("val_dice", average_dice, global_step=e)
        print(f"epoch:{e},val_loss:{average_val_loss},dice:{average_dice}")
        return average_val_loss

    def main(self):
        train_loader = self.Init_train_loader()
        val_loader = self.Init_val_loader()
        optimizer = self.Init_optimizer(is_weight_decay=self.weight_decay)
        if self.check_train:
            start_epoch = self.check_point['epoch'] + 1
            val_best = self.check_point['loss']
            optimizer.load_state_dict(self.check_point['optimizer_state'])
        else:
            start_epoch = 1
            val_best = 1
        for e in range(start_epoch, self.epoch + 1):
            self.train_(train_loader=train_loader, optimizer=optimizer, e=e)
            val_loss = self.val_(val_loader=val_loader, e=e)
            state = {
                'model_state': self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                "epoch": e,
                "loss": val_loss,
                'time': self.date
            }

            if val_loss < val_best:
                val_best = val_loss
                if not os.path.exists('model_pth'):
                    os.makedirs('model_pth')
                torch.save(state, 'model_pth/best1.pth')
        self.writer.close()


if __name__ == "__main__":
    Train = TrainApp(weigh_decay=True)
    Train.main()
