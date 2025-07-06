import numpy as np
import pandas as pd
import os


import numpy as np

def mask_to_rle(mask):
    """
    将 0/1 掩码图转为 RLE 格式
    参数:
        mask: 2D numpy array, binary mask (0 for background, 1 for foreground)
    返回:
        RLE 字符串
    """
    pixels = mask.flatten()  # 默认 order='C'，行优先，与你给出的代码一致
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))



def save_submission_csv(id_list, mask_list, save_path='submission.csv'):
    """
    保存提交的 CSV 文件
    参数:
        id_list: 图像编号列表，如 ['00001', '00002', ...]
        mask_list: 每个图像的预测掩码，0/1 二值 numpy 数组
        save_path: CSV 保存路径
    """
    rle_list = [mask_to_rle(mask) for mask in mask_list]
    df = pd.DataFrame({'ID': id_list, 'Mask': rle_list})
    df.to_csv(save_path, index=False)
    print(f"[✓] submission.csv saved to: {save_path}")


