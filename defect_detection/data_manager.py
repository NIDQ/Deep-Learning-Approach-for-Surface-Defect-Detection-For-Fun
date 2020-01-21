import torch
import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from config import *


class AllDataset(Dataset):
    def __init__(self,input,label):
        self.input = input
        self.label = label
    def __getitem__(self, index):
        x = self.input[index]
        y = self.label[index]
        # batch_x = batch_x[np.newaxis,:, :]
        # batch_y = batch_y[np.newaxis, :, :]
        # batch_x = torch.from_numpy(batch_x)
        # batch_y = torch.from_numpy(batch_y)
        return x, y
    def __len__(self):
        return 0


# positive,negative,positive_label,negative_label = datagenerator()
# print(np.array(negative).shape)
# a = [random.randint(0,347) for _ in range(30)]
# negative_random = []
# negative_random_label = []
# for num in a:
#     negative_random.append(negative[num])
#     tem = cv2.resize(negative_label[num],(int(IMAGE_SIZE[1] / 8), int(IMAGE_SIZE[0] / 8)))
#     negative_random_label.append(tem)
# print(np.array(negative_random).shape)
# print(np.array(negative_random_label).shape)





def datagenerator(data_dir='dataset/KolektorSDD/', ):
    dir_list = os.listdir(data_dir)
    input = []
    label = []
    positive = []
    negative = []
    positive_label = []
    negative_label = []
    for dir_name in dir_list:
        input_name = 'dataset/KolektorSDD/' + dir_name + '/*.jpg'
        a = glob.glob(input_name)
        input.append(a)
        label_name = 'dataset/KolektorSDD/' + dir_name + '/*.bmp'
        b = glob.glob(label_name)
        label.append(b)

    for i in range(len(input)):
        tem = POSITIVE_KolektorSDD[i]  # 每组缺陷样本序号
        for j in range(len(input[i])):  #正负样本及对应标签提取
            if j in tem:
                img = cv2.imread(input[i][j], 0)
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                positive.append(img)
                img = cv2.imread(label[i][j], 0)
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                positive_label.append(img)
            else:
                img = cv2.imread(input[i][j], 0)
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                negative.append(img)
                img = cv2.imread(label[i][j], 0)
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
                negative_label.append(img)
    return positive,negative,positive_label,negative_label

