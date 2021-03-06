import torch
import torch.nn as nn
import numpy as np
import torchvision
import random
import cv2
from config import *
from model import SegmentNet
from model import decisionNet
from torch.utils.data import DataLoader,TensorDataset
from torch.autograd import Variable
from data_manager import datagenerator
from data_manager import AllDataset

# DefaultParam = {
#     'mode': 'testing',
#     'train_mode': 'decision',
#     'epochs_num': 50,
#     'batch_size': 1,
#     'learn_rate': 0.001,
#     'momentum': 0.9,
#     'data_dir':'dataset/KolektorSDD',
#     'checkPoint_dir': 'checkpoint',
#     'Log_dir': 'log',
#     'valid_ratio': 0,
#     'valid_frequency' :3,
#     'save_frequenchy': 2,
#     'max_to_keep':10,
#     'b_restore':True,
#     'b_saveNG':True,
# }

epochs_num = 50

# 生成所有数据
positive,negative,positive_label,negative_label = datagenerator()

model_segment = SegmentNet().cuda()

# Loss and optimizer
criterion_segment = nn.MSELoss()
optimizer_segment = torch.optim.Adam(model_segment.parameters(), lr=0.001, betas=(0.5,0.99))

# print(np.array(input).shape)

#################################
#segment-net
for i in range(epochs_num):
    # 负样本/标签 提取（30/347）
    a = [random.randint(0, 346) for _ in range(30)]
    negative_random = []
    negative_random_label = []
    for num in a:
        negative_random.append(negative[num])
        tem = cv2.resize(negative_label[num], (int(IMAGE_SIZE[1] / 8), int(IMAGE_SIZE[0] / 8)))
        negative_random_label.append(tem)
    # 标签 = 输入/8
    for i in range(len(positive_label)):
        tem = cv2.resize(positive_label[i], (int(IMAGE_SIZE[1] / 8), int(IMAGE_SIZE[0] / 8)))
        positive_label[i] = tem

    input = []
    label = []
    # 整合 一正一负
    for i in range(30):
        input.append(negative_random[i])
        input.append(positive[i])
        label.append(negative_random_label[i])
        label.append(positive_label[i])

    # input = input[np.newaxis, :, :]
    # label = label[np.newaxis, :, :]
    input = torch.from_numpy(np.array(input))
    label = torch.from_numpy(np.array(label))
    input = torch.unsqueeze(input,1)
    label = torch.unsqueeze(label,1)
    dataset = TensorDataset(input,label)



    Train_Loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False)
    #.type(torch.FloatTensor)
    for i,(inputs, labels) in enumerate(Train_Loader):
        # inputs, labels = Variable(inputs),Variable(labels)
        # print("inputs: ",inputs.data.size(), "labels: ",labels.data.size())
        inputs = inputs.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.FloatTensor).cuda()
        outputs,copy_feature = model_segment(inputs)
        loss = criterion_segment(outputs, labels)
        print(i)
    #
    #     # Backward and optimize
        optimizer_segment.zero_grad()
        loss.backward()
        optimizer_segment.step()
        print(loss.item())

    #     pass

# state = model_segment.state_dict()
torch.save(model_segment.state_dict(), 'model_segment.pth')

