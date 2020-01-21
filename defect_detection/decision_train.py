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

epochs_num = 1

model_segment = SegmentNet().cuda()
model_segment.load_state_dict(torch.load( 'model_segment.pth'))
model_segment.eval()
# print(model_segment)


#################################
#decision-net

model_decision = decisionNet().cuda()

# Loss and optimizer
criterion_decision = nn.MSELoss()
optimizer_decision = torch.optim.Adam(model_decision.parameters(), lr=0.001, betas=(0.5,0.99))

label_decision = np.array([1,0] * 30)
label_decision = torch.from_numpy(label_decision).type(torch.FloatTensor).cuda()

#将segment_net 的对应数据集带入训练好的网络，是训练好的，（不用再训练了）
#因为数据随机性，好要继续随机提取
positive,negative,positive_label,negative_label = datagenerator()
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

    input = torch.from_numpy(np.array(input))
    label = torch.from_numpy(np.array(label))
    input = torch.unsqueeze(input,1)
    label = torch.unsqueeze(label,1)
    dataset = TensorDataset(input,label)

    Train_Loader = DataLoader(dataset=dataset, num_workers=0, batch_size=1, shuffle=False)
    for i, (inputs, labels) in enumerate(Train_Loader):
        inputs = inputs.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.FloatTensor).cuda()
        outputs, copy_feature = model_segment(inputs)

        # decision-net
        outputs = model_decision(copy_feature, outputs)
        loss = criterion_decision(outputs, label_decision[i])

        # Backward and optimize
        optimizer_decision.zero_grad()
        loss.backward()
        optimizer_decision.step()
        print(loss.item())
