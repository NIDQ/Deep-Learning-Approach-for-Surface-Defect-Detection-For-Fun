import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random

# # b = [1,0] * 15
# b = np.array([1,0] * 30)
# print(b)
# c = torch.from_numpy(b).type(torch.FloatTensor)
# print(len(c))

# target output size of 5x7
# m = nn.AdaptiveAvgPool2d((1,1))
# input = torch.randn(1, 32, 20, 8)
# output = m(input)
# print(output.shape)

# input = torch.randn(1, 66, 1, 1)
# output = torch.squeeze(input,3)
# print(output.shape)
# output = torch.squeeze(output,2)
# print(output.shape)
# model1 = nn.Linear(66,1,bias=False)
# output = model1(output)