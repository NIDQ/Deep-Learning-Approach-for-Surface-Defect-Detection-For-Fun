import torch
import torch.nn as nn
import torchvision



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# class Model(object):
#     def __init__(self,param):
#         self.step =0
#         self.is_training = True
#         self.__learn_rate = param["learn_rate"]
#         self.__learn_rate = param["learn_rate"]
#         self.__max_to_keep = param["max_to_keep"]
#         self.__checkPoint_dir = param["checkPoint_dir"]
#         self.__restore = param["b_restore"]
#         self.__mode = param["mode"]
#         self.is_training = True
#         self.__batch_size = param["batch_size"]






class SegmentNet(nn.Module):
    def __init__(self):
        super(SegmentNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 1024, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out5,out4



class decisionNet(nn.Module):
    def __init__(self):
        super(decisionNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1025, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(66,1,bias=False),
            nn.Sigmoid()
        )
        self.max32 = nn.AdaptiveMaxPool2d((1, 1))
        self.max1 = nn.AdaptiveMaxPool2d((1, 1))
        self.avg32 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, feature, seg_output):
        out = torch.cat([feature, seg_output], dim=1)
        out = self.layer1(out)
        # print(out.shape)
        out1 = self.avg32(out)
        # print(out1.shape)
        out2 = self.max32(out)
        out3 = self.avg1(seg_output)
        out4 = self.max1(seg_output)
        output = torch.cat([out1,out2,out3,out4], dim=1)
        output = torch.squeeze(output,3)
        output = torch.squeeze(output,2)
        # print(output.shape)
        output = self.fc(output)
        return output