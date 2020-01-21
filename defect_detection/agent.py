import torch
import torch.nn
import torchvision
from model import SegmentNet
from model import decisionNet
from data_manager import datagenerator



class Agent(object):
    def __init__(self, param):
        self.__Param = param
        self.init_datasets()
        self.model_seg = SegmentNet()
        self.model_dec = decisionNet()

    def run(self):
        if self.__Param['mode'] is 'training':
            train_mode = self.__Param['train_mode']
            self.train(train_mode)
        elif self.__Param['mode'] is 'testing':
            self.test()