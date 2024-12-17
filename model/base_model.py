import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def set_device(self, x): #将输入数据 x 转移到指定的设备（self.device）
        # 字典（dict）：递归处理字典中的每个值。
        # 列表（list）：递归处理列表中的每个元素。
        # 单一对象：直接转移到设备。如tensor
        if isinstance(x, dict):
            for key, item in x.items(): #值不为0就转移到该设备上
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module #如果是分布式训练需要剥离包装得到模型module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))##计算所有参数数量
        return s, n
