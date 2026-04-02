import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
from datasets.utils.federated_dataset import FederatedDataset, partition_pu_loaders
from backbone.MLP import MLP
# 这里的 DeNormalize 可能不适用 1D 数据，但为了接口兼容先保留引用，如果不使用可以忽略
from datasets.transforms.denormalization import DeNormalize
class TabularNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std


class LetterDataset(data.Dataset):
    """
    UCI Letter Recognition 数据集
    格式: Letter, x1, x2, ..., x16 (20000行)
    """
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        
        file_path = os.path.join(root, 'letter-recognition.data')
        if not os.path.exists(file_path):
             raise RuntimeError(f"Dataset not found at {file_path}")
        
        # 读取数据 (无 header)
        df = pd.read_csv(file_path, header=None)
        
        # 第一列是 Label (A-Z)，转换为 0-25
        df[0] = df[0].apply(lambda x: ord(x) - 65)
        
        # 提取数据和标签
        all_data = df.iloc[:, 1:].values.astype(np.float32)
        all_targets = df.iloc[:, 0].values.astype(np.int64)
        
        # 手动划分 Train/Test (16000 / 4000)
        train_size = 16000
        if self.train:
            self.data = all_data[:train_size]
            self.targets = all_targets[:train_size]
        else:
            self.data = all_data[train_size:]
            self.targets = all_targets[train_size:]

    def __getitem__(self, index):
        # 转换为 Tensor
        feat = torch.from_numpy(self.data[index]) # Shape: (16,)
        target = int(self.targets[index])
        
        if self.transform is not None:
            feat = self.transform(feat)
            
        return feat, target

    def __len__(self):
        return len(self.data)

class FedPULetter(FederatedDataset):
    NAME = 'fedpu_letter'
    SETTING = 'pu_learning'
    N_CLASS = 26 # PU Logits
    
    # 特征值约为 0-15，简单设定均值为 8，标准差为 4 进行归一化
    NORM_MEAN = (8.0,) * 16
    NORM_STD = (4.0,) * 16
    
    def get_data_loaders(self, train_transform=None):
        # 如果没有指定 transform，使用默认的归一化
        if train_transform is None:
             train_transform = self.get_normalization_transform()

        data_root = os.path.expanduser('~/chicken/FL_PU/datasets')
        
        # 加载数据
        train_dataset = LetterDataset(root=data_root, train=True, transform=train_transform)
        # 测试集使用相同的 transform (即归一化)
        test_dataset = LetterDataset(root=data_root, train=False, transform=train_transform)
        
        # 联邦 PU 划分
        traindls, testdl, public_dl, client_priors = partition_pu_loaders(train_dataset, test_dataset, self)
        
        return traindls, testdl, public_dl, client_priors

    @staticmethod
    def get_transform():
        return None 

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        """
        使用 MLP Backbone
        """
        nets_list = []
        for j in range(parti_num):
            # Input: 16, Hidden: [64, 32], Output: 1
            net = MLP(input_dim=16, hidden_dims=[64, 32], output_dim=1)
            nets_list.append(net)
        return nets_list

    @staticmethod
    def get_normalization_transform():
        # 1D Tensor 的归一化
        return TabularNormalize(FedPULetter.NORM_MEAN, FedPULetter.NORM_STD)
    
    @staticmethod
    def get_denormalization_transform():
        # 表格数据一般不需要反归一化可视化，返回 None 或 Identity
        return None