import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets.utils.federated_dataset import FederatedDataset, partition_pu_loaders
from backbone.ResNet import resnet18
from datasets.transforms.denormalization import DeNormalize
import os
import torch.nn as nn

def replace_bn_with_gn(module):
    """
    递归遍历模型，将所有的 BatchNorm2d 替换为 GroupNorm。
    这是联邦视觉任务中解决 Non-IID 统计量坍缩的标准做法。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            # ResNet 的通道数通常是 16 的倍数，32 是最常用的 group 数量
            num_groups = 32 if num_features >= 32 else num_features
            # 替换该层
            setattr(module, name, nn.GroupNorm(num_groups, num_features))
        else:
            # 递归子模块
            replace_bn_with_gn(child)
    return module


class FedPUCIFAR10(FederatedDataset):
    NAME = 'fedpu_cifar10'
    SETTING = 'pu_learning'
    N_CLASS = 10  # PU Learning 输出维度为 1 (Logits)
    
    # CIFAR-10 标准均值和方差
    NORM_MEAN = (0.4914, 0.4822, 0.4465)
    NORM_STD = (0.2470, 0.2435, 0.2615)

    # 标准化 Transform
    Nor_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    def get_data_loaders(self, train_transform=None):
        # 1. 设置 Transform
        if train_transform is None:
            train_transform = self.Nor_TRANSFORM
        
        test_transform = transforms.Compose([
            transforms.ToTensor(), 
            self.get_normalization_transform()
        ])

        # 2. 指定数据路径 (根据你的服务器路径)
        # 建议：虽然 utils/conf.py 有 data_path()，但为了保险，这里优先使用你的具体路径
        data_root = os.path.expanduser('~/chicken/FL_PU/datasets')

        # 3. 加载原始数据
        train_dataset = CIFAR10(root=data_root, train=True, 
                                download=True, transform=train_transform)
        
        test_dataset = CIFAR10(root=data_root, train=False, 
                               download=True, transform=test_transform)

        # 4. 调用 FedPU 核心划分逻辑 (由 datasets/utils/federated_dataset.py 提供)
        # 返回: 本地训练集列表, 测试集, 公共探针集, 客户端先验字典
        traindls, testdl, public_dl, client_priors = partition_pu_loaders(train_dataset, test_dataset, self)
        
        return traindls, testdl, public_dl, client_priors

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            FedPUCIFAR10.Nor_TRANSFORM
        ])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        """
        为每个客户端实例化模型
        注意: num_classes=1 (PU Learning)
        """
        nets_list = []
        for j in range(parti_num):
            # 使用 ResNet18 作为基准骨架
            net = resnet18(num_classes=1)
            net = replace_bn_with_gn(net)
            nets_list.append(net)
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(FedPUCIFAR10.NORM_MEAN, FedPUCIFAR10.NORM_STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(FedPUCIFAR10.NORM_MEAN, FedPUCIFAR10.NORM_STD)
        return transform