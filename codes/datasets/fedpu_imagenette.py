import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
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


class FedPUImageNette(FederatedDataset):
    NAME = 'fedpu_imagenette'
    SETTING = 'pu_learning'
    N_CLASS = 10  # PU Learning 输出 1 维 Logits
    
    # ImageNet 标准均值和方差
    NORM_MEAN = (0.485, 0.456, 0.406)
    NORM_STD = (0.229, 0.224, 0.225)

    # 训练预处理：Resize 64x64 -> Augmentation -> Norm
    Nor_TRANSFORM = transforms.Compose([
        transforms.Resize(256),           
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    def get_data_loaders(self, train_transform=None):
        if train_transform is None:
            train_transform = self.Nor_TRANSFORM
        
        # 测试预处理：Resize 64x64 -> Norm
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            self.get_normalization_transform()
        ])

        # 1. 构建路径
        data_root = os.path.expanduser('~/chicken/FL_PU/datasets/imagenette2')
        train_dir = os.path.join(data_root, 'train')
        val_dir = os.path.join(data_root, 'val')

        # 2. 加载数据 (使用 ImageFolder)
        # ImageFolder 会自动根据子文件夹名称分配 label (0, 1, ..., N)
        train_dataset = ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = ImageFolder(root=val_dir, transform=test_transform)

        # 3. 调用 FedPU 核心划分逻辑
        # partition_pu_loaders 会自动处理 ImageFolder 的 targets 属性
        traindls, testdl, public_dl, client_priors = partition_pu_loaders(train_dataset, test_dataset, self)
        
        return traindls, testdl, public_dl, client_priors

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            FedPUImageNette.Nor_TRANSFORM
        ])
        return transform

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        """
        实例化 ResNet18
        现有 ResNet 代码中的 AdaptiveAvgPool2d((1,1)) 可以自适应 64x64 输入
        """
        nets_list = []
        for j in range(parti_num):
            net = resnet18(num_classes=1)
            net = replace_bn_with_gn(net)
            nets_list.append(net)
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(FedPUImageNette.NORM_MEAN, FedPUImageNette.NORM_STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(FedPUImageNette.NORM_MEAN, FedPUImageNette.NORM_STD)
        return transform