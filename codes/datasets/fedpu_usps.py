import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import bz2
import os
from datasets.utils.federated_dataset import FederatedDataset, partition_pu_loaders
from backbone.SimpleCNN import SimpleCNNMNIST
from datasets.transforms.denormalization import DeNormalize

class MyUSPS(data.Dataset):
    """
    自定义 USPS 数据集读取器，用于读取 .bz2 格式的原始数据
    """
# USPS 的原始数据是文本格式。每一行代表一张图：第一个数是 Label (如 1, 2... 10)。后面跟着 256 个浮点数，代表 $16 \times 16$ 个像素点的值。
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        
        # 确定文件名
        filename = 'usps.bz2' if train else 'usps.t.bz2'
        file_path = os.path.join(root, filename)
        
        if not os.path.exists(file_path):
            raise RuntimeError(f"Dataset not found at {file_path}")

        # 读取 bz2 文件
        # 格式: label 1:pixel1 2:pixel2 ... (LIBSVM sparse format or dense)
        # 但标准的 usps.bz2 通常是: label pixel1 pixel2 ... (space separated)
        images = []
        labels = []
        
        with bz2.open(file_path, 'rt') as f:
            for line in f:
                split_line = line.strip().split()
                # 第一列是标签 (1-10 或 0-9，USPS 原始可能是 1-10，需要处理)
                label = int(split_line[0])
                # 将标签 10 转换为 0 (如果有的话)，保证是 0-9
                if label == 10:
                    label = 0
                # 剩余 256 列是像素值 (-1 到 1 或 0 到 1)
                img_vector = np.zeros(256, dtype=np.float32)
                for item in split_line[1:]:
                    idx_str, val_str = item.split(':')
                    # LIBSVM 格式的 index 通常是从 1 开始的，我们需要减 1 变成 0-255 的索引
                    idx = int(idx_str) - 1 
                    img_vector[idx] = float(val_str)
                # Reshape 为 16x16
                img_matrix = img_vector.reshape(16, 16)
                
                images.append(img_matrix)
                labels.append(label)

        self.data = np.array(images)
        self.targets = np.array(labels)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        
        # 转为 Tensor 并增加 Channel 维度 (1, 16, 16) -> (16, 16, 1) or similar for PIL
        # 为了兼容 transforms，最好先转为 PIL 或者保持 numpy (H, W) -> Tensor (1, H, W)
        # 这里我们手动处理为 numpy (H, W, 1) 以便 ToTensor 工作，或者直接转 Tensor
        
        # 由于是灰度图，transform 期望输入通常是 PIL Image 或 Tensor
        # 我们这里直接转 Tensor: (16, 16) -> (1, 16, 16)
        img = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform is not None:
            # 注意: 如果 transform 包含 ToPILImage，输入需要适配。
            # 为了简单，我们假设 transform 处理 Tensor 或 PIL
            # 这里为了配合 Resize(28, 28)，我们使用 torchvision 的 functional 或 Compose
            # 由于数据已经在 Tensor 且是 float，直接 Resize 即可
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

class FedPUUSPS(FederatedDataset):
    NAME = 'fedpu_usps'
    SETTING = 'pu_learning'
    N_CLASS = 10  # PU Learning 输出 1 维 Logits
    
    # USPS 归一化 (基于 -1 到 1 的原始数据或 0-1)
    # 假设原始数据是 -1 到 1，均值约 0，方差约 1。如果是 0-1，则不同。
    # 这里使用经验值，或者简单的 0.5
    NORM_MEAN = (0.5,)
    NORM_STD = (0.5,)

    # 预处理：Resize 到 28x28 以适配 SimpleCNNMNIST
    Nor_TRANSFORM = transforms.Compose([
        transforms.Resize((28, 28)),
        # transforms.ToTensor(), # MyUSPS 已经输出了 Tensor
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    def get_data_loaders(self, train_transform=None):
        if train_transform is None:
            train_transform = self.Nor_TRANSFORM
        
        test_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            self.get_normalization_transform()
        ])

        # 1. 路径
        data_root = os.path.expanduser('~/chicken/FL_PU/datasets')

        # 2. 加载自定义 Dataset
        train_dataset = MyUSPS(root=data_root, train=True, transform=train_transform)
        test_dataset = MyUSPS(root=data_root, train=False, transform=test_transform)

        # 3. 联邦划分
        traindls, testdl, public_dl, client_priors = partition_pu_loaders(train_dataset, test_dataset, self)
        
        return traindls, testdl, public_dl, client_priors

    @staticmethod
    def get_transform():
        # 用于可视化的 transform，这里稍微简化
        return FedPUUSPS.Nor_TRANSFORM

    @staticmethod
    def get_backbone(parti_num, names_list, model_name=''):
        """
        使用 SimpleCNNMNIST (适配 28x28 单通道输入)
        """
        nets_list = []
        for j in range(parti_num):
            # input_dim=16*4*4 是 SimpleCNNMNIST 内部全连接层的输入尺寸 (28x28经过两次池化)
            # hidden_dims=[120, 84] 是经典 LeNet 设置
            # output_dim=1 (PU Logits)
            net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=1)
            nets_list.append(net)
        return nets_list

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(FedPUUSPS.NORM_MEAN, FedPUUSPS.NORM_STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(FedPUUSPS.NORM_MEAN, FedPUUSPS.NORM_STD)
        return transform