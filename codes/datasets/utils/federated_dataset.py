from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch.optim
from torch.utils.data import TensorDataset
import copy
import random
class FederatedDataset:
    """
    Federated learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_SAMPLES_PER_Class = None
    N_CLASS = None
    Nor_TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loaders = []
        self.test_loader = []
        self.args = args
        self.public_loader = None  # 用于存储 D_pub
        self.client_priors = {}    # 用于存储每个客户端估计的先验 hat_pi_k
    @abstractmethod
    def get_data_loaders(self, selected_domain_list=[]) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone(parti_num, names_list, model_name='') -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        pass

    @staticmethod
    def get_epochs():
        pass


# 它让 DataLoader 以为自己在读一个普通的二分类数据集，实际上它是在读原始数据，并动态替换了标签。
class PUDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, parent_dataset, indices, new_labels,true_labels):
        self.parent_dataset = parent_dataset
        self.indices = indices
        self.new_labels = new_labels # s_labels
        self.true_labels = true_labels # 真实的二分类标签 (测试伪标签用的底牌)
            
    def __len__(self):
        return len(self.indices)
                
    def __getitem__(self, index):
        real_idx = self.indices[index]
        img, _ = self.parent_dataset[real_idx] # 调用原始的 getitem 获取 transform 后的图
        target = self.new_labels[index] # 使用修改后的 PU label
        true_target = self.true_labels[index] # 上帝视角的真实标签
        # 返回 img, target (s), real_idx (可选)
        return img, target,true_target,real_idx

class BinaryTestDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, parent_dataset, new_labels):
            self.parent_dataset = parent_dataset
            self.new_labels = new_labels
            
        def __len__(self):
            return len(self.parent_dataset)
            
        def __getitem__(self, index):
            img, _ = self.parent_dataset[index]
            target = self.new_labels[index]
            return img, target



def partition_pu_loaders(train_dataset: datasets, test_dataset: datasets,
                         setting: FederatedDataset) -> Tuple[list, DataLoader, DataLoader, dict]:
    """
    FedPU 核心划分逻辑: 
    1. 分离 D_pub 
    2. Non-IID 划分 
    3. 二值化 & PU Mask 
    4. 先验估计
    """
    args = setting.args
    seed = getattr(args, 'seed', 0) # 获取传入的 seed，如果没有设则默认为 0
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    n_participants = args.parti_num
    pos_classes = args.pos_class_list
    label_freq = args.label_freq # c

    # ---------------------------------------------------------
    # 1. 提取全量 Label 和 Data (假设 dataset 都有 .data 和 .targets/.labels)
    # 注意：这里需要兼容 CIFAR 等数据集的属性名
    # ---------------------------------------------------------
    if hasattr(train_dataset, 'targets'):
        y_train = np.array(train_dataset.targets)
        # CIFAR 等通常是 uint8 的 numpy array，无需转换
    else:
        y_train = np.array(train_dataset.dataset.targets)
    
    # 获取总索引
    total_idxs = np.arange(len(y_train))
    
    # ---------------------------------------------------------
    # 2. 构建公共探针集 D_pub (从全局随机抽取)
    # ---------------------------------------------------------
    public_size = args.public_size
    # 随机打乱索引
    np.random.shuffle(total_idxs)
    
    public_idxs = total_idxs[:public_size]
    remain_idxs = total_idxs[public_size:]
    
    # 创建 Public Loader
    # 注意：这里需要根据具体的 Dataset 类实现方式微调，这里假设支持 SubsetSampler

    # SubsetRandomSampler：这就像一个过滤器，它告诉 DataLoader：“喂，一会取数据的时候，只许取 public_idxs 列表里记下来的那些牌。”
    public_sampler = SubsetRandomSampler(public_idxs)
    setting.public_loader = DataLoader(train_dataset, batch_size=args.local_batch_size, 
                                       sampler=public_sampler, num_workers=4)

    # ---------------------------------------------------------
    # 3. 联邦 Non-IID 划分 (基于剩余数据)
    # 使用原有的 Dirichelt 逻辑，但仅针对 remain_idxs
    # ---------------------------------------------------------
    # 为了复用逻辑，我们先对 remain_idxs 进行 Dirichelt 划分
    y_remain = y_train[remain_idxs]
    N_remain = len(y_remain)
    min_size = 0  #记录分得数据最少的客户端的数据量
    min_require_size = 10 # 限制：每个客户端至少要有 10 条数据，防止死机
    n_class = setting.N_CLASS # 原始类别数 (如 10)
    
    net_dataidx_map = {} # 存储每个客户端分到的【原始数据索引】结果字典】：key是客户端ID，value是该客户端拥有的数据索引列表

    # --- 复用原有的 Dirichlet 划分逻辑 (稍作修改以适配 remain_idxs) ---
    while min_size < min_require_size:
        # idx_batch 是一个列表的列表，idx_batch[j] 存放第 j 个客户端分
        idx_batch = [[] for _ in range(n_participants)]
        for k in range(n_class):
            # 在 remain_idxs 中找到属于类别 k 的索引
            idx_k = [remain_idxs[i] for i, label in enumerate(y_remain) if label == k]
            np.random.shuffle(idx_k)  # 打乱顺序
            
            beta = args.beta
            if beta == 0:
                # np.array_split 把 idx_k 平均切成 N 份，分给 N 个客户端
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.array_split(idx_k, n_participants))]
            else:
                # 1. 生成比例：生成 N 个随机数，和为 1 (例如 [0.1, 0.8, 0.1])
                proportions = np.random.dirichlet(np.repeat(a=beta, repeats=n_participants))
                proportions = np.array([p * (len(idx_j) < N_remain / n_participants) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            
    for j in range(n_participants):
        net_dataidx_map[j] = idx_batch[j]
    
    # ---------------------------------------------------------
    # 4. 本地二值化 & PU Masking & 先验估计
    # ---------------------------------------------------------
    setting.train_loaders = []
    
    for client_id in range(n_participants):
        client_idxs = net_dataidx_map[client_id]
        client_targets_original = y_train[client_idxs] # 获取该客户端的原始标签
        
        # --- A. 二值化 (Binary Labeling) ---
        # 1 if in pos_class_list else 0
        # np.isin 判断：如果是正类则为 1，否则为 0
        client_labels_binary = np.isin(client_targets_original, pos_classes).astype(int)
        
        # --- B. PU Masking (生成 s 变量) ---
        # s=1 (Labeled Positive): 真实的 Positive 且 被选中 (概率 c)
        # s=0 (Unlabeled): 真实的 Negative 或者 真实的 Positive 但未被选中
        
        # 找出真实正样本的索引 (在 client_idxs 数组中的相对索引)
        true_pos_indices = np.where(client_labels_binary == 1)[0]
        true_neg_indices = np.where(client_labels_binary == 0)[0]
        
        # 对正样本进行采样，保留 label_freq 比例为 1，其余重置为 0
        n_pos = len(true_pos_indices)
        n_labeled = int(n_pos * label_freq)
        
        # 随机选择被标记的正样本
        labeled_pos_choice = np.random.choice(true_pos_indices, n_labeled, replace=False)
        
        # 构建最终的训练标签 s
        s_labels = np.zeros_like(client_labels_binary) # 先全部初始化为 0 (Unlabeled)
        s_labels[labeled_pos_choice] = 1 
        # 注意：这里 s_labels 为 1 的是 Labeled Positive (P)，为 0 的是 Unlabeled (U)
        
        # --- C. 先验估计 (Prior Estimation) ---
        # 公式: hat_pi = |P_k| / (c * N_k)
        # N_k = len(client_idxs)
        # |P_k| = n_labeled
        N_k = len(client_idxs)
        if N_k > 0:
            est_prior = n_labeled / (label_freq * N_k + 1e-10) # 加上 epsilon 防止除零
        else:
            est_prior = 0.0
            
        setting.client_priors[client_id] = est_prior
        
        # --- D. 封装 DataLoader ---
        # 这里有一个难点：原始 Dataset 只有原始 label。
        # 我们需要一种方式将新的 s_labels 喂给模型，或者修改 Dataset 的 __getitem__。
        # 为了通用性，我们这里采用构造一个新的 TensorDataset 或使用 Wrapper。
        # 考虑到内存，如果数据量大，建议用 Wrapper。但 CIFAR10 较小，可以直接构造 Subset 并覆盖 target。
        
        # 简单方案：构造一个只包含当前客户端数据的 Dataset 对象，并替换其 targets 为 s_labels
        # 注意：这需要保证 train_dataset 支持拷贝和修改 targets
        
        # 为了不破坏原始 train_dataset，我们使用索引获取数据，然后重新封装
        # (这种方式在数据量极大时可能耗内存，但对 CIFAR/USPS/Letter 没问题)
        
        # 从原始 dataset 获取 transform 后的数据可能比较麻烦，
        # 最稳妥的方法是自定义一个 PUClientDataset 类，但这涉及更多文件修改。
        # 鉴于 RevisitFL 使用 SubsetRandomSampler，我们可以通过修改 Dataset 的 behavior 来实现，
        # 但最快的方法可能是直接提取数据。
        
        # *临时解决方案*：利用 SubsetRandomSampler 依然加载原始数据，
        # 但是在 Model 的 training loop 中，我们不仅取 data, target，还需要取 index，
        # 然后通过 index 查找我们生成的 s_labels。
        # 或者：在这里直接构建一个新的 DataLoader，里面也是 (data, s_label)
        
        # 推荐方案：为了最小化修改，我们保持 loader 返回 (img, original_target)，
        # 但我们将 s_labels 存入 model 或 setting 中，在训练时通过某种方式获取。
        # *更好的方案*：直接在此处构建一个轻量级的 Dataset 类。
        
        # 这里为了演示，我假设我们可以直接从 dataset 获取数据 tensor (CIFAR10.data 是 numpy)
        # 并在内存中转换。
        
        #client_data = train_dataset.data[client_idxs] # numpy array (N, H, W, C)
        # 需要转成 PIL 或 Tensor 并应用 transform？
        # RevisitFL 的 transform 是在 Dataset.__getitem__ 做的。
        
        # 既然是 "告诉我在现在的基础上如何添加"，我建议采用【Wrapper Dataset】方式：
        client_pu_dataset = PUDatasetWrapper(train_dataset, client_idxs, s_labels,client_labels_binary)
        
        train_loader = DataLoader(client_pu_dataset,
                                  shuffle=True, # 本地训练需要 shuffle
                                  num_workers=2, drop_last=False)
        
        setting.train_loaders.append(train_loader)

    # ---------------------------------------------------------
    # 5. 处理 Test Loader (也需要二值化，但不需要 Mask)
    # ---------------------------------------------------------
    if hasattr(test_dataset, 'targets'):
        y_test = np.array(test_dataset.targets)
    else:
        y_test = np.array(test_dataset.dataset.targets)
        
    y_test_binary = np.isin(y_test, pos_classes).astype(int)
    test_binary_dataset = BinaryTestDatasetWrapper(test_dataset, y_test_binary)
    setting.test_loader = DataLoader(test_binary_dataset,
                                     batch_size=args.local_batch_size, shuffle=False, num_workers=2)

    return setting.train_loaders, setting.test_loader, setting.public_loader, setting.client_priors



def partition_label_skew_loaders(train_dataset: datasets, test_dataset: datasets,
                                 setting: FederatedDataset) -> Tuple[list, DataLoader, dict]:
    n_class = setting.N_CLASS
    n_participants = setting.args.parti_num
    n_class_sample = setting.N_SAMPLES_PER_Class
    min_size = 0
    min_require_size = 10
    if hasattr(train_dataset, 'targets'):
        y_train = train_dataset.targets
    else:
        y_train = train_dataset.dataset.targets
    N = len(y_train)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_participants)]
        for k in range(n_class):
            idx_k = [i for i, j in enumerate(y_train) if j == k]
            np.random.shuffle(idx_k)
            if n_class_sample != None:
                idx_k = idx_k[0:n_class_sample * n_participants]
            beta = setting.args.beta
            if beta == 0:
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.array_split(idx_k, n_participants))]
            else:
                proportions = np.random.dirichlet(np.repeat(a=beta, repeats=n_participants))
                proportions = np.array([p * (len(idx_j) < N / n_participants) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_participants):
        np.random.shuffle(idx_batch[j])
        if n_class_sample != None:
            idx_batch[j] = idx_batch[j][0:n_class_sample * n_class]
        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts = record_net_data_stats(y_train, net_dataidx_map,n_class)

    for j in range(n_participants):
        train_sampler = SubsetRandomSampler(net_dataidx_map[j])
        train_loader = DataLoader(train_dataset,
                                  batch_size=setting.args.local_batch_size, sampler=train_sampler, num_workers=1, drop_last=False)
        setting.train_loaders.append(train_loader)

    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.local_batch_size, shuffle=False, num_workers=1)
    setting.test_loader = test_loader

    return setting.train_loaders, setting.test_loader, net_cls_counts


def record_net_data_stats(y_train, net_dataidx_map, n_class):
    net_cls_counts = {}
    y_train = np.array(y_train)
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp={}
        for i in range(n_class):
            if i in unq:
                tmp[i] = unq_cnt[unq==i][0]
            else:
                tmp[i] = 0
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    print('Data statistics: %s' % str(net_cls_counts))
    # save_data_stat(net_cls_counts)
    return net_cls_counts


def save_data_stat(net_cls_counts):
    path = 'datastat.csv'
    with open(path, 'w') as f:
        for k1 in net_cls_counts:
            data = net_cls_counts[k1]
            out_str = ''
            for k2 in data:
                out_str += str(data[k2]) + ','
            out_str += '\n'
            f.write(out_str)