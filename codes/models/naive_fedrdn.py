import torch
import torch.nn as nn
import torch.optim as optim
import copy
import random
from models.utils.federated_model import FederatedModel

class NaiveFedRDN(FederatedModel):
    """
    Baseline 5: Naive PU + FedRDN (Federated Random Data Normalization)
    预热期和正式期均使用 Naive PU (未标记数据视为负类 0，标记数据视为正类 1)。
    在联邦学习层面：
    1. 每轮收集各个客户端数据的均值(Mean)和标准差(Std)，构建“联邦统计特征池”。
    2. 客户端在本地训练时，随机抽取池中的全局特征强行注入到本地 Batch 中 (解决 Covariate Shift)。
    3. 最后使用标准的 FedAvg 进行参数聚合。
    """
    NAME = 'naive_fedrdn' 
    COMPATIBILITY = ['homogeneity']
    def __init__(self, nets_list, args, transform):
        super(NaiveFedRDN, self).__init__(nets_list, args, transform)
        self.criterion = nn.BCEWithLogitsLoss()

    def ini(self):
        # 必须的起跑线同步机制
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def _collect_global_stats(self, priloader_list, online_clients):
        """
        核心机制 1：构建联邦统计特征池
        从当前在线的客户端中，快速抽取一个 Batch 的数据计算通道统计量
        """
        pool = []
        for idx in online_clients:
            dl = priloader_list[idx]
            try:
                # 快速抽取第一个 Batch
                images, _ = next(iter(dl))
                images = images.to(self.device)
                
                # 兼容 4D 图像数据 (B, C, H, W) 和 2D 表格数据 (B, D)
                if images.dim() == 4:
                    mu = images.mean(dim=[0, 2, 3], keepdim=True)
                    std = images.std(dim=[0, 2, 3],unbiased=False, keepdim=True) + 1e-6
                elif images.dim() == 2:
                    mu = images.mean(dim=[0], keepdim=True)
                    std = images.std(dim=[0],unbiased=False, keepdim=True) + 1e-6
                else:
                    continue
                    
                pool.append((mu.detach(), std.detach()))
            except StopIteration:
                continue
        return pool

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        # 1. 聚合前收集本轮的全局统计特征池
        global_stats_pool = self._collect_global_stats(priloader_list, online_clients)

        total_loss = 0.0
        
        # 2. 客户端本地训练 (传入特征池)
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i], global_stats_pool)
            total_loss += client_loss 

        # 3. 全局聚合：使用标准 FedAvg
        self.aggregate_nets(None)
            
        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss 

    def _train_net(self, index, net, train_loader, global_stats_pool):
        net = net.to(self.device)
        net.train()
        
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        avg_loss = 0.0 
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                # ==========================================
                # 核心机制 2：实时张量级特征注入 (FedRDN)
                # 以 50% 的概率应用全局特征注入，保留一部分本地原始分布
                if len(global_stats_pool) > 0 and random.random() > 0.5:
                    target_mu, target_std = random.choice(global_stats_pool)
                    
                    if images.dim() == 4:
                        batch_mu = images.mean(dim=[0, 2, 3], keepdim=True)
                        batch_std = images.std(dim=[0, 2, 3],  unbiased=False,keepdim=True) + 1e-6
                        # AdaIN 数学公式：抹除本地统计量，注入抽取的全局统计量
                        images = ((images - batch_mu) / batch_std) * target_std + target_mu
                    elif images.dim() == 2:
                        batch_mu = images.mean(dim=[0], keepdim=True)
                        batch_std = images.std(dim=[0],  unbiased=False,keepdim=True) + 1e-6
                        # AdaIN 数学公式：抹除本地统计量，注入抽取的全局统计量
                        images = ((images - batch_mu) / batch_std) * target_std + target_mu
                # ==========================================
                
                optimizer.zero_grad()
                
                outputs = net(images).view(-1)
                labels = labels.view(-1)
                
                # Naive PU 的核心：强制将 P 视为 1，U 视为 0 算 BCE
                loss = self.criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            
        return avg_loss