import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torchvision.transforms as transforms
from models.utils.federated_model import FederatedModel

class FixMatchFedAvg(FederatedModel):
    """
    Baseline 4: FixMatch + FedAvg
    结合半监督学习经典算法 FixMatch 和标准联邦平均 FedAvg。
    对于标记数据 (P)：直接计算监督 BCE Loss。
    对于无标记数据 (U)：弱增强过模型得伪标签，若置信度满足阈值 (默认 >0.95 或 <0.05)，
    则对强增强后的图片计算 BCE Loss。
    """
    NAME = 'fixmatch_fedavg' 
    COMPATIBILITY = ['homogeneity']
    def __init__(self, nets_list, args, transform):
        super(FixMatchFedAvg, self).__init__(nets_list, args, transform)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # FixMatch 核心超参数：高低阈值与无监督损失权重
        self.tau_pos = getattr(self.args, 'tau_pos', 0.8)
        self.tau_neg = getattr(self.args, 'tau_neg', 0.2)
        self.lambda_u = getattr(self.args, 'lambda_u', 1.0)
        
        # 定义实时的 Tensor 强增强 (模拟 FixMatch 的 RandAugment)
        # RandomErasing 能直接作用于已归一化的 Tensor，是非常好的强增强平替
        self.strong_aug = transforms.Compose([
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.2)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random')
        ])

    def ini(self):
        # 必须的起跑线同步机制
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        total_loss = 0.0
        
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            total_loss += client_loss 

        # 全局聚合：使用标准 FedAvg
        self.aggregate_nets(None)
            
        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss 

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        avg_loss = 0.0 
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                # --- 1. 数据分离 ---
                # P 集合 (labels == 1)，U 集合 (labels == 0)
                mask_p = (labels == 1)
                mask_u = (labels == 0)
                
                images_p = images[mask_p]
                images_u = images[mask_u]
                
                loss_p = torch.tensor(0.0).to(self.device)
                loss_u = torch.tensor(0.0).to(self.device)
                
                # --- 2. 计算 P 集合的监督 Loss ---
                if images_p.size(0) > 0:
                    outputs_p = net(images_p).view(-1)
                    # 目标值强制为 1
                    targets_p = torch.ones_like(outputs_p)
                    loss_p = self.criterion(outputs_p, targets_p)
                    
                # --- 3. 计算 U 集合的 FixMatch Loss ---
                if images_u.size(0) > 0:
                    # 3.1 弱增强过模型获取概率 (images 本身已经经过数据集的普通增强，视作弱增强)
                    with torch.no_grad():
                        weak_outputs = net(images_u).view(-1)
                        probs_u = torch.sigmoid(weak_outputs)
                    
                    # 3.2 筛选高置信度样本并生成硬伪标签 (Hard Pseudo-labels)
                    pos_mask = probs_u >= self.tau_pos
                    neg_mask = probs_u <= self.tau_neg
                    valid_mask = pos_mask | neg_mask
                    
                    if valid_mask.sum() > 0:
                        # 构造伪标签: 置信度高的设为 1，置信度低的设为 0
                        pseudo_targets = torch.ones_like(probs_u)
                        pseudo_targets[neg_mask] = 0.0
                        
                        # 只取有效的样本
                        valid_images_u = images_u[valid_mask]
                        valid_targets_u = pseudo_targets[valid_mask]
                        if valid_images_u.dim() == 2:
                            # 🚨 触发表格数据兼容模式 (Tabular Data)
                            # 强增强策略：随机高斯噪声 + 随机抹除 20% 的特征节点
                            noise = torch.randn_like(valid_images_u) * 0.05
                            mask = (torch.rand_like(valid_images_u) > 0.2).float()
                            strong_images_u = (valid_images_u + noise) * mask
                        # 3.3 对有效的 U 样本进行强增强
                        else: strong_images_u = self.strong_aug(valid_images_u)
                        
                        # 3.4 强增强过模型计算 Loss
                        strong_outputs = net(strong_images_u).view(-1)
                        loss_u = self.criterion(strong_outputs, valid_targets_u)
                
                # --- 4. 损失组合与反向传播 ---
                # FixMatch 总 Loss = 监督 Loss + lambda_u * 无监督 Loss
                loss = loss_p + self.lambda_u * loss_u

                # 防止整个 Batch 全是无效 U 样本导致 loss 无法 backward
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            
        return avg_loss