import torch
import torch.nn as nn
import torch.optim as optim
import copy
from models.utils.federated_model import FederatedModel

class DistPUFedAvg(FederatedModel):
    """
    Baseline 6: Dist-PU + FedAvg
    结合标签分布视角的 PU 学习 (Dist-PU) 与标准的联邦平均。
    预热期：Naive PU (BCE Loss)。
    正式期：监督损失 L_p + 分布对齐损失 L_dist + 熵极小化损失 L_ent。
    """
    NAME = 'distpu_fedavg' 
    COMPATIBILITY = ['homogeneity']
    def __init__(self, nets_list, args, transform):
        super(DistPUFedAvg, self).__init__(nets_list, args, transform)
        # 用于计算正样本 P 的监督损失
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 提取 Dist-PU 的两个核心超参数 (带有安全默认值)
        self.lambda_dist = 10
        self.lambda_ent = 1

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
        
        # 获取当前客户端的正样本先验比例 pi
        prior = self.client_priors.get(index, getattr(self.args, 'label_freq', 0.2))
        
        avg_loss = 0.0 
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                # 前向传播并计算概率
                outputs = net(images).view(-1)
                labels = labels.view(-1)
                probs = torch.sigmoid(outputs)
                
                # 如果在预热期，直接使用 Naive PU
                if self.epoch_index < -1:
                    loss = self.criterion(outputs, labels)
                else:
                    # ==========================================
                    # Phase 2: Dist-PU 核心逻辑
                    # ==========================================
                    mask_p = (labels == 1)
                    mask_u = (labels == 0)
                    
                    loss_p = torch.tensor(0.0).to(self.device)
                    loss_dist = torch.tensor(0.0).to(self.device)
                    loss_ent = torch.tensor(0.0).to(self.device)
                    
                    # 1. 监督损失 L_p (仅对 P 集合计算)
                    if mask_p.sum() > 0:
                        outputs_p = outputs[mask_p]
                        targets_p = torch.ones_like(outputs_p)
                        loss_p = self.criterion(outputs_p, targets_p)
                        
                    # 2. 对 U 集合计算 L_dist 和 L_ent
                    if mask_u.sum() > 0:
                        probs_u = probs[mask_u]
                        
                        # L_dist: 分布对齐 (MSE 逼近先验 pi)
                        mean_prob_u = probs_u.mean()
                        loss_dist = torch.square(mean_prob_u - prior)
                        
                        # L_ent: 熵极小化 (强制模型非黑即白，避免全是模棱两可的概率)
                        # 加入 1e-7 防止 log(0) 导致 NaN 梯度爆炸
                        eps = 1e-7
                        entropy = - (probs_u * torch.log(probs_u + eps) + (1.0 - probs_u) * torch.log(1.0 - probs_u + eps)).mean()
                        loss_ent = entropy
                        
                    # 3. 组合总损失
                    loss = loss_p + self.lambda_dist * loss_dist + self.lambda_ent * loss_ent

                # 反向传播
                if loss.requires_grad:
                    loss.backward()
                    # 依然保留极其重要的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            
        return avg_loss