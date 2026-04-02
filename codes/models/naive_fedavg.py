import torch
import torch.nn as nn
import torch.optim as optim
from models.utils.federated_model import FederatedModel
import copy
class NaiveFedAvg(FederatedModel):


    NAME = 'naive_fedavg' 
    COMPATIBILITY = ['homogeneity']
    """
    Baseline 1: Naive PU + FedAvg
    将所有未标记数据 (U) 视为负类 (0)，标记数据 (P) 视为正类 (1)，
    使用标准二分类交叉熵 (BCE) 进行纯本地训练，然后使用标准 FedAvg 进行全局聚合。
    """
    def __init__(self, nets_list, args, transform):
        super(NaiveFedAvg, self).__init__(nets_list, args, transform)
        self.criterion = nn.BCEWithLogitsLoss()

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        # 随机挑选在线的客户端
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        total_loss = 0.0
        
        # 1. 客户端本地训练
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            total_loss += client_loss 

        # 2. 全局聚合 
        # 直接调用基类 FederatedModel 的 aggregate_nets(None) 方法
        # 它会自动执行标准的 FedAvg（按数据量加权平均）并下发给所有客户端
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
                # 数据集里 P 已经是 1，U 已经是 0，直接 float() 用于算 BCE
                labels = labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                # 前向传播并展平
                outputs = net(images).view(-1)
                labels = labels.view(-1)
                
                # 计算 Naive PU Loss
                loss = self.criterion(outputs, labels)

                loss.backward()
                
                # 保持我们之前加的优良传统：梯度裁剪防爆，保障训练稳定
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            print(f"[Client {index}] Local Epoch {local_epoch+1}/{self.local_epoch} | Loss: {avg_loss:.4f}")
        return avg_loss