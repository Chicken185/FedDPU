import torch
import torch.nn as nn
import torch.optim as optim
import copy
from models.utils.federated_model import FederatedModel
# 务必确保从你存放 loss 的地方正确导入 nnPULoss
from models.utils.losses import nnPULoss 

class nnPUFedAvg(FederatedModel):
    """
    Baseline 3: nnPU + FedAvg
    在本地客户端使用无偏风险估计和非负截断的 nnPULoss 进行训练，
    以解决 U 集合中隐藏正样本带来的偏差，并防止负风险过拟合。
    全局采用标准的 FedAvg 进行参数聚合。
    """
    NAME = 'nnpu_fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(nnPUFedAvg, self).__init__(nets_list, args, transform)
        # 注意：这里我们不在 __init__ 中直接实例化 self.criterion。
        # 因为 nnPULoss 需要针对每个客户端的 prior (先验 pi) 进行动态实例化。

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        total_loss = 0.0
        
        # 1. 客户端本地训练
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            total_loss += client_loss 

        # 2. 全局聚合 (标准的 FedAvg，按数据量均分权重)
        self.aggregate_nets(None)
            
        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss 

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        # ==========================================
        # 【核心差异 1】：动态获取客户端的先验 pi
        # ==========================================
        # client_priors 是我们在 datasets 层 partition_pu_loaders 时算好并挂载到 model 上的字典。
        # 如果获取不到，默认给一个 0.5 防止报错。
        prior = getattr(self, 'client_priors', {}).get(index, 0.5) 
        
        # 为当前客户端专属实例化 nnPULoss
        criterion = nnPULoss(prior=prior, beta=0.0, gamma=1.0)
        
        avg_loss = 0.0 
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                # 纯净的前向传播 (绝不使用 torch.cat，保护 BatchNorm)
                outputs = net(images).view(-1)
                labels = labels.view(-1)
                
                # ==========================================
                # 【核心差异 2】：调用 nnPULoss
                # ==========================================
                # 我们之前修改好的 nnPULoss 内部已经集成了 U' = U ∪ P 的机制，
                # 所以直接把原始的 outputs 和 labels 喂进去即可。
                loss = criterion(outputs, labels)

                loss.backward()
                
                # 必备安全阀：梯度裁剪
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            print(f"[Client {index}] Local Epoch {local_epoch+1}/{self.local_epoch} | Loss: {avg_loss:.4f}")
            
        return avg_loss