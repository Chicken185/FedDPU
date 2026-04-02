import torch
import torch.nn as nn
import torch.optim as optim
import copy
from models.utils.federated_model import FederatedModel

class NaiveFedProx(FederatedModel):
    NAME = 'naive_fedprox' 
    COMPATIBILITY = ['homogeneity']
    # 修复 1：严格对齐接口参数
    def __init__(self, nets_list, args, transform):
        super(NaiveFedProx, self).__init__(nets_list, args, transform)
        self.criterion = nn.BCEWithLogitsLoss()
        self.mu = getattr(self.args, 'mu', 0.1)

    # 修复 2：加入必须的起跑线同步机制
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
        
        global_weight_collector = [param.detach().clone() for param in self.global_net.to(self.device).parameters()]
        
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i], global_weight_collector)
            total_loss += client_loss 

        self.aggregate_nets(None)
            
        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss 

    def _train_net(self, index, net, train_loader, global_weight_collector):
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
                outputs = net(images).view(-1)
                labels = labels.view(-1)
                
                loss_bce = self.criterion(outputs, labels)
                
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_weight_collector):
                    proximal_term += torch.square(local_param - global_param).sum()
                
                loss = loss_bce + (self.mu / 2.0) * proximal_term

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            
        return avg_loss