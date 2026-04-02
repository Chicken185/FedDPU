import torch
import torch.nn as nn
import torch.optim as optim
import copy
from models.utils.federated_model import FederatedModel

class MeanTeacherFedAvg(FederatedModel):
    """
    Baseline 8: Mean Teacher + FedAvg (PU Adaptation)
    - 师生双轨制：学生网络接收梯度更新，老师网络通过 EMA (指数移动平均) 跟踪学生。
    - P 集合：学生网络计算 BCE Loss (逼近 1)。
    - U 集合：保留 Naive PU 压制 (逼近 0) 防止模式崩塌；
              同时引入师生一致性 (Consistency Loss)，计算学生(强增强)与老师(弱增强)输出概率的 MSE。
    - 严格无预热：从第 0 轮直接启动师生博弈。
    """
    NAME = 'meanteacher_fedavg' 
    COMPATIBILITY = ['homogeneity']
    def __init__(self, nets_list, args, transform):
        super(MeanTeacherFedAvg, self).__init__(nets_list, args, transform)
        
        # 基础交叉熵损失 (用于 P 和 Naive PU)
        self.criterion_bce = nn.BCEWithLogitsLoss()
        # 师生一致性损失 (MSE)
        self.criterion_mse = nn.MSELoss()
        
        # Mean Teacher 超参数
        self.ema_decay = getattr(self.args, 'mt_ema_decay', 0.999) # 老师的保留率，通常极高
        self.lambda_c = getattr(self.args, 'mt_lambda_c', 1.0)      # 一致性损失的权重
        
        # 核心改造：为每个客户端在本地初始化一个“专属老师网络”
        self.teacher_nets = {i: copy.deepcopy(nets_list[i]) for i in range(self.args.parti_num)}

    def ini(self):
        # 极其重要的起跑线同步机制：必须同时同步学生和老师！
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for i in range(self.args.parti_num):
            self.nets_list[i].load_state_dict(global_w)
            self.teacher_nets[i].load_state_dict(global_w) # 老师也同步到最新全局状态

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        total_loss = 0.0
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            total_loss += client_loss 

        # 联邦聚合：依然只聚合学生网络 (Student)，老师网络(Teacher)留在本地维持历史记忆
        self.aggregate_nets(None)
            
        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss 

    def _train_net(self, index, net, train_loader):
        # 将当前客户端的学生和老师网络搬到显存
        net = net.to(self.device)
        teacher_net = self.teacher_nets[index].to(self.device)
        teacher_net.load_state_dict(net.state_dict())
        net.train()
        # 极其关键：老师网络永远处于 eval 模式 (关闭 Dropout 和 BatchNorm 的运行状态更新)
        teacher_net.eval()
        
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        avg_loss = 0.0 
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0
            
            # 兼容 3 元素的 DataLoader (indices, images, labels)
            for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                # ==========================================
                # 动态张量级强增强 (用于喂给学生网络)
                if images.dim() == 2:
                    # 表格数据：随机噪声 + 神经元 Dropout
                    noise = torch.randn_like(images) * 0.05
                    feat_mask = (torch.rand_like(images) > 0.2).float()
                    strong_images = (images + noise) * feat_mask
                else:
                    # 图像数据：通道级抹除 + 噪声
                    dropout_mask = (torch.rand_like(images) > 0.2).float()
                    noise = torch.randn_like(images) * 0.1
                    strong_images = (images * dropout_mask) + noise
                # ==========================================
                
                optimizer.zero_grad()
                
                # 1. 老师网络前向传播 (看弱增强原图，绝对禁止梯度)
                with torch.no_grad():
                    teacher_outputs = teacher_net(images).view(-1)
                    teacher_probs = torch.sigmoid(teacher_outputs)
                
                # 2. 学生网络前向传播 (看强增强图片，开启梯度)
                student_outputs = net(strong_images).view(-1)
                student_probs = torch.sigmoid(student_outputs)
                
                labels = labels.view(-1)
                mask_p = (labels == 1)
                mask_u = (labels == 0)
                
                loss_p = torch.tensor(0.0).to(self.device)
                loss_u = torch.tensor(0.0).to(self.device)
                loss_cons = torch.tensor(0.0).to(self.device)
                
                # --- P 集合损失 ---
                if mask_p.sum() > 0:
                    loss_p = self.criterion_bce(student_outputs[mask_p], torch.ones_like(student_outputs[mask_p]))
                    
                # --- U 集合损失 ---
                if mask_u.sum() > 0:
                    # 基础压制：Naive PU 假设，防止模型产生全部预测为 1 的“模式崩塌”
                    loss_u = self.criterion_bce(student_outputs[mask_u], torch.zeros_like(student_outputs[mask_u]))
                    
                    # 师生一致性：让学生(强增强)逼近老师(弱增强)的软概率
                    loss_cons = self.criterion_mse(student_probs[mask_u], teacher_probs[mask_u].detach())
                
                # 3. 混合总损失
                loss = loss_p + loss_u + self.lambda_c * loss_cons

                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                    # ==========================================
                    # 4. 极其关键的 EMA 教师更新
                    # 在每一次 SGD step 之后，利用学生的最新参数，平滑更新老师
                    with torch.no_grad():
                        for param_t, param_s in zip(teacher_net.parameters(), net.parameters()):
                            # 公式: theta_t = alpha * theta_t + (1 - alpha) * theta_s
                            param_t.data.mul_(self.ema_decay).add_(param_s.data, alpha=1.0 - self.ema_decay)

            
                        for buf_t, buf_s in zip(teacher_net.buffers(), net.buffers()):
                            # running_mean/var 是浮点数，可以做 EMA
                            if buf_t.is_floating_point():
                                buf_t.data.mul_(self.ema_decay).add_(buf_s.data, alpha=1.0 - self.ema_decay)
                            # num_batches_tracked 是整数 (LongTensor)，不支持 EMA 的浮点运算，直接硬拷贝
                            else:
                                buf_t.data.copy_(buf_s.data)
                    # ==========================================
                
            avg_loss += epoch_loss / max(1, len(train_loader)) 
            
        # 本地更新结束时，将更新好的老师网络存回字典
        self.teacher_nets[index] = teacher_net.cpu()
            
        return avg_loss / self.local_epoch