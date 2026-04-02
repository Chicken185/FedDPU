# import torch
# import torch.optim as optim
# import torch.nn as nn
# import numpy as np
# import copy
# from tqdm import tqdm
# from models.utils.federated_model import FederatedModel
# from models.utils.losses import nnPULoss, naive_pu_loss
# from utils.args import *

# def get_parser() -> ArgumentParser:
#     parser = ArgumentParser(description='Federated PU Learning')
#     add_management_args(parser)
#     add_experiment_args(parser)
#     return parser

# class FedPU(FederatedModel):
#     NAME = 'fedpu'
#     COMPATIBILITY = ['homogeneity']

#     def __init__(self, nets_list, args, transform):
#         super(FedPU, self).__init__(nets_list, args, transform)
#         self.public_loader = None # 需在 main.py 或 dataset 初始化后注入
#         self.client_priors = {}   # 需注入
#         self.Twarm = args.Twarm
#         self.weight_balance = args.weight_balance # lambda
#         self.public_feat_global = None # 缓存全局模型的探针特征

#     def ini(self):
#         # 初始化全局模型
#         # 联邦学习的起手式：
#         # 把第 0 个客户端的模型深拷贝一份，当作初始的"全局模型"
#         self.global_net = copy.deepcopy(self.nets_list[0])
#         global_w = self.nets_list[0].state_dict()
#         # 把这个初始权重强制下发给所有客户端，确保大家在同一起跑线
#         for net in self.nets_list:
#             net.load_state_dict(global_w)

#     # 本地的聚合更新阶段
#     def loc_update(self, priloader_list):
#         total_clients = list(range(self.args.parti_num))
#         # 模拟随机选择客户端
#         online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
#         self.online_clients = online_clients

#         # 1. 本地训练
#         for i in online_clients:
#             self._train_net(i, self.nets_list[i], priloader_list[i])

#         # 2. 聚合 (根据阶段选择策略)
#         if self.epoch_index < self.Twarm:
#             # 预热阶段：标准平均
#             self._pseudo_aggregate_for_eval()
#         else:
#             # 正式阶段：一致性感知聚合
#             self._consistency_aware_aggregation()
            
#         return None

#     def _train_net(self, index, net, train_loader):
#         net = net.to(self.device)
#         net.train()
        
#         optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
#         # Phase 1: 预热阶段使用 Naive PU (BCE Loss)
#         criterion_warmup = torch.nn.BCEWithLogitsLoss()
        
#         # Phase 2: 正式阶段使用 MSE Loss (软伪标签蒸馏)
#         criterion_mse = torch.nn.MSELoss()

#         avg_loss = 0.0 
#         for local_epoch in range(self.local_epoch):
#             epoch_loss = 0.0
            
#             for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):
#                 images = images.to(self.device)
#                 labels = labels.to(self.device).float()
                
#                 optimizer.zero_grad()
                
#                 if self.epoch_index < self.Twarm:
#                     # ==========================================
#                     # Phase 1: 纯本地 Naive PU 盲训打底
#                     # 将 U 视为 0，P 视为 1，快速建立初始决策边界
#                     # ==========================================
#                     outputs = net(images).view(-1)
#                     loss = criterion_warmup(outputs, labels)
                    
#                 else:
#                     # ==========================================
#                     # Phase 2: 全局双重一致性模型驱动的软标签 MSE 蒸馏
#                     # ==========================================
                    
#                     # 1. 使用最强的全局教师模型生成软伪标签 (Soft Pseudo-labels)
#                     with torch.no_grad():
#                         self.global_net.eval() # 确保 BatchNorm/Dropout 处于测试模式
#                         global_outputs = self.global_net(images).view(-1)
#                         global_probs = torch.sigmoid(global_outputs)
                    
#                     # 2. 构建融合目标值 (Targets)
#                     # 【核心逻辑】：如果你本来就是真实正样本(labels==1)，我坚决让你保持为 1.0
#                     # 如果你是未标记数据(labels==0)，我听从全局模型的概率打分 (global_probs)
#                     targets = torch.where(labels == 1.0, torch.ones_like(global_probs), global_probs)
                    
#                     # 3. 本地模型前向传播，并转换为概率 (0~1)
#                     local_outputs = net(images).view(-1)
#                     local_probs = torch.sigmoid(local_outputs)
                    
#                     # 4. 计算纯粹的均方误差 (MSE Loss)
#                     loss = criterion_mse(local_probs, targets)

#                 # 反向传播与优化
#                 loss.backward()
                
#                 # 【安全阀】：梯度裁剪防爆，保障本地持续训练的绝对稳定
#                 torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                
#                 optimizer.step()
#                 epoch_loss += loss.item()
                
#             avg_loss = epoch_loss / len(train_loader) 
#             print(f"    [Client {index}] Local Epoch {local_epoch+1}/{self.local_epoch} | Loss: {avg_loss:.4f}")
            
#         return avg_loss

# #################下面是用于预热阶段本地训练的代码############################
#     def _pseudo_aggregate_for_eval(self):
#         """
#         仅在预热阶段使用：将本地模型平均并赋给 global_net 以便计算测试指标。
#         【核心】：绝不下发覆盖本地的 nets_list，保持客户端纯本地训练！
#         """
#         global_w = copy.deepcopy(self.global_net.state_dict())
#         online_clients = getattr(self, 'online_clients', list(range(self.args.parti_num)))
        
#         # 简单平均
#         freq = 1.0 / len(online_clients)
        
#         first = True
#         for index, net_id in enumerate(online_clients):
#             net_para = self.nets_list[net_id].state_dict()
#             if first:
#                 first = False
#                 for key in net_para:
#                     global_w[key] = net_para[key] * freq
#             else:
#                 for key in net_para:
#                     global_w[key] += net_para[key] * freq
                    
#         # 仅更新全局模型，用于 global_evaluate 测试
#         self.global_net.load_state_dict(global_w)
#         # 注意：这里故意不调用 self.nets_list[i].load_state_dict()
# #######################################################################


#     def _consistency_aware_aggregation(self):
#         # 核心算法: 计算 beta_k 并加权聚合
        
#         # 0. 准备工作
#         if self.public_loader is None:
#             print("Warning: No public loader found. Fallback to FedAvg.")
#             self.aggregate_nets(None)
#             return

#         # 1. 计算全局模型在 D_pub 上的特征 h_global
#         h_global = self._get_features(self.global_net, self.public_loader)
#         h_global = h_global.cpu() # 移到 CPU 方便计算

#         # 2. 计算参考先验 pi_ref (加权平均)
#         # 获取在线客户端的样本数 n_k
#         n_list = []
#         pi_list = []
#         for i in self.online_clients:
#             dl = self.trainloaders[i]
#             # === 修改开始：安全获取 n_k ===
#             if hasattr(dl.sampler, 'indices'):
#                 n_k = len(dl.sampler.indices)
#             else:
#                 n_k = len(dl.dataset)
#             # === 修改结束 ===
#             pi_k = self.client_priors.get(i, 0.0)
#             n_list.append(n_k)
#             pi_list.append(pi_k)
        
#         n_total = sum(n_list)
#         pi_ref = sum([p * n for p, n in zip(pi_list, n_list)]) / (n_total + 1e-10)

#         # 3. 计算每个客户端的 beta_k
#         beta_scores = []
        
#         # 归一化因子 (论文公式中的 Max Deviation)
#         # 简单起见，我们动态计算当前轮次的最大距离作为分母，或者设为常数
#         # 这里先计算所有原始距离
#         dist_priors = []
#         dist_feats = []
        
#         client_feats = {}
        
#         for idx in self.online_clients:
#             # 计算客户端特征 h_k
#             net = self.nets_list[idx]
#             h_k = self._get_features(net, self.public_loader).cpu()
#             client_feats[idx] = h_k
            
#             # 记录距离
#             dist_p = abs(self.client_priors.get(idx, 0) - pi_ref)
#             dist_f = torch.norm(h_k - h_global, p=2).item()
            
#             dist_priors.append(dist_p)
#             dist_feats.append(dist_f)
            
#         # 获取归一化分母 (防止除零)
#         max_dist_p = max(dist_priors) + 1e-6
#         max_dist_f = max(dist_feats) + 1e-6
        
#         lambda_val = self.weight_balance
#         mode = getattr(self.args, 'consistency_mode', 'dual')
#         for i, idx in enumerate(self.online_clients):
#             d_p = dist_priors[i]
#             d_f = dist_feats[i]
#             if mode == 'prior_only':
#                 # 仅看先验比例，无视特征分布差异
#                 term = d_p / max_dist_p
#             elif mode == 'feature_only':
#                 # 仅看特征表示，无视本地正负样本比例的差异
#                 term = d_f / max_dist_f
#             else:
#                 term = lambda_val * (d_p / max_dist_p) + (1 - lambda_val) * (d_f / max_dist_f)
                
#             beta_k = max(0, 1 - term)
#             beta_scores.append(beta_k)

#         # 4. 计算最终权重 alpha_k propto n_k * beta_k
#         final_weights = []
#         for n, beta in zip(n_list, beta_scores):
#             final_weights.append(n * beta)
        
#         # 归一化权重
#         total_w = sum(final_weights) + 1e-10
#         norm_weights = [w / total_w for w in final_weights]
        
#         # 5. 执行聚合
#         self.aggregate_nets(freq=norm_weights)
#         print(f"Aggregation Weights (Top 5): {[round(w, 3) for w in norm_weights[:5]]}")


#     def _get_features(self, net, loader):
#         """辅助函数: 提取特征 h"""
#         net.eval()
#         net.to(self.device)
#         features_list = []
#         with torch.no_grad():
#             for batch in loader:
#                 imgs = batch[0].to(self.device)
#                 # 所有 Backbone 都实现了 features() 方法 (MLP, ResNet, SimpleCNN)
#                 # 如果没有，可能会报错，需确保 Backbone 统一接口
#                 try:
#                     h = net.features(imgs)
#                     # ResNet features 返回可能是 (B, 512, 1, 1)，需展平
#                     if h.dim() > 2:
#                         h = h.view(h.size(0), -1)
#                     features_list.append(h)
#                 except AttributeError:
#                     # Fallback: 如果没有 features 方法 (例如标准 ResNet)
#                     # 这里的 Backbone 应该都修改/确认过有 features 方法
#                     print("Error: Backbone missing 'features()' method.")
#                     return torch.tensor([])
                    
#         return torch.cat(features_list, dim=0)

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
from models.utils.federated_model import FederatedModel
from models.utils.losses import nnPULoss, naive_pu_loss
from utils.args import *

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Federated PU Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser

class FedPU(FederatedModel):
    NAME = 'fedpu'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FedPU, self).__init__(nets_list, args, transform)
        self.public_loader = None # 需在 main.py 或 dataset 初始化后注入
        self.client_priors = {}   # 需注入
        self.Twarm = args.Twarm
        self.weight_balance = args.weight_balance # lambda
        self.public_feat_global = None # 缓存全局模型的探针特征
        self.pseudo_refresh_gap = max(1, int(getattr(args, 'pseudo_refresh_gap', 1)))
        self.teacher_top_m = max(1, int(getattr(args, 'teacher_top_m', 1)))
        self.teacher_alpha = float(getattr(args, 'teacher_alpha', 1.0))
        self.teacher_beta = float(getattr(args, 'teacher_beta', 1.0))
        self.teacher_gamma = float(getattr(args, 'teacher_gamma', 1.0))
        self.client_pseudo_labels = {i: {} for i in range(self.args.parti_num)}
        self.client_teacher_weights = {}
        self.client_u_prototypes = {}

    def ini(self):
        # 初始化全局模型
        # 联邦学习的起手式：
        # 把第 0 个客户端的模型深拷贝一份，当作初始的"全局模型"
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        # 把这个初始权重强制下发给所有客户端，确保大家在同一起跑线
        for net in self.nets_list:
            net.load_state_dict(global_w)

    # 本地的聚合更新阶段
    def loc_update(self, priloader_list):
        self.trainloaders = priloader_list
        total_clients = list(range(self.args.parti_num))
        # 模拟随机选择客户端
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        if self.epoch_index >= self.Twarm and self._should_refresh_pseudo_labels():
            self._refresh_client_pseudo_labels()

        # 1. 本地训练
        total_loss = 0.0
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            total_loss += client_loss

        # 2. 预热阶段与正式阶段统一使用标准 FedAvg
        self.aggregate_nets(None)

        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss

    def _should_refresh_pseudo_labels(self):
        if self.epoch_index < self.Twarm:
            return False
        return (self.epoch_index - self.Twarm) % self.pseudo_refresh_gap == 0

    def _refresh_client_pseudo_labels(self):
        """
        正式阶段的伪标签刷新入口。
        """
        if self.trainloaders is None:
            raise RuntimeError("Train loaders must be initialized before refreshing pseudo-labels.")

        self.client_u_prototypes = self._compute_all_u_prototypes()
        self.client_teacher_weights = {}

        for client_id in range(self.args.parti_num):
            candidate_ids = self._select_teacher_candidates(client_id)
            teacher_weights = self._compute_teacher_weights(client_id, candidate_ids)
            teacher_net = self._build_client_teacher(client_id, teacher_weights)

            self.client_teacher_weights[client_id] = teacher_weights
            self.client_pseudo_labels[client_id] = self._generate_client_pseudo_labels(client_id, teacher_net)

        print(
            f"Refreshed cached pseudo-labels at epoch {self.epoch_index} "
            f"for {self.args.parti_num} clients."
        )

    def _compute_all_u_prototypes(self):
        prototypes = {}
        for client_id in range(self.args.parti_num):
            prototypes[client_id] = self._compute_client_u_prototype(client_id)
        return prototypes

    def _compute_client_u_prototype(self, client_id):
        net = self.nets_list[client_id]
        loader = self.trainloaders[client_id]
        net.eval()
        net.to(self.device)

        feature_sum = None
        u_count = 0

        with torch.no_grad():
            for images, labels, true_labels, indices in loader:
                labels = labels.view(-1)
                u_mask = (labels == 0)
                if not u_mask.any():
                    continue

                images_u = images[u_mask].to(self.device)
                features_u = self._extract_features_batch(net, images_u)
                batch_sum = features_u.sum(dim=0)

                if feature_sum is None:
                    feature_sum = batch_sum
                else:
                    feature_sum += batch_sum
                u_count += features_u.size(0)

        if feature_sum is None or u_count == 0:
            feature_dim = self._infer_feature_dim(net, loader)
            return torch.zeros(feature_dim, dtype=torch.float32)

        return (feature_sum / u_count).detach().cpu()

    def _extract_features_batch(self, net, images):
        try:
            features = net.features(images)
        except AttributeError as exc:
            raise RuntimeError("Backbone missing required features() method for teacher construction.") from exc

        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return features

    def _infer_feature_dim(self, net, loader):
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(self.device)
                features = self._extract_features_batch(net, images[:1])
                return features.size(1)
        return 1

    def _select_teacher_candidates(self, client_id):
        target_proto = self.client_u_prototypes[client_id]
        distances = []

        for other_id in range(self.args.parti_num):
            other_proto = self.client_u_prototypes[other_id]
            if target_proto.numel() != other_proto.numel():
                raise RuntimeError(
                    f"Prototype dimension mismatch between client {client_id} and client {other_id}."
                )
            dist = torch.norm(target_proto - other_proto, p=2).item()
            distances.append((other_id, dist))

        distances.sort(key=lambda item: (item[1], item[0]))
        top_m = min(self.teacher_top_m, len(distances))
        return [cid for cid, _ in distances[:top_m]]

    def _compute_teacher_weights(self, client_id, candidate_ids):
        eps = 1e-12
        target_proto = self.client_u_prototypes[client_id]

        n_terms = {}
        p_terms = {}
        s_terms = {}

        for other_id in candidate_ids:
            train_loader = self.trainloaders[other_id]
            n_terms[other_id] = np.log1p(len(train_loader.dataset))
            p_terms[other_id] = float(self.client_priors.get(other_id, 0.0)) + eps

            other_proto = self.client_u_prototypes[other_id]
            dist = torch.norm(target_proto - other_proto, p=2).item()
            s_terms[other_id] = 1.0 / (dist + eps)

        n_scores = self._normalize_scores(n_terms)
        p_scores = self._normalize_scores(p_terms)
        s_scores = self._normalize_scores(s_terms)

        raw_weights = {}
        for other_id in candidate_ids:
            raw_weights[other_id] = (
                (n_scores[other_id] ** self.teacher_alpha)
                * (p_scores[other_id] ** self.teacher_beta)
                * (s_scores[other_id] ** self.teacher_gamma)
            )

        weights = self._normalize_scores(raw_weights)
        return self._cap_self_weight(client_id, weights)

    def _normalize_scores(self, score_dict):
        total = sum(score_dict.values())
        if total <= 0:
            uniform = 1.0 / max(1, len(score_dict))
            return {key: uniform for key in score_dict}
        return {key: value / total for key, value in score_dict.items()}

    def _cap_self_weight(self, client_id, weight_dict, max_self_weight=0.5):
        if client_id not in weight_dict:
            return weight_dict
        if weight_dict[client_id] <= max_self_weight or len(weight_dict) == 1:
            return weight_dict

        other_ids = [other_id for other_id in weight_dict if other_id != client_id]
        other_total = sum(weight_dict[other_id] for other_id in other_ids)

        capped_weights = dict(weight_dict)
        capped_weights[client_id] = max_self_weight
        remaining_mass = 1.0 - max_self_weight

        if other_total > 0:
            for other_id in other_ids:
                capped_weights[other_id] = weight_dict[other_id] / other_total * remaining_mass
        else:
            uniform_other = remaining_mass / max(1, len(other_ids))
            for other_id in other_ids:
                capped_weights[other_id] = uniform_other

        return self._normalize_scores(capped_weights)

    def _build_client_teacher(self, client_id, teacher_weights):
        teacher_net = copy.deepcopy(self.nets_list[client_id])
        teacher_net.to(self.device)
        teacher_net.eval()

        source_states = {
            source_id: self.nets_list[source_id].state_dict()
            for source_id in teacher_weights
        }
        dominant_source = max(teacher_weights, key=teacher_weights.get)

        teacher_state = teacher_net.state_dict()
        for key, value in teacher_state.items():
            if value.is_floating_point():
                mixed_value = None
                for source_id, weight in teacher_weights.items():
                    source_value = source_states[source_id][key].detach().to(self.device)
                    weighted_value = source_value * weight
                    mixed_value = weighted_value if mixed_value is None else mixed_value + weighted_value
                teacher_state[key] = mixed_value
            else:
                teacher_state[key] = source_states[dominant_source][key].detach().clone()

        teacher_net.load_state_dict(teacher_state)
        return teacher_net

    def _generate_client_pseudo_labels(self, client_id, teacher_net):
        teacher_net.eval()
        teacher_net.to(self.device)
        loader = self.trainloaders[client_id]
        pseudo_cache = {}

        with torch.no_grad():
            for images, labels, true_labels, indices in loader:
                labels = labels.view(-1)
                indices = torch.as_tensor(indices)
                u_mask = (labels == 0)
                if not u_mask.any():
                    continue

                images_u = images[u_mask].to(self.device)
                indices_u = indices[u_mask].tolist()
                probs_u = torch.sigmoid(teacher_net(images_u).view(-1)).detach().cpu().tolist()

                for sample_index, pseudo_prob in zip(indices_u, probs_u):
                    pseudo_cache[int(sample_index)] = float(pseudo_prob)

        teacher_net.cpu()
        return pseudo_cache

    def _get_cached_targets(self, client_id, labels, indices, device):
        labels = labels.view(-1)
        client_cache = self.client_pseudo_labels.get(client_id, {})
        cached_targets = []

        for label, sample_idx in zip(labels.detach().cpu().tolist(), indices):
            if int(label) == 1:
                cached_targets.append(1.0)
                continue

            sample_index = int(sample_idx)
            if sample_index not in client_cache:
                raise RuntimeError(
                    f"Missing cached pseudo-label for client {client_id}, sample index {sample_index} "
                    f"at epoch {self.epoch_index}. Please refresh pseudo-labels before formal-stage training."
                )
            cached_targets.append(float(client_cache[sample_index]))

        return torch.tensor(cached_targets, dtype=torch.float32, device=device)

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()
        
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)
        
        # Phase 1: 预热阶段使用 Naive PU (BCE Loss)
        criterion_warmup = torch.nn.BCEWithLogitsLoss()
        
        # Phase 2: 正式阶段使用 MSE Loss (软伪标签蒸馏)
        criterion_mse = torch.nn.MSELoss()

        avg_loss = 0.0 
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0
            
            for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                
                optimizer.zero_grad()
                
                if self.epoch_index < self.Twarm:
                    # ==========================================
                    # Phase 1: 纯本地 Naive PU 盲训打底
                    # 将 U 视为 0，P 视为 1，快速建立初始决策边界
                    # ==========================================
                    outputs = net(images).view(-1)
                    loss = criterion_warmup(outputs, labels)
                    
                else:
                    # ==========================================
                    # Phase 2: 使用按样本 index 缓存的 soft pseudo-label 训练
                    # ==========================================

                    # 1. 对 P 样本固定使用 1，对 U 样本按真实 index 读取缓存伪标签
                    targets = self._get_cached_targets(index, labels, indices, images.device)

                    # 2. 本地模型前向传播，并转换为概率 (0~1)
                    local_outputs = net(images).view(-1)
                    local_probs = torch.sigmoid(local_outputs)
                    
                    # 3. 计算纯粹的均方误差 (MSE Loss)
                    loss = criterion_mse(local_probs, targets)

                # 反向传播与优化
                loss.backward()
                
                # 【安全阀】：梯度裁剪防爆，保障本地持续训练的绝对稳定
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader) 
            print(f"    [Client {index}] Local Epoch {local_epoch+1}/{self.local_epoch} | Loss: {avg_loss:.4f}")
            
        return avg_loss

# #################下面是用于预热阶段本地训练的代码############################
#     def _pseudo_aggregate_for_eval(self):
#         """
#         仅在预热阶段使用：将本地模型平均并赋给 global_net 以便计算测试指标。
#         【核心】：绝不下发覆盖本地的 nets_list，保持客户端纯本地训练！
#         """
#         global_w = copy.deepcopy(self.global_net.state_dict())
#         online_clients = getattr(self, 'online_clients', list(range(self.args.parti_num)))
        
#         # 简单平均
#         freq = 1.0 / len(online_clients)
        
#         first = True
#         for index, net_id in enumerate(online_clients):
#             net_para = self.nets_list[net_id].state_dict()
#             if first:
#                 first = False
#                 for key in net_para:
#                     global_w[key] = net_para[key] * freq
#             else:
#                 for key in net_para:
#                     global_w[key] += net_para[key] * freq
                    
#         # 仅更新全局模型，用于 global_evaluate 测试
#         self.global_net.load_state_dict(global_w)
#         # 注意：这里故意不调用 self.nets_list[i].load_state_dict()
# #######################################################################


    # def _consistency_aware_aggregation(self):
    #     # 核心算法: 计算 beta_k 并加权聚合
        
    #     # 0. 准备工作
    #     if self.public_loader is None:
    #         print("Warning: No public loader found. Fallback to FedAvg.")
    #         self.aggregate_nets(None)
    #         return

    #     # 1. 计算全局模型在 D_pub 上的特征 h_global
    #     h_global = self._get_features(self.global_net, self.public_loader)
    #     h_global = h_global.cpu() # 移到 CPU 方便计算

    #     # 2. 计算参考先验 pi_ref (加权平均)
    #     # 获取在线客户端的样本数 n_k
    #     n_list = []
    #     pi_list = []
    #     for i in self.online_clients:
    #         dl = self.trainloaders[i]
    #         # === 修改开始：安全获取 n_k ===
    #         if hasattr(dl.sampler, 'indices'):
    #             n_k = len(dl.sampler.indices)
    #         else:
    #             n_k = len(dl.dataset)
    #         # === 修改结束 ===
    #         pi_k = self.client_priors.get(i, 0.0)
    #         n_list.append(n_k)
    #         pi_list.append(pi_k)
        
    #     n_total = sum(n_list)
    #     pi_ref = sum([p * n for p, n in zip(pi_list, n_list)]) / (n_total + 1e-10)

    #     # 3. 计算每个客户端的 beta_k
    #     beta_scores = []
        
    #     # 归一化因子 (论文公式中的 Max Deviation)
    #     # 简单起见，我们动态计算当前轮次的最大距离作为分母，或者设为常数
    #     # 这里先计算所有原始距离
    #     dist_priors = []
    #     dist_feats = []
        
    #     client_feats = {}
        
    #     for idx in self.online_clients:
    #         # 计算客户端特征 h_k
    #         net = self.nets_list[idx]
    #         h_k = self._get_features(net, self.public_loader).cpu()
    #         client_feats[idx] = h_k
            
    #         # 记录距离
    #         dist_p = abs(self.client_priors.get(idx, 0) - pi_ref)
    #         dist_f = torch.norm(h_k - h_global, p=2).item()
            
    #         dist_priors.append(dist_p)
    #         dist_feats.append(dist_f)
            
    #     # 获取归一化分母 (防止除零)
    #     max_dist_p = max(dist_priors) + 1e-6
    #     max_dist_f = max(dist_feats) + 1e-6
        
    #     lambda_val = self.weight_balance
    #     mode = getattr(self.args, 'consistency_mode', 'dual')
    #     for i, idx in enumerate(self.online_clients):
    #         d_p = dist_priors[i]
    #         d_f = dist_feats[i]
    #         if mode == 'prior_only':
    #             # 仅看先验比例，无视特征分布差异
    #             term = d_p / max_dist_p
    #         elif mode == 'feature_only':
    #             # 仅看特征表示，无视本地正负样本比例的差异
    #             term = d_f / max_dist_f
    #         else:
    #             term = lambda_val * (d_p / max_dist_p) + (1 - lambda_val) * (d_f / max_dist_f)
                
    #         beta_k = max(0, 1 - term)
    #         beta_scores.append(beta_k)

    #     # 4. 计算最终权重 alpha_k propto n_k * beta_k
    #     final_weights = []
    #     for n, beta in zip(n_list, beta_scores):
    #         final_weights.append(n * beta)
        
    #     # 归一化权重
    #     total_w = sum(final_weights) + 1e-10
    #     norm_weights = [w / total_w for w in final_weights]
        
    #     # 5. 执行聚合
    #     self.aggregate_nets(freq=norm_weights)
    #     print(f"Aggregation Weights (Top 5): {[round(w, 3) for w in norm_weights[:5]]}")


    def _get_features(self, net, loader):
        """辅助函数: 提取特征 h"""
        net.eval()
        net.to(self.device)
        features_list = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch[0].to(self.device)
                # 所有 Backbone 都实现了 features() 方法 (MLP, ResNet, SimpleCNN)
                # 如果没有，可能会报错，需确保 Backbone 统一接口
                try:
                    h = net.features(imgs)
                    # ResNet features 返回可能是 (B, 512, 1, 1)，需展平
                    if h.dim() > 2:
                        h = h.view(h.size(0), -1)
                    features_list.append(h)
                except AttributeError:
                    # Fallback: 如果没有 features 方法 (例如标准 ResNet)
                    # 这里的 Backbone 应该都修改/确认过有 features 方法
                    print("Error: Backbone missing 'features()' method.")
                    return torch.tensor([])
                    
        return torch.cat(features_list, dim=0)
