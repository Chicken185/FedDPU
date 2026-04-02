import torch
import torch.nn as nn
import torch.nn.functional as F

class nnPULoss(nn.Module):
    """
    Non-negative PU Loss (nnPU) wrapper based on BCEWithLogitsLoss.
    Risk Estimator: pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
    """
    def __init__(self, prior, beta=0.0, gamma=1.0):
        super(nnPULoss, self).__init__()
        self.prior = prior # π: 先验概率，即正样本在总样本中的比例 P(y=1)
        self.beta = beta  # β: 阈值参数，允许风险估计略微为负（松弛变量）
        self.gamma = gamma  # γ: 梯度缩放参数（本代码片段逻辑中未显式展示完整优化逻辑，通常用于非负约束不满足时的梯度惩罚）
        # 关键点：reduction='none'。
        # 我们不能让 PyTorch 自动求平均，因为 Positive 和 Unlabeled 的 Loss 需要分别加权处理。
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')  

    def forward(self, x, target):
        """
        x: Model output logits (N, 1)
        target: Targets (N, ) or (N, 1), 1 for Positive, 0 for Unlabeled
        """
        # 确保形状一致
        if x.dim() > 1: x = x.view(-1)
        if target.dim() > 1: target = target.view(-1)
            
        target = target.float()
        positive_mask = (target == 1)
        
        # 1. R_p^+ 和 R_p^- : 仅在标记为正的样本 (P) 上计算
        if positive_mask.sum() > 0:
            positive_loss = self.loss_func(x[positive_mask], torch.ones_like(x[positive_mask]))
            r_p_plus = torch.mean(positive_loss)
            
            positive_negative_loss = self.loss_func(x[positive_mask], torch.zeros_like(x[positive_mask]))
            r_p_minus = torch.mean(positive_negative_loss)
        else:
            # 如果当前 Batch 没有抽到正样本，正风险为 0
            r_p_plus = torch.tensor(0.0).to(x.device)
            r_p_minus = torch.tensor(0.0).to(x.device)
            
        # 2. R_u^- : 在无标记数据上计算负风险
        # 【核心修复】：为了实现 U' = U ∪ P 消除标签偏移，
        # 我们直接将当前整个 Batch (包含原 P 和原 U) 视为全集来计算负风险！
        all_unlabeled_loss = self.loss_func(x, torch.zeros_like(x))
        r_u_minus = torch.mean(all_unlabeled_loss)
        
        # 3. 组合 nnPU Risk
        positive_risk = self.prior * r_p_plus
        negative_risk_estimator = r_u_minus - self.prior * r_p_minus
        
        if negative_risk_estimator < -self.beta:
            loss = positive_risk
        else:
            loss = positive_risk + negative_risk_estimator
            
        return loss

# 辅助函数：标准的 CrossEntropy (Naive PU 用)
def naive_pu_loss(x, target):
    if x.dim() > 1:
        x = x.view(-1)
    if target.dim() > 1:
        target = target.view(-1)
    return F.binary_cross_entropy_with_logits(x, target.float())


class uPULoss(nn.Module):
    """
    Unbiased PU Loss (uPU)
    Risk Estimator:
        R = pi * R_p^+ + (R_u^- - pi * R_p^-)
    不做 non-negative 截断。
    """
    def __init__(self, prior):
        super(uPULoss, self).__init__()
        self.prior = prior
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, target):
        if x.dim() > 1:
            x = x.view(-1)
        if target.dim() > 1:
            target = target.view(-1)

        target = target.float()
        positive_mask = (target == 1)

        # ===== 1. R_p^+ 和 R_p^- =====
        if positive_mask.sum() > 0:
            positive_loss = self.loss_func(x[positive_mask],
                                           torch.ones_like(x[positive_mask]))
            r_p_plus = torch.mean(positive_loss)

            positive_negative_loss = self.loss_func(x[positive_mask],
                                                    torch.zeros_like(x[positive_mask]))
            r_p_minus = torch.mean(positive_negative_loss)
        else:
            r_p_plus = torch.tensor(0.0, device=x.device)
            r_p_minus = torch.tensor(0.0, device=x.device)

        # ===== 2. R_u^- =====
        # 标准写法：只在 U 上算
        unlabeled_mask = (target == 0)

        if unlabeled_mask.sum() > 0:
            unlabeled_loss = self.loss_func(x[unlabeled_mask],
                                            torch.zeros_like(x[unlabeled_mask]))
            r_u_minus = torch.mean(unlabeled_loss)
        else:
            r_u_minus = torch.tensor(0.0, device=x.device)

        # ===== 3. 组合 uPU Risk =====
        loss = self.prior * r_p_plus + (r_u_minus - self.prior * r_p_minus)

        return loss