import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=16, hidden_dims=[64, 32], output_dim=1):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        
        # 特征提取部分 (用于计算特征一致性)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        # 分类头 (Logits)
        self.fc = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        y = self.fc(h)
        return y

    def features(self, x):
        """返回特征向量 h (用于聚合时的特征距离计算)"""
        return self.feature_extractor(x)
    
    def classifier(self, h):
        """从特征向量预测 logits"""
        return self.fc(h)