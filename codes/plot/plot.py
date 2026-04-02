import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 🚨 核心设定：确保 SVG 在 PPT 中完美可编辑
# ==========================================
# 强制将字体导出为真实的 Text 对象，而不是 Path
plt.rcParams['svg.fonttype'] = 'none'
# 设置通用无衬线字体（PPT兼容性最好）
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

def generate_feature_shift_svg():
    # 随机种子，确保每次生成的散点位置固定，方便复现
    np.random.seed(42)
    
    # 定义数据量与比例 (20% 先验比例)
    N_total = 150
    N_pos = int(N_total * 0.20)  # 红点：隐藏正样本
    N_unl = N_total - N_pos      # 灰点：未标记数据（背景）

    # ==========================================
    # 数据生成：模拟 Client A (蓝用户) - 聚集在左上角
    # ==========================================
    # 正样本中心 (-2.5, 2.5)，未标记样本中心围绕它略微扩散
    pos_A = np.random.normal(loc=[-2.5, 2.5], scale=0.6, size=(N_pos, 2))
    unl_A = np.random.normal(loc=[-1.5, 1.5], scale=0.9, size=(N_unl, 2))

    # ==========================================
    # 数据生成：模拟 Client B (橙用户) - 聚集在右下角
    # ==========================================
    # 正样本中心 (2.5, -2.5)
    pos_B = np.random.normal(loc=[2.5, -2.5], scale=0.6, size=(N_pos, 2))
    unl_B = np.random.normal(loc=[1.5, -1.5], scale=0.9, size=(N_unl, 2))

    # 定义统一的坐标轴范围，确保两张图的空间刻度完全对齐
    axis_limits = (-5, 5)

    def plot_client_features(pos_data, unl_data, title_text, filename):
        fig, ax = plt.subplots(figsize=(5, 4))
        
        # 1. 先画灰色的 Unlabeled 样本 (在底层)
        # edgecolors='white' 增加白边，在 PPT 中会有非常高级的矢量立体感
        ax.scatter(unl_data[:, 0], unl_data[:, 1], 
                   c='#B0B0B0', s=70, alpha=0.8, 
                   edgecolors='white', linewidth=0.8, 
                   label='Unlabeled (Mixture)')
        
        # 2. 再画红色的 Positive 样本 (在顶层，突出显示)
        ax.scatter(pos_data[:, 0], pos_data[:, 1], 
                   c='#E63946', s=70, alpha=1.0, 
                   edgecolors='white', linewidth=0.8, 
                   label='Positive (Hidden)')

        # 3. 坐标轴美化 (学术概念图风格)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # 隐藏具体的刻度数字，只保留轴线，显得更抽象高级
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加文本标签
        ax.set_xlabel('Feature Dimension 1', fontsize=12, fontweight='bold', color='#333333')
        ax.set_ylabel('Feature Dimension 2', fontsize=12, fontweight='bold', color='#333333')
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=15)

        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        
        # 调整布局并导出为透明背景的 SVG
        plt.tight_layout()
        plt.savefig(filename, format='svg', transparent=True)
        plt.close()
        print(f"✅ 成功生成: {filename}")

    # 分别生成两张图
    plot_client_features(pos_A, unl_A, 'Client A Feature Space', 'feature_shift_clientA.svg')
    plot_client_features(pos_B, unl_B, 'Client B Feature Space', 'feature_shift_clientB.svg')

if __name__ == "__main__":
    generate_feature_shift_svg()