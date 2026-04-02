import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def denormalize_image(tensor):
    """
    通用反归一化函数，将 Tensor 转换为适合 matplotlib 显示的 numpy 数组 (H, W, C)
    通过 Min-Max 缩放保证图像颜色在 [0, 1] 之间，无需手动传入均值和方差。
    """
    img = tensor.clone().detach().cpu()
    min_val = float(img.min())
    max_val = float(img.max())
    img.clamp_(min=min_val, max=max_val)
    img = (img - min_val) / (max_val - min_val + 1e-5)
    
    # 将 (C, H, W) 转换为 (H, W, C)
    if img.dim() == 3:
        img = img.permute(1, 2, 0).numpy()
    else:
        img = img.numpy()
        
    # 如果是灰度图 (USPS/MNIST)，去掉最后一个单通道维度
    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        
    return img

def find_and_plot_cases(ours_model, base_model, dataloaders, device, threshold=0.5, num_cases=5, save_dir='./experiment_results'):
    """
    在 U 集合中寻找完美的对比案例，并绘制高对比度的展示图。
    """
    ours_model.eval()
    base_model.eval()
    ours_model.to(device)
    base_model.to(device)
    
    cases_discovery = []  # Ours=1, Base=0, True=1
    cases_robustness = [] # Ours=0, Base=1, True=0
    
    print(f"\n>>> 正在 U 集合中搜寻完美对比案例 (阈值: {threshold})...")
    
    with torch.no_grad():
        for loader in dataloaders:
            for images, s_labels, true_labels, indices in loader:
                # 核心过滤：只在未标记集合 U 中寻找
                u_mask = (s_labels == 0)
                if not u_mask.any():
                    continue
                
                imgs_u = images[u_mask].to(device)
                true_y = true_labels[u_mask]
                
                # 两个模型分别进行预测
                ours_logits = ours_model(imgs_u).view(-1)
                base_logits = base_model(imgs_u).view(-1)
                
                ours_prob = torch.sigmoid(ours_logits).cpu()
                base_prob = torch.sigmoid(base_logits).cpu()
                
                ours_pred = (ours_prob > threshold).int()
                base_pred = (base_prob > threshold).int()
                
                # 开始筛选
                for i in range(len(true_y)):
                    t = true_y[i].item()
                    o = ours_pred[i].item()
                    b = base_pred[i].item()
                    
                    # 案例 1：慧眼识珠 (Discovery)
                    if t == 1 and o == 1 and b == 0 and len(cases_discovery) < num_cases:
                        cases_discovery.append({
                            'img': imgs_u[i], 't_prob': ours_prob[i].item(), 'b_prob': base_prob[i].item()
                        })
                    
                    # 案例 2：抗噪护体 (Robustness)
                    elif t == 0 and o == 0 and b == 1 and len(cases_robustness) < num_cases:
                        cases_robustness.append({
                            'img': imgs_u[i], 't_prob': ours_prob[i].item(), 'b_prob': base_prob[i].item()
                        })
                        
                if len(cases_discovery) >= num_cases and len(cases_robustness) >= num_cases:
                    break
            if len(cases_discovery) >= num_cases and len(cases_robustness) >= num_cases:
                break

    print(f"搜寻完毕！找到 {len(cases_discovery)} 个 Discovery 案例，{len(cases_robustness)} 个 Robustness 案例。")
    
    # ================= 开始绘图 =================
    if len(cases_discovery) == 0 and len(cases_robustness) == 0:
        print("未找到符合条件的极端对比案例，可能是两个模型预测高度一致。")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # 我们画两行：第一行是 Discovery，第二行是 Robustness
    fig, axes = plt.subplots(2, num_cases, figsize=(3 * num_cases, 7))
    
    # 如果只找到一类，适配 axes 的维度
    if axes.ndim == 1:
        axes = np.expand_dims(axes, axis=0)

    # 绘制第一行 (Discovery)
    for idx in range(num_cases):
        ax = axes[0, idx]
        if idx < len(cases_discovery):
            case = cases_discovery[idx]
            img_np = denormalize_image(case['img'])
            # 兼容灰度图和彩色图
            if img_np.ndim == 2:
                ax.imshow(img_np, cmap='gray')
            else:
                ax.imshow(img_np)
                
            ax.set_title(f"Ground Truth: POSITIVE\n"
                         f"Ours: {case['t_prob']:.2f} (Pos) ✅\n"
                         f"Base: {case['b_prob']:.2f} (Neg) ❌", 
                         fontsize=10, color='darkgreen', weight='bold')
        ax.axis('off')

    # 绘制第二行 (Robustness)
    for idx in range(num_cases):
        ax = axes[1, idx]
        if idx < len(cases_robustness):
            case = cases_robustness[idx]
            img_np = denormalize_image(case['img'])
            if img_np.ndim == 2:
                ax.imshow(img_np, cmap='gray')
            else:
                ax.imshow(img_np)
                
            ax.set_title(f"Ground Truth: NEGATIVE\n"
                         f"Ours: {case['t_prob']:.2f} (Neg) ✅\n"
                         f"Base: {case['b_prob']:.2f} (Pos) ❌", 
                         fontsize=10, color='darkred', weight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pseudo_label_case_study.svg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n>>> 高清对比可视化图已保存至: {save_path}")