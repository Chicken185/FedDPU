import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors
import numpy as np
import os

# =========================
# 1. 全局设置
# =========================
plt.rcParams['svg.fonttype'] = 'none'   # 文字保留为文本，便于PPT编辑
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 2. baseline 顺序
# =========================
baseline_order = [
    "FixMatch",
    "FedProx",
    "nnPU",
    "DistPU",
    "FedNova",
    "FreeMatch",
    "FedAvg",
    "MeanTeacher",
    "uPU",
    "FedPU-Feature",
    "FedPU-Prior",
    "Target"
]

client_order = ["C43", "C03", "C46", "C31", "C35"]

# =========================
# 3. 数据
# =========================
data = {
    "C43": {
        "FixMatch": 13.50,
        "FedProx": 47.85,
        "nnPU": 88.34,
        "DistPU": 90.18,
        "FedNova": 91.41,
        "FreeMatch": 92.02,
        "FedAvg": 93.87,
        "MeanTeacher": 95.09,
        "uPU": 95.09,
        "FedPU-Feature": 96.32,
        "FedPU-Prior": 97.55,
        "Target": 98.77,
    },
    "C03": {
        "FixMatch": 68.75,
        "FedProx": 72.50,
        "nnPU": 87.50,
        "DistPU": 81.25,
        "FedNova": 83.75,
        "FreeMatch": 93.75,
        "FedAvg": 83.75,
        "MeanTeacher": 90.00,
        "uPU": 92.50,
        "FedPU-Feature": 96.25,
        "FedPU-Prior": 96.25,
        "Target": 97.50,
    },
    "C46": {
        "FixMatch": 30.07,
        "FedProx": 34.97,
        "nnPU": 83.22,
        "DistPU": 83.22,
        "FedNova": 86.01,
        "FreeMatch": 71.33,
        "FedAvg": 82.52,
        "MeanTeacher": 82.52,
        "uPU": 88.81,
        "FedPU-Feature": 85.31,
        "FedPU-Prior": 86.71,
        "Target": 93.71,
    },
    "C31": {
        "FixMatch": 23.94,
        "FedProx": 51.41,
        "nnPU": 88.03,
        "DistPU": 59.15,
        "FedNova": 87.32,
        "FreeMatch": 77.46,
        "FedAvg": 85.21,
        "MeanTeacher": 81.69,
        "uPU": 78.87,
        "FedPU-Feature": 88.73,
        "FedPU-Prior": 90.85,
        "Target": 92.96,
    },
    "C35": {
        "FixMatch": 41.28,
        "FedProx": 41.28,
        "nnPU": 86.24,
        "DistPU": 81.65,
        "FedNova": 89.91,
        "FreeMatch": 87.16,
        "FedAvg": 88.07,
        "MeanTeacher": 89.91,
        "uPU": 89.91,
        "FedPU-Feature": 88.99,
        "FedPU-Prior": 88.07,
        "Target": 91.74,
    }
}

# =========================
# 4. 每行主色（淡色系）
# =========================
row_base_colors = {
    "C43": "#FFC000",  # 淡黄
    "C03": "#ADEB7B",  # 淡绿
    "C46": "#49EBE6",  # 淡蓝
    "C31": "#FAC0DB",  # 淡粉
    "C35": "#D86E58",  # 淡紫
}

# =========================
# 5. 白色 -> 主色 插值函数
# =========================
def blend_with_white(base_color, t):
    """
    t in [0,1]
    t越小越接近白色
    t越大越接近base_color
    """
    base_rgb = np.array(mcolors.to_rgb(base_color))
    white_rgb = np.array([1.0, 1.0, 1.0])
    rgb = white_rgb * (1 - t) + base_rgb * t
    return rgb

# =========================
# 6. 构造数值矩阵
# =========================
values = np.array([
    [data[c][b] for b in baseline_order]
    for c in client_order
], dtype=float)

# =========================
# 7. 绘图函数
# =========================
def plot_acc_heatmap_soft(save_dir="acc_heatmap_soft_svg_results",
                          filename="usps_client_acc_heatmap_soft.svg"):
    os.makedirs(save_dir, exist_ok=True)

    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 逐格绘制
    for i in range(n_rows):
        client_name = client_order[i]
        row_vals = values[i]
        row_min = row_vals.min()
        row_max = row_vals.max()

        for j in range(n_cols):
            val = values[i, j]

            # 行内归一化
            if abs(row_max - row_min) < 1e-12:
                t = 0.55
            else:
                t = (val - row_min) / (row_max - row_min)

            # 控制颜色整体更淡
            # 这里是最关键的深浅控制参数
            t = 0.2 + 0.5 * t

            color = blend_with_white(row_base_colors[client_name], t)

            rect = Rectangle(
                (j, i), 1, 1,
                facecolor=color,
                edgecolor="#B8B8B8",
                linewidth=1.0
            )
            ax.add_patch(rect)

            ax.text(
                j + 0.5, i + 0.5,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="black"
            )

    # 高亮 Target 列
    target_col = baseline_order.index("Target")
    for i in range(n_rows):
        highlight_rect = Rectangle(
            (target_col, i), 1, 1,
            fill=False,
            edgecolor="#666666",
            linewidth=1.8
        )
        ax.add_patch(highlight_rect)

    # 坐标轴设置
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(baseline_order, rotation=35, ha='right', fontsize=10)

    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(client_order, fontsize=11)

    ax.set_title("ACC Comparison Across Baselines and Target",
                 fontsize=15, fontweight='bold', pad=10)

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#808080")

    plt.tight_layout()

    out_path = os.path.join(save_dir, filename)
    plt.savefig(out_path, format="svg")
    plt.close()

    print(f"Saved: {out_path}")

# =========================
# 8. 执行
# =========================
plot_acc_heatmap_soft()