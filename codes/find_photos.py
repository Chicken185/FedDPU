import os
import sys
import numpy as np
import torch
from torchvision.datasets import ImageFolder

# 复用项目里的导入逻辑
conf_path = os.getcwd()
sys.path.append(conf_path)

from datasets import get_prive_dataset
from main import parse_args


def get_image_path(train_dataset, real_idx):
    """根据绝对索引，从 ImageFolder 中提取原始图片物理路径"""
    dataset_to_check = train_dataset

    # 兼容可能的封装
    if hasattr(dataset_to_check, 'dataset'):
        dataset_to_check = dataset_to_check.dataset

    try:
        if hasattr(dataset_to_check, 'samples'):
            if 0 <= real_idx < len(dataset_to_check.samples):
                return dataset_to_check.samples[real_idx][0]
            return f"[Index越界] real_idx={real_idx}, samples_len={len(dataset_to_check.samples)}"
        elif hasattr(dataset_to_check, 'imgs'):
            if 0 <= real_idx < len(dataset_to_check.imgs):
                return dataset_to_check.imgs[real_idx][0]
            return f"[Index越界] real_idx={real_idx}, imgs_len={len(dataset_to_check.imgs)}"
        else:
            return f"[无法解析路径] Index={real_idx}"
    except Exception as e:
        return f"[路径解析失败] Index={real_idx}, Error={str(e)}"


def safe_load_state_dict(model, ckpt_path, device):
    """更稳健地加载 checkpoint，兼容多种保存格式"""
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint
    else:
        raise ValueError(f"无法识别的 checkpoint 格式: {type(checkpoint)}")

    # 去掉 DataParallel 前缀
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)

    print(f"[加载权重] {ckpt_path}")
    if missing:
        print(f"  - missing keys: {len(missing)}")
    if unexpected:
        print(f"  - unexpected keys: {len(unexpected)}")

    if len(missing) > 20 or len(unexpected) > 20:
        print("  [警告] state_dict 与模型结构差异较大，请确认 model_name 是否正确。")


def init_and_load_model(args, priv_dataset, model_name, pth_path, device):
    """提取底层 PyTorch 骨干网络并加载权重"""
    args.model = model_name
    backbones_list = priv_dataset.get_backbone(args.parti_num, None, model_name=args.model)
    pure_net = backbones_list[0]

    safe_load_state_dict(pure_net, pth_path, device)

    pure_net.to(device)
    pure_net.eval()
    return pure_net


def compute_f1_from_probs(probs, labels, threshold):
    """
    probs: np.ndarray, shape [N]
    labels: np.ndarray, shape [N], 0/1
    """
    preds = (probs >= threshold).astype(np.int32)

    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

    return f1, precision, recall, tp, fp, fn


def find_oracle_max_f1_threshold(net, dataloader, device):
    """
    在带真值的数据集上搜索 Oracle Max-F1 阈值。
    搜索点使用所有实际出现过的概率值，更精确。
    """
    net.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # test_loader: (img, target)
            # train_loader: (img, s_label, true_label, idx)
            if len(batch) == 2:
                images, labels = batch
            elif len(batch) == 4:
                images, _, labels, _ = batch
            else:
                raise ValueError(f"不支持的 batch 格式，len(batch)={len(batch)}")

            images = images.to(device)
            labels = labels.cpu().numpy().astype(np.int32)

            logits = net(images)
            probs = torch.sigmoid(logits.view(-1)).cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    candidate_thresholds = np.unique(all_probs)
    candidate_thresholds = np.concatenate(([0.0], candidate_thresholds, [1.0]))

    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_tp, best_fp, best_fn = 0, 0, 0

    for th in candidate_thresholds:
        f1, precision, recall, tp, fp, fn = compute_f1_from_probs(all_probs, all_labels, th)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(th)
            best_precision = float(precision)
            best_recall = float(recall)
            best_tp, best_fp, best_fn = int(tp), int(fp), int(fn)

    return {
        "threshold": best_threshold,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "tp": best_tp,
        "fp": best_fp,
        "fn": best_fn,
        "num_samples": int(len(all_labels))
    }


def search_category_cases(
    ours_net,
    base_nets_dict,
    dataloaders,
    train_dataset,
    device,
    category_name,
    ours_threshold,
    base_thresholds_dict,
    ours_metric,
    base_metrics_dict
):
    """
    找出极限案例：
    1) discovery_cases: 真实为正类，Ours 判正，所有 Baseline 判负
    2) robustness_cases: 真实为负类，Ours 判负，所有 Baseline 判正
    使用各模型自己的 Oracle Max-F1 阈值。
    """
    ours_net.eval()
    for net in base_nets_dict.values():
        net.eval()

    discovery_cases = []
    robustness_cases = []

    base_names = list(base_nets_dict.keys())

    print("\n" + "=" * 70)
    print(f"🚀 开始搜寻 [{category_name}] 类别的极限翻盘照片...")
    print(f"对手阵容: {base_names}")
    print(f"[Ours] threshold={ours_threshold:.6f}, F1={ours_metric['f1']:.6f}")
    for name in base_names:
        print(
            f"[{name}] threshold={base_thresholds_dict[name]:.6f}, "
            f"F1={base_metrics_dict[name]['f1']:.6f}"
        )
    print("=" * 70)

    with torch.no_grad():
        for loader_id, loader in enumerate(dataloaders):
            print(f"[{category_name}] 扫描 loader {loader_id + 1}/{len(dataloaders)}")

            for batch in loader:
                if len(batch) != 4:
                    raise ValueError(
                        f"train_loader 的 batch 必须是 4 元组 (images, s_labels, true_labels, indices)，"
                        f"实际长度={len(batch)}"
                    )

                images, s_labels, true_labels, indices = batch

                # 只考察未标注样本 U
                u_mask = (s_labels == 0)
                if not u_mask.any():
                    continue

                imgs_u = images[u_mask].to(device)
                true_y = true_labels[u_mask].cpu()
                real_indices = indices[u_mask].cpu()

                # Ours 概率与预测
                ours_probs = torch.sigmoid(ours_net(imgs_u).view(-1)).cpu()
                ours_preds = (ours_probs >= ours_threshold).int()

                # Baselines 概率与预测
                base_probs_dict = {}
                base_preds_dict = {}

                for name, net in base_nets_dict.items():
                    probs = torch.sigmoid(net(imgs_u).view(-1)).cpu()
                    preds = (probs >= base_thresholds_dict[name]).int()
                    base_probs_dict[name] = probs
                    base_preds_dict[name] = preds

                # 逐个样本审查
                for i in range(len(true_y)):
                    t = int(true_y[i].item())
                    real_idx = int(real_indices[i].item())

                    ours_prob = float(ours_probs[i].item())
                    ours_pred = int(ours_preds[i].item())

                    base_pred_list = [int(base_preds_dict[name][i].item()) for name in base_names]

                    # discovery: 真正类，ours判正，所有baseline判负
                    if t == 1 and ours_pred == 1 and all(bp == 0 for bp in base_pred_list):
                        img_path = get_image_path(train_dataset, real_idx)
                        case_info = {
                            "path": img_path,
                            "real_idx": real_idx,
                            "true_label": t,
                            "ours_prob": ours_prob
                        }
                        for name in base_names:
                            case_info[f"{name}_prob"] = float(base_probs_dict[name][i].item())
                        discovery_cases.append(case_info)

                    # robustness: 真负类，ours判负，所有baseline判正
                    elif t == 0 and ours_pred == 0 and all(bp == 1 for bp in base_pred_list):
                        img_path = get_image_path(train_dataset, real_idx)
                        case_info = {
                            "path": img_path,
                            "real_idx": real_idx,
                            "true_label": t,
                            "ours_prob": ours_prob
                        }
                        for name in base_names:
                            case_info[f"{name}_prob"] = float(base_probs_dict[name][i].item())
                        robustness_cases.append(case_info)

    # 保存
    save_dir = "./experiment_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"case_study_{category_name}.txt")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"【{category_name} 类别】极端对比案例 (Ours 全对 vs Baselines 全错)\n")
        f.write("=" * 110 + "\n")
        f.write("Threshold selection protocol: Oracle Max-F1 on binary test set\n\n")

        f.write("[Ours Metric]\n")
        f.write(f"  threshold : {ours_metric['threshold']:.6f}\n")
        f.write(f"  f1        : {ours_metric['f1']:.6f}\n")
        f.write(f"  precision : {ours_metric['precision']:.6f}\n")
        f.write(f"  recall    : {ours_metric['recall']:.6f}\n")
        f.write(f"  tp/fp/fn  : {ours_metric['tp']}/{ours_metric['fp']}/{ours_metric['fn']}\n")
        f.write(f"  samples   : {ours_metric['num_samples']}\n\n")

        for name in base_names:
            metric = base_metrics_dict[name]
            f.write(f"[{name} Metric]\n")
            f.write(f"  threshold : {metric['threshold']:.6f}\n")
            f.write(f"  f1        : {metric['f1']:.6f}\n")
            f.write(f"  precision : {metric['precision']:.6f}\n")
            f.write(f"  recall    : {metric['recall']:.6f}\n")
            f.write(f"  tp/fp/fn  : {metric['tp']}/{metric['fp']}/{metric['fn']}\n")
            f.write(f"  samples   : {metric['num_samples']}\n\n")

        f.write("=" * 110 + "\n\n")

        f.write(">>> PART 1: 慧眼识珠 (真实为正类，Ours判正，所有Baselines判负)\n\n")
        if len(discovery_cases) == 0:
            f.write("[无符合条件样本]\n\n")
        else:
            for case_id, case in enumerate(discovery_cases, 1):
                f.write(f"[Case {case_id}]\n")
                f.write(f"路径: {case['path']}\n")
                f.write(f"绝对索引: {case['real_idx']}\n")
                f.write(f"真实标签: {case['true_label']}\n")
                f.write(f"  [Ours] ✅ {case['ours_prob']:.6f} (th={ours_threshold:.6f})\n")
                for name in base_names:
                    f.write(
                        f"  [{name}] ❌ {case[f'{name}_prob']:.6f} "
                        f"(th={base_thresholds_dict[name]:.6f})\n"
                    )
                f.write("\n")

        f.write("\n>>> PART 2: 抗噪护体 (真实为负类，Ours判负，所有Baselines判正)\n\n")
        if len(robustness_cases) == 0:
            f.write("[无符合条件样本]\n\n")
        else:
            for case_id, case in enumerate(robustness_cases, 1):
                f.write(f"[Case {case_id}]\n")
                f.write(f"路径: {case['path']}\n")
                f.write(f"绝对索引: {case['real_idx']}\n")
                f.write(f"真实标签: {case['true_label']}\n")
                f.write(f"  [Ours] ✅ {case['ours_prob']:.6f} (th={ours_threshold:.6f})\n")
                for name in base_names:
                    f.write(
                        f"  [{name}] ❌ {case[f'{name}_prob']:.6f} "
                        f"(th={base_thresholds_dict[name]:.6f})\n"
                    )
                f.write("\n")

    print(f"[{category_name}] 搜寻完成！")
    print(f"  正类极限案例: {len(discovery_cases)} 个")
    print(f"  负类极限案例: {len(robustness_cases)} 个")
    print(f"  结果已保存至: {save_path}\n")


def main():
    # 1. 获取参数
    args = parse_args()

    # 2. 强制固定数据集划分环境（必须与你训练时完全一致）
    args.dataset = 'fedpu_imagenette'
    args.beta = 0.75
    args.parti_num = 50
    args.public_size = 100
    args.pos_class_list = [0, 1, 2, 8, 9]
    args.label_freq = 0.2
    args.seed = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备: {device}")

    print("正在严格对齐数据划分...")
    priv_dataset = get_prive_dataset(args)

    # 强制构建 loaders，避免 train_loaders/test_loader 还是初始化时的空列表
    priv_dataset.get_data_loaders()

    pri_train_loaders = priv_dataset.train_loaders
    test_loader = priv_dataset.test_loader

    print("train_loaders num:", len(pri_train_loaders))
    print("test_loader type:", type(test_loader))
    if hasattr(test_loader, 'dataset'):
        print("test dataset size:", len(test_loader.dataset))
    else:
        print("test_loader has no dataset attribute")

    # 用于反查原图路径
    data_root = os.path.expanduser('~/chicken/FL_PU/datasets/imagenette2')
    train_dir = os.path.join(data_root, 'train')
    original_train_dataset = ImageFolder(root=train_dir)

    # 4. 路径配置
    base_dir = './experiment_results'
    os.makedirs(base_dir, exist_ok=True)

    ours_model_name = 'fedpu'
    ours_path = os.path.join(
        base_dir,
        'fedpu_imagenette_fedpu_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm60_wb0.5_Bs108_modedual.pth'
    )

    categories = {
        'PU': {
            'distpu': (
                'distpu',
                'fedpu_imagenette_distpu_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs4_modenone.pth'
            ),
            'nnpu': (
                'nnpu',
                'fedpu_imagenette_nnpu_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs32_modenone.pth'
            ),
            'upu': (
                'upu',
                'fedpu_imagenette_upu_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs64_modenone.pth'
            ),
        },
        'FL': {
            'fedavg': (
                'naive_fedavg',
                'fedpu_imagenette_naive_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs64_modenone.pth'
            ),
            'fedprox': (
                'naive_fedprox',
                'fedpu_imagenette_naive_fedprox_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs64_modenone.pth'
            ),
            'fednova': (
                'naive_fednova',
                'fedpu_imagenette_naive_fednova_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs64_modenone.pth'
            ),
        },
        'SSL': {
            'meanteacher': (
                'meanteacher_fedavg',
                'fedpu_imagenette_meanteacher_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs120_modenone.pth'
            ),
            'fixmatch': (
                'fixmatch_fedavg',
                'fedpu_imagenette_fixmatch_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs64_modenone.pth'
            ),
            'freematch': (
                'freematch_fedavg',
                'fedpu_imagenette_freematch_fedavg_lr0.01_ce100_le5_P50_beta0.75_on0.2_pos01289_pub100_freq0.2_warm10_wb0.5_Bs64_modenone.pth'
            ),
        }
    }

    # 5. 加载 Ours
    if not os.path.exists(ours_path):
        raise FileNotFoundError(f"Ours 模型不存在: {ours_path}")

    print("正在加载 Ours 模型...")
    ours_net = init_and_load_model(args, priv_dataset, ours_model_name, ours_path, device)

    print("正在为 Ours 搜索 Oracle Max-F1 阈值...")
    ours_metric = find_oracle_max_f1_threshold(ours_net, test_loader, device)
    ours_threshold = ours_metric["threshold"]
    print(
        f"[Ours] best threshold={ours_metric['threshold']:.6f}, "
        f"F1={ours_metric['f1']:.6f}, "
        f"P={ours_metric['precision']:.6f}, "
        f"R={ours_metric['recall']:.6f}"
    )

    # 6. 遍历类别
    for cat_name, base_files in categories.items():
        print("\n" + "#" * 90)
        print(f"开始处理类别: {cat_name}")
        print("#" * 90)

        base_nets_dict = {}
        base_thresholds_dict = {}
        base_metrics_dict = {}

        for display_name, (actual_model_name, file_name) in base_files.items():
            pth_full_path = os.path.join(base_dir, file_name)

            if not os.path.exists(pth_full_path):
                print(f"[跳过] 文件不存在: {pth_full_path}")
                continue

            print(f"正在加载 Baseline: {display_name} (model_name={actual_model_name})")
            base_net = init_and_load_model(
                args=args,
                priv_dataset=priv_dataset,
                model_name=actual_model_name,
                pth_path=pth_full_path,
                device=device
            )

            base_nets_dict[display_name] = base_net

            print(f"正在为 {display_name} 搜索 Oracle Max-F1 阈值...")
            metric = find_oracle_max_f1_threshold(base_net, test_loader, device)
            base_metrics_dict[display_name] = metric
            base_thresholds_dict[display_name] = metric["threshold"]

            print(
                f"[{display_name}] best threshold={metric['threshold']:.6f}, "
                f"F1={metric['f1']:.6f}, "
                f"P={metric['precision']:.6f}, "
                f"R={metric['recall']:.6f}"
            )

        if len(base_nets_dict) == 0:
            print(f"[{cat_name}] 没有成功加载任何 baseline，跳过。")
            continue

        search_category_cases(
            ours_net=ours_net,
            base_nets_dict=base_nets_dict,
            dataloaders=pri_train_loaders,
            train_dataset=original_train_dataset,
            device=device,
            category_name=cat_name,
            ours_threshold=ours_threshold,
            base_thresholds_dict=base_thresholds_dict,
            ours_metric=ours_metric,
            base_metrics_dict=base_metrics_dict
        )


if __name__ == '__main__':
    main()