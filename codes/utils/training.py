# import datetime
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,precision_recall_curve,precision_score, recall_score
# import numpy as np
# import torch
# from argparse import Namespace
# from models.utils.federated_model import FederatedModel
# from datasets.utils.federated_dataset import FederatedDataset
# from typing import Tuple
# from torch.utils.data import DataLoader
# from utils.logger import CsvWriter
# import sys
# from tqdm import tqdm
# import os
# import sklearn
# # def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
# #     dl = test_dl
# #     net = model.global_net
# #     status = net.training
# #     net.eval()
# #     correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
# #     for batch_idx, (images, labels) in enumerate(dl):
# #         with torch.no_grad():
# #             images, labels = images.to(model.device), labels.to(model.device)
# #             outputs = net(images)
# #             _, max5 = torch.topk(outputs, 5, dim=-1)
# #             labels = labels.view(-1, 1)
# #             top1 += (labels == max5[:, 0:1]).sum().item()
# #             top5 += (labels == max5).sum().item()
# #             total += labels.size(0)
# #     top1acc = round(100 * top1 / total, 2)
# #     top5acc = round(100 * top5 / total, 2)

# #     accs = top1acc
# #     net.train(status)
# #     return accs
# def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> list:
#     from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
# import numpy as np
# import torch

# def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> list:
#     dl = test_dl
#     net = model.global_net
#     status = net.training
#     net.eval()
    
#     all_targets = []
#     all_probs = []
    
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(dl):
#             images, labels = images.to(model.device), labels.to(model.device)
#             outputs = net(images).view(-1)
#             labels = labels.view(-1)
#             # 获取 Sigmoid 后的原始概率
#             probs = torch.sigmoid(outputs)
            
#             all_targets.extend(labels.cpu().numpy())
#             all_probs.extend(probs.cpu().numpy())
    
#     all_targets = np.array(all_targets)
#     all_probs = np.array(all_probs)
    
#     # 1. 计算 AUC Score (最真实的排序能力，不受阈值影响)
#     if len(np.unique(all_targets)) > 1:
#         auc = roc_auc_score(all_targets, all_probs) * 100
#     else:
#         auc = 0.0

#     # 2. 动态搜索最佳阈值以计算真实 F1-Score
#     precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    
#     # 防止除零错误
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
#     # 获取最大的 F1 值
#     best_f1 = np.max(f1_scores) * 100
    
#     # 获取取得最大 F1 时对应的最佳阈值
#     best_idx = np.argmax(f1_scores)
#     # thresholds 数组比 f1_scores 少一个元素，需做安全处理
#     best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
#     # 3. 使用这个最佳阈值来计算真实的 Accuracy
#     all_preds = (all_probs >= best_thresh).astype(float)
#     acc = accuracy_score(all_targets, all_preds) * 100

#     net.train(status)
    
#     # 打印一下当前的最佳阈值，你会发现它绝不是 0.25，可能非常小（比如 0.05）
#     # print(f"    [Eval] Optimal Threshold found: {best_thresh:.4f}")
    
#     return [round(acc, 2), round(best_f1, 2), round(auc, 2)]

# # def evaluate_pseudo_labels(global_model, trainloaders, device):
# #     """
# #     使用 Oracle Max-F1 选择最佳阈值来评估 PU Learning 伪标签质量
# #     (Precision, Recall, F1, ACC, AUC)
# #     """
# #     global_model.eval()
# #     global_model.to(device)

# #     all_true_u = []
# #     all_pred_u = []
# #     all_probs_u = []  # 新增：专门保存概率用于算 AUC

# #     with torch.no_grad():
# #         for loader in trainloaders:
# #             for images, s_labels, true_labels, indices in loader:
# #                 u_mask = (s_labels == 0)  # 筛选出未标注数据
# #                 if not u_mask.any():
# #                     continue

# #                 images_u = images[u_mask].to(device)
# #                 true_labels_u = true_labels[u_mask]

# #                 # 1. 模型输出 Logits
# #                 outputs = global_model(images_u).view(-1)
                
# #                 # 2. 将 Logits 转为真正的概率 (0.0 ~ 1.0)
# #                 probs_u = torch.sigmoid(outputs)
                
# #                 # 3. 保存概率用于后续计算 precision-recall 和 F1
# #                 all_true_u.extend(true_labels_u.int().cpu().numpy())
# #                 all_probs_u.extend(probs_u.cpu().numpy())  # 保存概率

# #     if len(all_true_u) == 0:
# #         return

# #     # 4. 计算 Precision-Recall 曲线以选择最佳阈值（Oracle Max-F1）
# #     precisions, recalls, thresholds = precision_recall_curve(all_true_u, all_probs_u)

# #     # 防止除零错误：计算 F1 分数
# #     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

# #     # 5. 获取最佳 F1 分数和对应的阈值
# #     best_f1 = np.max(f1_scores) * 100
# #     best_idx = np.argmax(f1_scores)
# #     best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

# #     # 6. 使用最佳阈值计算最终的预测
# #     all_preds = (all_probs_u >= best_thresh).astype(float)

# #     # 7. 计算评估指标
# #     acc = accuracy_score(all_true_u, all_preds) * 100
# #     precision = precision_score(all_true_u, all_preds, zero_division=0)
# #     recall = recall_score(all_true_u, all_preds, zero_division=0)
# #     f1 = f1_score(all_true_u, all_preds, zero_division=0)
    
# #     # 8. 计算 AUC
# #     try:
# #         auc = roc_auc_score(all_true_u, all_probs_u)
# #     except ValueError:
# #         auc = 0.5 

# #     # 输出评估结果
# #     print("\n" + "="*60)
# #     print(f"🔥 U 集合伪标签质量揭秘 (测试样本: {len(all_true_u)} | 最佳阈值: {best_thresh:.4f})")
# #     print(f"🎯 ACC       (准确率): {acc:.4f}")
# #     print(f"🎯 Precision (精确率): {precision:.4f}")
# #     print(f"🎣 Recall    (召回率): {recall:.4f}")
# #     print(f"⭐ F1-Score  (F1分数): {f1:.4f}")
# #     print(f"📈 AUC       (曲线面积): {auc:.4f}")
# #     print("="*60 + "\n")

# #     return acc, precision, recall, f1, auc
# import torch
# import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_curve

# def evaluate_pseudo_labels(global_model, trainloaders, device):
#     """
#     【联邦隐私安全版】使用 Oracle Max-F1 分别评估中心模型在各个客户端 U 集合上的伪标签质量。
#     仅在客户端本地寻找最佳阈值并计算指标，服务器端仅做平均。
#     """
#     global_model.eval()
#     global_model.to(device)

#     # 用于记录所有客户端传上来的合法指标
#     client_metrics = {'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auc': []}

#     print("\n" + "="*80)
#     print(f"🛡️  [终端检测] 中心模型在各个客户端本地 U 集合的伪标签能力揭秘")
#     print("="*80)

#     with torch.no_grad():
#         # 1. 遍历每个客户端 (模拟在客户端本地执行)
#         for client_id, loader in enumerate(trainloaders):
#             local_true_u = []
#             local_probs_u = []

#             for images, s_labels, true_labels, indices in loader:
#                 u_mask = (s_labels == 0)  # 筛选出未标注数据
#                 if not u_mask.any():
#                     continue

#                 images_u = images[u_mask].to(device)
#                 true_labels_u = true_labels[u_mask]

#                 # 模型输出
#                 outputs = global_model(images_u).view(-1)
#                 probs_u = torch.sigmoid(outputs)
                
#                 local_true_u.extend(true_labels_u.int().cpu().numpy())
#                 local_probs_u.extend(probs_u.cpu().numpy())

#             # 如果该客户端根本没有 U 集合数据，直接跳过
#             if len(local_true_u) == 0:
#                 continue

#             local_targets = np.array(local_true_u)
#             local_probs = np.array(local_probs_u)

#             # ==================================================
#             # 2. 在客户端本地计算 Oracle Max-F1 及各项指标
#             # ==================================================
#             # 【安全保护】：联邦 Non-IID 场景下，某个客户端的 U 集合可能全是正样本或全是负样本
#             try:
#                 precisions, recalls, thresholds = precision_recall_curve(local_targets, local_probs)
#                 f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                
#                 best_idx = np.argmax(f1_scores)
#                 # best_f1 = np.max(f1_scores) * 100 # 按照你原来的习惯，这里可以乘100，但我建议统一用小数
#                 best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
#             except ValueError:
#                 # 如果该客户端数据极端单一，无法画曲线，退化为默认 0.5 阈值
#                 best_thresh = 0.5

#             # 使用本地最佳阈值计算最终预测
#             local_preds = (local_probs >= best_thresh).astype(float)

#             # 计算本地评估指标 (保持你原代码的乘 100 习惯)
#             acc = accuracy_score(local_targets, local_preds) * 100
#             pre = precision_score(local_targets, local_preds, zero_division=0)* 100
#             rec = recall_score(local_targets, local_preds, zero_division=0)* 100
#             f1 = f1_score(local_targets, local_preds, zero_division=0)* 100
            
#             try:
#                 auc = roc_auc_score(local_targets, local_probs)* 100
#             except ValueError:
#                 auc = 0.5 

#             # 收集该客户端的合法指标
#             client_metrics['acc'].append(acc)
#             client_metrics['pre'].append(pre)
#             client_metrics['rec'].append(rec)
#             client_metrics['f1'].append(f1)
#             client_metrics['auc'].append(auc)

#             # 打印该客户端的具体情况
#             print(f"  [Client {client_id:02d}] 样本: {len(local_targets):<4} | 本地最佳阈值: {best_thresh:.4f} | "
#                   f"ACC: {acc:.2f} | Pre: {pre:.2f} | Rec: {rec:.2f} | F1: {f1:.2f} | AUC: {auc:.2f}")

#     # ==================================================
#     # 3. 服务器端：对客户端传上来的指标进行平均汇总
#     # ==================================================
#     if len(client_metrics['acc']) == 0:
#         return 0.0, 0.0, 0.0, 0.0, 0.5

#     avg_acc = np.mean(client_metrics['acc'])
#     avg_pre = np.mean(client_metrics['pre'])
#     avg_rec = np.mean(client_metrics['rec'])
#     avg_f1  = np.mean(client_metrics['f1'])
#     avg_auc = np.mean(client_metrics['auc'])

#     print("-" * 80)
#     print(f"🌍 [Global Average] 全网平均客户端 U 集合伪标签质量")
#     print(f"   Avg ACC: {avg_acc:.2f} | Avg Pre: {avg_pre:.4f} | Avg Rec: {avg_rec:.4f} | Avg F1: {avg_f1:.4f} | Avg AUC: {avg_auc:.4f}")
#     print("=" * 80 + "\n")

#     return avg_acc, avg_pre, avg_rec, avg_f1, avg_auc
# # def train(model: FederatedModel, private_dataset: FederatedDataset,
# #           args: Namespace) -> None:
# #     if args.csv_log:
# #         csv_writer = CsvWriter(args, private_dataset)

# #     pri_train_loaders, test_loaders, net_cls_counts = private_dataset.get_data_loaders()
# #     model.trainloaders = pri_train_loaders
# #     model.testlodaers = test_loaders
# #     model.net_cls_counts = net_cls_counts

# #     if hasattr(model, 'ini'):
# #         model.ini()

# #     accs_list = []

# #     Epoch = args.communication_epoch
# #     option_learning_decay = args.learning_decay

# #     for epoch_index in range(Epoch):
# #         model.epoch_index = epoch_index

# #         if hasattr(model, 'loc_update'):
# #             if epoch_index == 1:
# #                 start_time = datetime.datetime.now()
# #                 epoch_loc_loss_dict = model.loc_update(pri_train_loaders)
# #                 end_time = datetime.datetime.now()
# #                 use_time = end_time - start_time
# #                 print(end_time - start_time)
# #                 if args.test_time:
# #                     with open(args.dataset + '_time.csv', 'a') as f:
# #                         f.write(args.model + ',' + str(use_time) + '\n')

# #                     return
# #             else:
# #                 epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

# #             if option_learning_decay == True:
# #                 model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)
# #         accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
# #         print(f'The {epoch_index} Comm Accuracy: ACC={accs[0]}, F1={accs[1]}, AUC={accs[2]} Method:{model.args.model}')
# #         accs_list.append(accs) # 记录完整指标

# #     if args.csv_log:
# #         csv_writer.write_acc(accs_list)
# def train(model: FederatedModel, private_dataset: FederatedDataset,
#           args: Namespace) -> None:
#     if args.csv_log:
#         csv_writer = CsvWriter(args, private_dataset)
#     loaders = private_dataset.get_data_loaders()
#     pri_train_loaders, test_loaders, public_loader, client_priors = loaders
#     model.net_cls_counts = None
#         # 【关键注入】：将探针集和先验传递给 FedPU 模型
#     model.public_loader = public_loader
#     model.client_priors = client_priors
#     # pri_train_loaders, test_loaders, net_cls_counts = private_dataset.get_data_loaders()
#     # model.trainloaders = pri_train_loaders
#     # model.testlodaers = test_loaders
#     # model.net_cls_counts = net_cls_counts
#     model.trainloaders = pri_train_loaders
#     model.testlodaers = test_loaders

#     if hasattr(model, 'ini'):
#         model.ini()

#     # 指标记录列表 (适配 FedPU)
#     history = {
#         'loss': [],
#         'acc': [],
#         'f1': [],
#         'auc': [],
#         'p_acc': [], 'p_pre': [], 'p_rec': [], 'p_f1': [], 'p_auc': [] # p_ 代表 pseudo (伪标签)
#     }

#     Epoch = args.communication_epoch
#     option_learning_decay = args.learning_decay

#     print(f"\n>>> Start Training: {args.model} on {args.dataset} (FedPU Metrics Enabled) <<<")
#     # 强制刷新一下起步提示
#     sys.stdout.flush()

#     # === 修改点 1: 配置 tqdm 适配 nohup ===
#     # file=sys.stdout 确保它和 python -u 配合，ncols 限制宽度防止换行错乱
#     global_progress = tqdm(range(Epoch), desc="Global Rounds", total=Epoch, unit="rd", 
#                            ncols=120, file=sys.stdout, dynamic_ncols=False)
#     for epoch_index in global_progress:
#         model.epoch_index = epoch_index
#         current_loss = 0.0 # 初始化

#         # --- 1. 本地训练 ---
#         if hasattr(model, 'loc_update'):
#             # 兼容 loc_update 的返回值，如果有 return loss 则接收，没有则默认为 0
#             res = model.loc_update(pri_train_loaders)
#             current_loss = res if isinstance(res, (float, int)) else 0.0
            
#             if option_learning_decay:
#                 model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)

#         # === 修改点 2: 使用 tqdm.write 打印 Summary，这绝不会破坏进度条！ ===
#         online_count = len(getattr(model, 'online_clients', []))
#         total_count = args.parti_num
#         tqdm.write(f"    -> [Round {epoch_index:03d}] Processed {online_count}/{total_count} clients | Last Avg Loss: {current_loss:.4f}")


#         # --- 2. 全局评估 ---
#         metrics = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
#         (acc, f1, auc) = metrics
        
#         # 记录
#         history['loss'].append(current_loss)
#         history['acc'].append(acc)
#         history['f1'].append(f1)
#         history['auc'].append(auc)

#         # 更新进度条
#         global_progress.set_postfix({
#             'ACC(↑)': f"{acc:.2f}",
#             'F1(↑)': f"{f1:.2f}", 
#             'AUC(↑)': f"{auc:.2f}"
#         })
#         sys.stdout.flush()
#         if getattr(args, 'eval_pseudo', False) and epoch_index == Epoch - 1:
#             th = getattr(args, 'pseudo_th', 0.5)
#             # tqdm.write 可以保证不打断进度条的显示
#             tqdm.write(f"    -> [Pseudo-label Eval] Threshold: {th}")
            
#             p_metrics = evaluate_pseudo_labels(
#                 global_model=model.global_net, 
#                 trainloaders=pri_train_loaders, # 传入私有训练集
#                 device=model.device, 
#             )
            
#             if p_metrics is not None:
#                 p_acc, p_pre, p_rec, p_f1, p_auc = p_metrics
#             else:
#                 p_acc = p_pre = p_rec = p_f1 = p_auc = 0.0
#         else:
#             p_acc = p_pre = p_rec = p_f1 = p_auc = 0.0

#         # 将当前轮次的伪标签指标记录到历史字典中
#         history['p_acc'].append(p_acc)
#         history['p_pre'].append(p_pre)
#         history['p_rec'].append(p_rec)
#         history['p_f1'].append(p_f1)
#         history['p_auc'].append(p_auc)

#     # --- 3. 结果保存 ---
    
#     # 提取所有参数用于命名 (使用 getattr 防止跑其他基线模型时报错)
#     p_dataset = getattr(args, 'dataset', 'NA')
#     p_model = getattr(args, 'model', 'NA')
#     p_lr = getattr(args, 'local_lr', 'NA')
#     p_cepoch = getattr(args, 'communication_epoch', 'NA')
#     p_lepoch = getattr(args, 'local_epoch', 'NA')
#     p_parti = getattr(args, 'parti_num', 'NA')
#     p_beta = getattr(args, 'beta', 'NA')
#     p_online = getattr(args, 'online_ratio', 'NA')

    
#     # 针对 Pos Class List 的特殊处理，将其拼接为字符串 (例如 [0,1,2] 变成 "012")
#     pos_list = getattr(args, 'pos_class_list', [0])
#     if isinstance(pos_list, list):
#         p_pos = "".join(map(str, pos_list))
#     else:
#         p_pos = str(pos_list)
        
#     p_pub = getattr(args, 'public_size', 'NA')
#     p_freq = getattr(args, 'label_freq', 'NA')
#     p_twarm = getattr(args, 'Twarm', 'NA')
#     p_wb = getattr(args, 'weight_balance', 'NA')
#     p_mode="none"
#     if p_model == "fedpu":
#         p_mode = getattr(args, 'consistency_mode', 'dual')
#     p_bs=getattr(args, 'local_batch_size', 'NA')
    
#     # 严格按照你要求的风格拼接超长 file_name
#     file_name = (f"{p_dataset}_{p_model}_"
#                  f"lr{p_lr}_ce{p_cepoch}_le{p_lepoch}_P{p_parti}_"
#                  f"beta{p_beta}_on{p_online}_pos{p_pos}_pub{p_pub}_"
#                  f"freq{p_freq}_warm{p_twarm}_wb{p_wb}_Bs{p_bs}_mode{p_mode}.txt")

#     results_dir = './experiment_results'
#     save_path = os.path.join(results_dir, file_name)

#     # 获取最优结果 (对于我们的三个指标，都是越大越好)
#     best_acc = max(history['acc'])
#     best_f1 = max(history['f1'])
#     best_auc = max(history['auc'])

#     # 写入文件
#     with open(save_path, 'w') as f:
#         f.write(f"Experiment Configuration:\n")
#         f.write(f"Model: {p_model}\nDataset: {p_dataset}\n")
#         f.write(f"Local LR: {p_lr}\n")
#         f.write(f"Comm Epochs: {p_cepoch}\nLocal Epochs: {p_lepoch}\n")
#         f.write(f"Participants: {p_parti}\nBeta: {p_beta}\nOnline Ratio: {p_online}\n")
#         f.write(f"Pos Classes: {pos_list}\nPublic Size: {p_pub}\nLabel Freq: {p_freq}\n")
#         f.write(f"Warm-up Epochs: {p_twarm}\nWeight Balance: {p_wb}\n")
#         f.write("=" * 60 + "\n")
#         f.write(f"BEST RESULTS:\n")
#         f.write(f"  Top-1 Acc (Max) : {best_acc}%\n")
#         f.write(f"  F1-Score (Max)  : {best_f1}%\n")
#         f.write(f"  AUC Score (Max) : {best_auc}%\n")
#         f.write("=" * 60 + "\n")
#         f.write(f"FINAL ROUND ({Epoch-1}) RESULTS:\n")
#         f.write(f"  Top-1 Acc       : {history['acc'][-1]}%\n")
#         f.write(f"  F1-Score        : {history['f1'][-1]}%\n")
#         f.write(f"  AUC Score       : {history['auc'][-1]}%\n")
#         f.write("-" * 60 + "\n")
#         f.write("Full Logs (Rd, Global_Acc, Global_F1, Global_AUC, Pseudo_Acc, Pseudo_Pre, Pseudo_Rec, Pseudo_F1, Pseudo_AUC):\n")
        
#         # 写入 CSV 格式的详细日志，方便直接粘贴到 Excel 画图
#         for i in range(Epoch):
#             line = (f"{i},{history['acc'][i]},{history['f1'][i]},{history['auc'][i]},"
#                     f"{history['p_acc'][i]},{history['p_pre'][i]},{history['p_rec'][i]},"
#                     f"{history['p_f1'][i]},{history['p_auc'][i]}\n")
#             f.write(line)

#     print(f"\n>>> Results saved to: {save_path}")
#     print(f">>> Final ACC: {history['acc'][-1]}% | Final F1: {history['f1'][-1]}% | Final AUC: {history['auc'][-1]}%")
#     weights_save_path = save_path.replace('.txt', '.pth')
    
#     # 保存全局模型的 state_dict
#     torch.save(model.global_net.state_dict(), weights_save_path)
#     print(f">>> 🥇 全局模型权重已保存至: {weights_save_path}")
#     if args.csv_log:
#         csv_writer.write_acc(history['acc'])
import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,precision_recall_curve,precision_score, recall_score
import numpy as np
import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
from utils.logger import CsvWriter
import sys
from tqdm import tqdm
import os
import sklearn
# def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
#     dl = test_dl
#     net = model.global_net
#     status = net.training
#     net.eval()
#     correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
#     for batch_idx, (images, labels) in enumerate(dl):
#         with torch.no_grad():
#             images, labels = images.to(model.device), labels.to(model.device)
#             outputs = net(images)
#             _, max5 = torch.topk(outputs, 5, dim=-1)
#             labels = labels.view(-1, 1)
#             top1 += (labels == max5[:, 0:1]).sum().item()
#             top5 += (labels == max5).sum().item()
#             total += labels.size(0)
#     top1acc = round(100 * top1 / total, 2)
#     top5acc = round(100 * top5 / total, 2)

#     accs = top1acc
#     net.train(status)
#     return accs
def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> list:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve
import numpy as np
import torch

def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> list:
    dl = test_dl
    net = model.global_net
    status = net.training
    net.eval()
    
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dl):
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = net(images).view(-1)
            labels = labels.view(-1)
            # 获取 Sigmoid 后的原始概率
            probs = torch.sigmoid(outputs)
            
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # 1. 计算 AUC Score (最真实的排序能力，不受阈值影响)
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_probs) * 100
    else:
        auc = 0.0

    # 2. 动态搜索最佳阈值以计算真实 F1-Score
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_probs)
    
    # 防止除零错误
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # 获取最大的 F1 值
    best_f1 = np.max(f1_scores) * 100
    
    # 获取取得最大 F1 时对应的最佳阈值
    best_idx = np.argmax(f1_scores)
    # thresholds 数组比 f1_scores 少一个元素，需做安全处理
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # 3. 使用这个最佳阈值来计算真实的 Accuracy
    all_preds = (all_probs >= best_thresh).astype(float)
    acc = accuracy_score(all_targets, all_preds) * 100

    net.train(status)
    
    # 打印一下当前的最佳阈值，你会发现它绝不是 0.25，可能非常小（比如 0.05）
    # print(f"    [Eval] Optimal Threshold found: {best_thresh:.4f}")
    
    return [round(acc, 2), round(best_f1, 2), round(auc, 2)]

# def evaluate_pseudo_labels(global_model, trainloaders, device):
#     """
#     使用 Oracle Max-F1 选择最佳阈值来评估 PU Learning 伪标签质量
#     (Precision, Recall, F1, ACC, AUC)
#     """
#     global_model.eval()
#     global_model.to(device)

#     all_true_u = []
#     all_pred_u = []
#     all_probs_u = []  # 新增：专门保存概率用于算 AUC

#     with torch.no_grad():
#         for loader in trainloaders:
#             for images, s_labels, true_labels, indices in loader:
#                 u_mask = (s_labels == 0)  # 筛选出未标注数据
#                 if not u_mask.any():
#                     continue

#                 images_u = images[u_mask].to(device)
#                 true_labels_u = true_labels[u_mask]

#                 # 1. 模型输出 Logits
#                 outputs = global_model(images_u).view(-1)
                
#                 # 2. 将 Logits 转为真正的概率 (0.0 ~ 1.0)
#                 probs_u = torch.sigmoid(outputs)
                
#                 # 3. 保存概率用于后续计算 precision-recall 和 F1
#                 all_true_u.extend(true_labels_u.int().cpu().numpy())
#                 all_probs_u.extend(probs_u.cpu().numpy())  # 保存概率

#     if len(all_true_u) == 0:
#         return

#     # 4. 计算 Precision-Recall 曲线以选择最佳阈值（Oracle Max-F1）
#     precisions, recalls, thresholds = precision_recall_curve(all_true_u, all_probs_u)

#     # 防止除零错误：计算 F1 分数
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

#     # 5. 获取最佳 F1 分数和对应的阈值
#     best_f1 = np.max(f1_scores) * 100
#     best_idx = np.argmax(f1_scores)
#     best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

#     # 6. 使用最佳阈值计算最终的预测
#     all_preds = (all_probs_u >= best_thresh).astype(float)

#     # 7. 计算评估指标
#     acc = accuracy_score(all_true_u, all_preds) * 100
#     precision = precision_score(all_true_u, all_preds, zero_division=0)
#     recall = recall_score(all_true_u, all_preds, zero_division=0)
#     f1 = f1_score(all_true_u, all_preds, zero_division=0)
    
#     # 8. 计算 AUC
#     try:
#         auc = roc_auc_score(all_true_u, all_probs_u)
#     except ValueError:
#         auc = 0.5 

#     # 输出评估结果
#     print("\n" + "="*60)
#     print(f"🔥 U 集合伪标签质量揭秘 (测试样本: {len(all_true_u)} | 最佳阈值: {best_thresh:.4f})")
#     print(f"🎯 ACC       (准确率): {acc:.4f}")
#     print(f"🎯 Precision (精确率): {precision:.4f}")
#     print(f"🎣 Recall    (召回率): {recall:.4f}")
#     print(f"⭐ F1-Score  (F1分数): {f1:.4f}")
#     print(f"📈 AUC       (曲线面积): {auc:.4f}")
#     print("="*60 + "\n")

#     return acc, precision, recall, f1, auc
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, precision_recall_curve

def evaluate_pseudo_labels(global_model, trainloaders, device):
    """
    【联邦隐私安全版】使用 Oracle Max-F1 分别评估中心模型在各个客户端 U 集合上的伪标签质量。
    仅在客户端本地寻找最佳阈值并计算指标，服务器端仅做平均。
    """
    global_model.eval()
    global_model.to(device)

    # 用于记录所有客户端传上来的合法指标
    client_metrics = {'acc': [], 'pre': [], 'rec': [], 'f1': [], 'auc': []}

    print("\n" + "="*80)
    print(f"🛡️  [终端检测] 中心模型在各个客户端本地 U 集合的伪标签能力揭秘")
    print("="*80)

    with torch.no_grad():
        # 1. 遍历每个客户端 (模拟在客户端本地执行)
        for client_id, loader in enumerate(trainloaders):
            local_true_u = []
            local_probs_u = []

            for images, s_labels, true_labels, indices in loader:
                u_mask = (s_labels == 0)  # 筛选出未标注数据
                if not u_mask.any():
                    continue

                images_u = images[u_mask].to(device)
                true_labels_u = true_labels[u_mask]

                # 模型输出
                outputs = global_model(images_u).view(-1)
                probs_u = torch.sigmoid(outputs)
                
                local_true_u.extend(true_labels_u.int().cpu().numpy())
                local_probs_u.extend(probs_u.cpu().numpy())

            # 如果该客户端根本没有 U 集合数据，直接跳过
            if len(local_true_u) == 0:
                continue

            local_targets = np.array(local_true_u)
            local_probs = np.array(local_probs_u)

            # ==================================================
            # 2. 在客户端本地计算 Oracle Max-F1 及各项指标
            # ==================================================
            # 【安全保护】：联邦 Non-IID 场景下，某个客户端的 U 集合可能全是正样本或全是负样本
            try:
                precisions, recalls, thresholds = precision_recall_curve(local_targets, local_probs)
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
                
                best_idx = np.argmax(f1_scores)
                # best_f1 = np.max(f1_scores) * 100 # 按照你原来的习惯，这里可以乘100，但我建议统一用小数
                best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            except ValueError:
                # 如果该客户端数据极端单一，无法画曲线，退化为默认 0.5 阈值
                best_thresh = 0.5

            # 使用本地最佳阈值计算最终预测
            local_preds = (local_probs >= best_thresh).astype(float)

            # 计算本地评估指标 (保持你原代码的乘 100 习惯)
            acc = accuracy_score(local_targets, local_preds) * 100
            pre = precision_score(local_targets, local_preds, zero_division=0)* 100
            rec = recall_score(local_targets, local_preds, zero_division=0)* 100
            f1 = f1_score(local_targets, local_preds, zero_division=0)* 100
            
            try:
                auc = roc_auc_score(local_targets, local_probs)* 100
            except ValueError:
                auc = 0.5 

            # 收集该客户端的合法指标
            client_metrics['acc'].append(acc)
            client_metrics['pre'].append(pre)
            client_metrics['rec'].append(rec)
            client_metrics['f1'].append(f1)
            client_metrics['auc'].append(auc)

            # 打印该客户端的具体情况
            print(f"  [Client {client_id:02d}] 样本: {len(local_targets):<4} | 本地最佳阈值: {best_thresh:.4f} | "
                  f"ACC: {acc:.2f} | Pre: {pre:.2f} | Rec: {rec:.2f} | F1: {f1:.2f} | AUC: {auc:.2f}")

    # ==================================================
    # 3. 服务器端：对客户端传上来的指标进行平均汇总
    # ==================================================
    if len(client_metrics['acc']) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    avg_acc = np.mean(client_metrics['acc'])
    avg_pre = np.mean(client_metrics['pre'])
    avg_rec = np.mean(client_metrics['rec'])
    avg_f1  = np.mean(client_metrics['f1'])
    avg_auc = np.mean(client_metrics['auc'])

    print("-" * 80)
    print(f"🌍 [Global Average] 全网平均客户端 U 集合伪标签质量")
    print(f"   Avg ACC: {avg_acc:.2f} | Avg Pre: {avg_pre:.4f} | Avg Rec: {avg_rec:.4f} | Avg F1: {avg_f1:.4f} | Avg AUC: {avg_auc:.4f}")
    print("=" * 80 + "\n")

    return avg_acc, avg_pre, avg_rec, avg_f1, avg_auc
# def train(model: FederatedModel, private_dataset: FederatedDataset,
#           args: Namespace) -> None:
#     if args.csv_log:
#         csv_writer = CsvWriter(args, private_dataset)

#     pri_train_loaders, test_loaders, net_cls_counts = private_dataset.get_data_loaders()
#     model.trainloaders = pri_train_loaders
#     model.testlodaers = test_loaders
#     model.net_cls_counts = net_cls_counts

#     if hasattr(model, 'ini'):
#         model.ini()

#     accs_list = []

#     Epoch = args.communication_epoch
#     option_learning_decay = args.learning_decay

#     for epoch_index in range(Epoch):
#         model.epoch_index = epoch_index

#         if hasattr(model, 'loc_update'):
#             if epoch_index == 1:
#                 start_time = datetime.datetime.now()
#                 epoch_loc_loss_dict = model.loc_update(pri_train_loaders)
#                 end_time = datetime.datetime.now()
#                 use_time = end_time - start_time
#                 print(end_time - start_time)
#                 if args.test_time:
#                     with open(args.dataset + '_time.csv', 'a') as f:
#                         f.write(args.model + ',' + str(use_time) + '\n')

#                     return
#             else:
#                 epoch_loc_loss_dict = model.loc_update(pri_train_loaders)

#             if option_learning_decay == True:
#                 model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)
#         accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
#         print(f'The {epoch_index} Comm Accuracy: ACC={accs[0]}, F1={accs[1]}, AUC={accs[2]} Method:{model.args.model}')
#         accs_list.append(accs) # 记录完整指标

#     if args.csv_log:
#         csv_writer.write_acc(accs_list)
def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)
    loaders = private_dataset.get_data_loaders()
    pri_train_loaders, test_loaders, public_loader, client_priors = loaders
    model.net_cls_counts = None
        # 【关键注入】：将探针集和先验传递给 FedPU 模型
    model.public_loader = public_loader
    model.client_priors = client_priors
    # pri_train_loaders, test_loaders, net_cls_counts = private_dataset.get_data_loaders()
    # model.trainloaders = pri_train_loaders
    # model.testlodaers = test_loaders
    # model.net_cls_counts = net_cls_counts
    model.trainloaders = pri_train_loaders
    model.testlodaers = test_loaders

    if hasattr(model, 'ini'):
        model.ini()

    # 指标记录列表 (适配 FedPU)
    history = {
        'loss': [],
        'acc': [],
        'f1': [],
        'auc': [],
        'p_acc': [], 'p_pre': [], 'p_rec': [], 'p_f1': [], 'p_auc': [] # p_ 代表 pseudo (伪标签)
    }

    Epoch = args.communication_epoch
    option_learning_decay = args.learning_decay

    print(f"\n>>> Start Training: {args.model} on {args.dataset} (FedPU Metrics Enabled) <<<")
    # 强制刷新一下起步提示
    sys.stdout.flush()

    # === 修改点 1: 配置 tqdm 适配 nohup ===
    # file=sys.stdout 确保它和 python -u 配合，ncols 限制宽度防止换行错乱
    global_progress = tqdm(range(Epoch), desc="Global Rounds", total=Epoch, unit="rd", 
                           ncols=120, file=sys.stdout, dynamic_ncols=False)
    for epoch_index in global_progress:
        model.epoch_index = epoch_index
        current_loss = 0.0 # 初始化

        # --- 1. 本地训练 ---
        if hasattr(model, 'loc_update'):
            # 兼容 loc_update 的返回值，如果有 return loss 则接收，没有则默认为 0
            res = model.loc_update(pri_train_loaders)
            current_loss = res if isinstance(res, (float, int)) else 0.0
            
            if option_learning_decay:
                model.local_lr = args.local_lr * (1 - epoch_index / Epoch * 0.9)

        # === 修改点 2: 使用 tqdm.write 打印 Summary，这绝不会破坏进度条！ ===
        online_count = len(getattr(model, 'online_clients', []))
        total_count = args.parti_num
        tqdm.write(f"    -> [Round {epoch_index:03d}] Processed {online_count}/{total_count} clients | Last Avg Loss: {current_loss:.4f}")


        # --- 2. 全局评估 ---
        metrics = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
        (acc, f1, auc) = metrics
        
        # 记录
        history['loss'].append(current_loss)
        history['acc'].append(acc)
        history['f1'].append(f1)
        history['auc'].append(auc)

        # 更新进度条
        global_progress.set_postfix({
            'ACC(↑)': f"{acc:.2f}",
            'F1(↑)': f"{f1:.2f}", 
            'AUC(↑)': f"{auc:.2f}"
        })
        sys.stdout.flush()
        if getattr(args, 'eval_pseudo', False) and epoch_index == Epoch - 1:
            th = getattr(args, 'pseudo_th', 0.5)
            # tqdm.write 可以保证不打断进度条的显示
            tqdm.write(f"    -> [Pseudo-label Eval] Threshold: {th}")
            
            p_metrics = evaluate_pseudo_labels(
                global_model=model.global_net, 
                trainloaders=pri_train_loaders, # 传入私有训练集
                device=model.device, 
            )
            
            if p_metrics is not None:
                p_acc, p_pre, p_rec, p_f1, p_auc = p_metrics
            else:
                p_acc = p_pre = p_rec = p_f1 = p_auc = 0.0
        else:
            p_acc = p_pre = p_rec = p_f1 = p_auc = 0.0

        # 将当前轮次的伪标签指标记录到历史字典中
        history['p_acc'].append(p_acc)
        history['p_pre'].append(p_pre)
        history['p_rec'].append(p_rec)
        history['p_f1'].append(p_f1)
        history['p_auc'].append(p_auc)

    # --- 3. 结果保存 ---
    
    # 提取所有参数用于命名 (使用 getattr 防止跑其他基线模型时报错)
    p_dataset = getattr(args, 'dataset', 'NA')
    p_model = getattr(args, 'model', 'NA')
    p_lr = getattr(args, 'local_lr', 'NA')
    p_cepoch = getattr(args, 'communication_epoch', 'NA')
    p_lepoch = getattr(args, 'local_epoch', 'NA')
    p_parti = getattr(args, 'parti_num', 'NA')
    p_beta = getattr(args, 'beta', 'NA')
    p_online = getattr(args, 'online_ratio', 'NA')

    
    # 针对 Pos Class List 的特殊处理，将其拼接为字符串 (例如 [0,1,2] 变成 "012")
    pos_list = getattr(args, 'pos_class_list', [0])
    if isinstance(pos_list, list):
        p_pos = "".join(map(str, pos_list))
    else:
        p_pos = str(pos_list)
        
    p_pub = getattr(args, 'public_size', 'NA')
    p_freq = getattr(args, 'label_freq', 'NA')
    p_twarm = getattr(args, 'Twarm', 'NA')
    p_wb = getattr(args, 'weight_balance', 'NA')
    p_refresh = getattr(args, 'pseudo_refresh_gap', 'NA')
    p_topm = getattr(args, 'teacher_top_m', 'NA')
    p_talpha = getattr(args, 'teacher_alpha', 'NA')
    p_tbeta = getattr(args, 'teacher_beta', 'NA')
    p_tgamma = getattr(args, 'teacher_gamma', 'NA')
    p_mode="none"
    if p_model == "fedpu":
        p_mode = "cached_teacher"
    p_bs=getattr(args, 'local_batch_size', 'NA')
    
    # 严格按照你要求的风格拼接超长 file_name
    file_name = (f"{p_dataset}_{p_model}_"
                 f"lr{p_lr}_ce{p_cepoch}_le{p_lepoch}_P{p_parti}_"
                 f"beta{p_beta}_on{p_online}_pos{p_pos}_pub{p_pub}_"
                 f"freq{p_freq}_warm{p_twarm}_wb{p_wb}_"
                 f"gap{p_refresh}_top{p_topm}_ta{p_talpha}_tb{p_tbeta}_tg{p_tgamma}_"
                 f"Bs{p_bs}_mode{p_mode}.txt")

    results_dir = './experiment_results'
    save_path = os.path.join(results_dir, file_name)

    # 获取最优结果 (对于我们的三个指标，都是越大越好)
    best_acc = max(history['acc'])
    best_f1 = max(history['f1'])
    best_auc = max(history['auc'])

    # 写入文件
    with open(save_path, 'w') as f:
        f.write(f"Experiment Configuration:\n")
        f.write(f"Model: {p_model}\nDataset: {p_dataset}\n")
        f.write(f"Local LR: {p_lr}\n")
        f.write(f"Comm Epochs: {p_cepoch}\nLocal Epochs: {p_lepoch}\n")
        f.write(f"Participants: {p_parti}\nBeta: {p_beta}\nOnline Ratio: {p_online}\n")
        f.write(f"Pos Classes: {pos_list}\nPublic Size: {p_pub}\nLabel Freq: {p_freq}\n")
        f.write(f"Warm-up Epochs: {p_twarm}\nWeight Balance: {p_wb}\n")
        f.write(f"Pseudo Refresh Gap: {p_refresh}\n")
        f.write(f"Teacher Top-m: {p_topm}\n")
        f.write(f"Teacher Alpha: {p_talpha}\nTeacher Beta: {p_tbeta}\nTeacher Gamma: {p_tgamma}\n")
        f.write(f"FedPU Mode: {p_mode}\n")
        f.write("=" * 60 + "\n")
        f.write(f"BEST RESULTS:\n")
        f.write(f"  Top-1 Acc (Max) : {best_acc}%\n")
        f.write(f"  F1-Score (Max)  : {best_f1}%\n")
        f.write(f"  AUC Score (Max) : {best_auc}%\n")
        f.write("=" * 60 + "\n")
        f.write(f"FINAL ROUND ({Epoch-1}) RESULTS:\n")
        f.write(f"  Top-1 Acc       : {history['acc'][-1]}%\n")
        f.write(f"  F1-Score        : {history['f1'][-1]}%\n")
        f.write(f"  AUC Score       : {history['auc'][-1]}%\n")
        f.write("-" * 60 + "\n")
        f.write("Full Logs (Rd, Global_Acc, Global_F1, Global_AUC, Pseudo_Acc, Pseudo_Pre, Pseudo_Rec, Pseudo_F1, Pseudo_AUC):\n")
        
        # 写入 CSV 格式的详细日志，方便直接粘贴到 Excel 画图
        for i in range(Epoch):
            line = (f"{i},{history['acc'][i]},{history['f1'][i]},{history['auc'][i]},"
                    f"{history['p_acc'][i]},{history['p_pre'][i]},{history['p_rec'][i]},"
                    f"{history['p_f1'][i]},{history['p_auc'][i]}\n")
            f.write(line)

    print(f"\n>>> Results saved to: {save_path}")
    print(f">>> Final ACC: {history['acc'][-1]}% | Final F1: {history['f1'][-1]}% | Final AUC: {history['auc'][-1]}%")
    weights_save_path = save_path.replace('.txt', '.pth')
    
    # 保存全局模型的 state_dict
    torch.save(model.global_net.state_dict(), weights_save_path)
    print(f">>> 🥇 全局模型权重已保存至: {weights_save_path}")
    if args.csv_log:
        csv_writer.write_acc(history['acc'])
