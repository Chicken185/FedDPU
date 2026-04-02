import torch
import torch.nn as nn
import torch.optim as optim
from models.utils.federated_model import FederatedModel
import copy


class NaiveFedNova(FederatedModel):

    NAME = 'naive_fednova'
    COMPATIBILITY = ['homogeneity']
    """
    Baseline: Naive PU + FedNova

    将所有未标记数据 (U) 视为负类 (0)，标记数据 (P) 视为正类 (1)，
    使用标准 BCE 进行本地训练。

    与 FedAvg 的区别：
    在服务器端聚合时使用 FedNova normalization，
    解决不同客户端 local update step 不一致导致的 objective inconsistency。
    """

    def __init__(self, nets_list, args, transform):
        super(NaiveFedNova, self).__init__(nets_list, args, transform)
        self.criterion = nn.BCEWithLogitsLoss()

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):

        total_clients = list(range(self.args.parti_num))

        online_clients = self.random_state.choice(
            total_clients, self.online_num, replace=False
        ).tolist()

        self.online_clients = online_clients

        total_loss = 0.0

        # ==============================
        # [FedNova ADD]
        # 用于记录每个客户端的 local update step 数
        # ==============================
        local_steps_list = []

        # 保存每个客户端训练后的模型参数
        client_weights = []

        # 保存当前 global model
        global_w = copy.deepcopy(self.global_net.state_dict())

        # 1. 客户端本地训练
        for i in online_clients:

            # 记录本地模型训练前参数
            self.nets_list[i].load_state_dict(global_w)

            client_loss, local_steps = self._train_net(
                i, self.nets_list[i], priloader_list[i]
            )

            total_loss += client_loss

            # ==============================
            # [FedNova ADD]
            # 保存 local step
            # ==============================
            local_steps_list.append(local_steps)

            # ==============================
            # [FedNova ADD]
            # 保存客户端模型参数
            # ==============================
            client_weights.append(
                copy.deepcopy(self.nets_list[i].state_dict())
            )

        # ==============================
        # [FedNova ADD]
        # 使用 FedNova 聚合
        # ==============================
        new_global_w = self._aggregate_fednova(
            global_w,
            client_weights,
            local_steps_list
        )

        # 更新 global model
        self.global_net.load_state_dict(new_global_w)

        # 下发给所有客户端
        for net in self.nets_list:
            net.load_state_dict(new_global_w)

        avg_round_loss = (
            total_loss / len(online_clients)
            if len(online_clients) > 0
            else 0.0
        )

        return avg_round_loss

    def _train_net(self, index, net, train_loader):

        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(
            net.parameters(),
            lr=self.local_lr,
            momentum=0.9,
            weight_decay=self.args.reg
        )

        avg_loss = 0.0

        # ==============================
        # [FedNova ADD]
        # 统计 local update step
        # ==============================
        local_steps = 0

        for local_epoch in range(self.local_epoch):

            epoch_loss = 0.0

            for batch_idx, (images, labels,true_labels, indices) in enumerate(train_loader):

                images = images.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()

                outputs = net(images).view(-1)
                labels = labels.view(-1)

                loss = self.criterion(outputs, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    max_norm=2.0
                )

                optimizer.step()

                epoch_loss += loss.item()

                # ==============================
                # [FedNova ADD]
                # 每个 batch 算一个 step
                # ==============================
                local_steps += 1

            avg_loss = epoch_loss / len(train_loader)

            print(
                f"[Client {index}] Local Epoch {local_epoch+1}/{self.local_epoch} | Loss: {avg_loss:.4f}"
            )

        # ==============================
        # [FedNova ADD]
        # 返回 local_steps
        # ==============================
        return avg_loss, local_steps

    # ==========================================
    # [FedNova ADD]
    # FedNova 聚合函数
    # ==========================================
    def _aggregate_fednova(self, global_w, client_weights, local_steps_list):

        new_global_w = copy.deepcopy(global_w)

        total_steps = sum(local_steps_list)

        for key in new_global_w.keys():

            update = torch.zeros_like(new_global_w[key])

            for client_w, steps in zip(client_weights, local_steps_list):

                delta = client_w[key] - global_w[key]

                # FedNova normalization
                update += (steps / total_steps) * delta

            new_global_w[key] = global_w[key] + update

        return new_global_w