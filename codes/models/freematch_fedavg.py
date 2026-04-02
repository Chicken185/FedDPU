import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torchvision.transforms as transforms
from models.utils.federated_model import FederatedModel


class FreeMatchFedAvg(FederatedModel):
    """
    Baseline: FreeMatch + FedAvg (Binary / PU-style adaptation)

    - Labeled (P): supervised BCE on y=1.
    - Unlabeled (U): weak -> pseudo-label + confidence, adaptive threshold (EMA of confidence),
      strong -> BCE on masked samples, optionally confidence-weighted.
    - Optional (recommended) distribution alignment via logit-shift using EMA of predicted positive rate
      (per-client), targeting client prior if available.

    This follows the same training skeleton as FixMatchFedAvg.
    """
    NAME = 'freematch_fedavg'
    COMPATIBILITY = ['homogeneity']

    def __init__(self, nets_list, args, transform):
        super(FreeMatchFedAvg, self).__init__(nets_list, args, transform)

        # Supervised loss (P)
        self.criterion_sup = nn.BCEWithLogitsLoss()

        # Unsupervised loss needs sample-wise weights sometimes
        self.criterion_unsup_none = nn.BCEWithLogitsLoss(reduction='none')

        # FreeMatch hyper-params
        self.lambda_u = getattr(self.args, 'lambda_u', 1.0)

        # EMA momentum for confidence threshold
        self.conf_ema_m = getattr(self.args, 'freematch_conf_ema_m', 0.99)
        self.tau_min = getattr(self.args, 'freematch_tau_min', 0.5)
        self.tau_max = getattr(self.args, 'freematch_tau_max', 0.95)

        # Optional: confidence-based soft weighting
        self.use_soft_weight = getattr(self.args, 'freematch_soft_weight', True)

        # Optional: distribution alignment (logit shift)
        self.use_da = getattr(self.args, 'freematch_use_da', True)
        self.da_ema_m = getattr(self.args, 'freematch_da_ema_m', 0.99)
        self.da_eps = getattr(self.args, 'freematch_da_eps', 1e-4)

        # Per-client running stats
        self.client_conf_ema = {i: None for i in range(self.args.parti_num)}   # tau source
        self.client_posrate_ema = {i: None for i in range(self.args.parti_num)}  # DA source

        # Strong augmentation (same spirit as your FixMatch implementation)
        self.strong_aug = transforms.Compose([
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.2)),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value='random')
        ])

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        total_loss = 0.0
        for i in online_clients:
            client_loss = self._train_net(i, self.nets_list[i], priloader_list[i])
            total_loss += client_loss

        self.aggregate_nets(None)
        avg_round_loss = total_loss / len(online_clients) if len(online_clients) > 0 else 0.0
        return avg_round_loss

    @staticmethod
    def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        p = torch.clamp(p, eps, 1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    def _get_client_prior_target(self, client_id: int) -> float:
        """
        If you have per-client priors estimated from your PU partition, use them as DA target.
        Falls back to 0.5 if not available.
        """
        priors = getattr(self, 'client_priors', None)
        if isinstance(priors, dict) and client_id in priors:
            # clamp to avoid extreme logit shift
            return float(max(self.da_eps, min(1.0 - self.da_eps, priors[client_id])))
        return 0.5

    def _update_ema_scalar(self, old_val, new_val: float, m: float):
        if old_val is None:
            return float(new_val)
        return float(m * old_val + (1.0 - m) * float(new_val))

    def _train_net(self, index, net, train_loader):
        net = net.to(self.device)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.args.reg)

        avg_loss = 0.0
        for local_epoch in range(self.local_epoch):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                # Compatible with (images, labels) or (indices, images, labels)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    _, images, labels = batch
                else:
                    images, labels,true_labels, indices = batch

                images = images.to(self.device)
                labels = labels.to(self.device).float().view(-1)

                optimizer.zero_grad()

                # Split P / U based on PU-style labels: P=1, U=0
                mask_p = (labels == 1)
                mask_u = (labels == 0)

                images_p = images[mask_p]
                images_u = images[mask_u]

                loss_p = 0.0
                loss_u = 0.0

                # -----------------------
                # 1) Supervised on P
                # -----------------------
                if images_p.size(0) > 0:
                    outputs_p = net(images_p).view(-1)
                    targets_p = torch.ones_like(outputs_p)
                    loss_p = self.criterion_sup(outputs_p, targets_p)

                # -----------------------
                # 2) FreeMatch on U
                # -----------------------
                if images_u.size(0) > 0:
                    with torch.no_grad():
                        weak_logits = net(images_u).view(-1)

                        # --- Optional distribution alignment (logit shift) ---
                        # Estimate current positive rate and shift logits so that predicted pos-rate
                        # matches a target (prefer client prior if available).
                        if self.use_da:
                            probs_raw = torch.sigmoid(weak_logits)
                            batch_pos_rate = float(probs_raw.mean().item())

                            # update per-client pos-rate EMA
                            self.client_posrate_ema[index] = self._update_ema_scalar(
                                self.client_posrate_ema[index], batch_pos_rate, self.da_ema_m
                            )

                            posrate_ema = self.client_posrate_ema[index]
                            target_pos_rate = self._get_client_prior_target(index)

                            # logit shift: logit(p_target) - logit(p_current)
                            shift = self._safe_logit(torch.tensor(target_pos_rate, device=self.device), self.da_eps) - \
                                    self._safe_logit(torch.tensor(posrate_ema, device=self.device), self.da_eps)

                            weak_logits = weak_logits + shift

                        probs_u = torch.sigmoid(weak_logits)

                        # confidence for binary: max(p, 1-p)
                        conf_u = torch.maximum(probs_u, 1.0 - probs_u)
                        batch_conf_mean = float(conf_u.mean().item())

                        # update per-client confidence EMA -> tau
                        self.client_conf_ema[index] = self._update_ema_scalar(
                            self.client_conf_ema[index], batch_conf_mean, self.conf_ema_m
                        )

                        tau = float(self.client_conf_ema[index])
                        tau = max(self.tau_min, min(self.tau_max, tau))

                        # select high-confidence samples
                        valid_mask = (conf_u >= tau)

                        # pseudo-label (hard)
                        pseudo_targets = (probs_u >= 0.5).float()

                    if valid_mask.sum() > 0:
                        valid_images_u = images_u[valid_mask]
                        valid_targets_u = pseudo_targets[valid_mask].detach()

                        # strong augmentation
                        if valid_images_u.dim() == 2:
                            # tabular: noise + feature dropout
                            noise = torch.randn_like(valid_images_u) * 0.05
                            feat_mask = (torch.rand_like(valid_images_u) > 0.2).float()
                            strong_images_u = (valid_images_u + noise) * feat_mask
                        else:
                            strong_images_u = self.strong_aug(valid_images_u)

                        strong_logits = net(strong_images_u).view(-1)

                        # unsupervised loss (optionally weighted by confidence)
                        raw_unsup = self.criterion_unsup_none(strong_logits, valid_targets_u)

                        if self.use_soft_weight:
                            # weight in [0,1]: (conf - tau)/(1-tau)
                            conf_valid = conf_u[valid_mask].detach()
                            denom = max(1e-6, (1.0 - tau))
                            w = torch.clamp((conf_valid - tau) / denom, 0.0, 1.0)
                            loss_u = (raw_unsup * w).mean()
                        else:
                            loss_u = raw_unsup.mean()

                # -----------------------
                # 3) Total loss
                # -----------------------
                loss = loss_p + self.lambda_u * loss_u

                # Safety: skip NaN/Inf
                if isinstance(loss, torch.Tensor) and torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                    optimizer.step()
                    epoch_loss += float(loss.item())

            avg_loss = epoch_loss / max(1, len(train_loader))

        return avg_loss