import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

class ImportanceWeightedTrainer:
    def __init__(
        self,
        model: nn.Module,
        domain_clf: nn.Module,
        feature_extractor: nn.Module,
        optimizer: torch.optim.Optimizer,
        task_loss: nn.Module,
        domain_loss: nn.Module,
        reg_loss: nn.Module,
        lambda_dom: float = 1.0,
        lambda_reg: float = 1e-4,
        max_weight: float = 10.0,
        eps: float = 1e-6,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.model = model.to(device)
        self.domain_clf = domain_clf.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.optimizer = optimizer
        self.task_loss = task_loss
        self.domain_loss = domain_loss
        self.reg_loss = reg_loss
        self.lambda_dom = lambda_dom
        self.lambda_reg = lambda_reg
        self.max_weight = max_weight
        self.eps = eps
        self.device = device

    def run_epoch(
        self,
        synth_loader: DataLoader,
        real_loader: DataLoader,
        steps_per_epoch: int
    ) -> Tuple[float, float, float]:
        self.model.train()
        self.domain_clf.train()
        self.feature_extractor.train()

        synth_iter = iter(synth_loader)
        real_iter = iter(real_loader)

        total_task = 0.0
        total_domain = 0.0
        total_reg = 0.0

        for _ in range(steps_per_epoch):
            xs, ys = next(synth_iter)
            xr, yr = next(real_iter)
            xs, ys, xr, yr = xs.to(self.device), ys.to(self.device), xr.to(self.device), yr.to(self.device)

            x_all = torch.cat([xs, xr], dim=0)
            with torch.no_grad():
                feats = self.feature_extractor(x_all)
                d_logits = self.domain_clf(feats)
                p_real = torch.sigmoid(d_logits).squeeze()
                p_sim = 1.0 - p_real
                w = (p_real / (p_sim + self.eps)).clamp(max=self.max_weight)

            preds = self.model(x_all)
            loss_task = self.task_loss(preds, torch.cat([ys, yr], dim=0))
            loss_weighted = (w * loss_task).mean()

            feats_s = self.feature_extractor(xs)
            feats_r = self.feature_extractor(xr)
            loss_domain = self.domain_loss(self.domain_clf(feats_s), torch.zeros(xs.size(0), device=self.device)) + \
                          self.domain_loss(self.domain_clf(feats_r), torch.ones(xr.size(0), device=self.device))

            loss_reg = self.reg_loss(self.model)

            loss = loss_weighted + self.lambda_dom * loss_domain + self.lambda_reg * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_task += loss_weighted.item()
            total_domain += loss_domain.item()
            total_reg += loss_reg.item()

        return total_task / steps_per_epoch, total_domain / steps_per_epoch, total_reg / steps_per_epoch