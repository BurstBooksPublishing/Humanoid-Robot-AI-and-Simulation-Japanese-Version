import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T
import segmentation_models_pytorch as smp  # pip install segmentation-models-pytorch

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ハイパーパラメータ（外部JSONで上書き可能）
CFG = {
    "num_classes": 5,
    "backbone": "resnet50",
    "pretrained": True,
    "lr_head": 1e-3,
    "lr_finetune": 1e-4,
    "freeze_epochs": 10,
    "total_epochs": 30,
    "batch_size": 8,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "./runs/segmentation",
    "resume": None,
}

class SegDataset(torch.utils.data.Dataset):
    """簡易Dataset例（独自実装に差し替え）"""
    def __init__(self, root: str, split: str, transform=None):
        self.transform = transform
        # ここに画像/マスクパスリストを読み込む処理を実装
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = T.ToTensor()(Image.open(img_path).convert("RGB"))
        mask = torch.load(mask_path).long()  # (H,W)
        if self.transform:
            img = self.transform(img)
        return img, mask

def build_model(num_classes: int, backbone: str, pretrained: bool) -> nn.Module:
    # AuxClassifier無しで軽量化
    model = smp.FCN(
        encoder_name=backbone,
        encoder_weights="imagenet" if pretrained else None,
        classes=num_classes,
        activation=None,
    )
    return model

def freeze_encoder(model: nn.Module) -> None:
    # エンコーダ全凍結
    for p in model.encoder.parameters():
        p.requires_grad = False

def unfreeze_encoder(model: nn.Module) -> None:
    # 後半ブロックだけ解凍（メモリ節約）
    for p in model.encoder.layer4.parameters():
        p.requires_grad = True

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        logits = model(img)
        loss = criterion(logits, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * img.size(0)
    return running / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            logits = model(img)
            loss = criterion(logits, mask)
            running += loss.item() * img.size(0)
    return running / len(loader.dataset)

def main(cfg: Dict[str, Any]):
    os.makedirs(cfg["save_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["save_dir"])

    train_ds = SegDataset(root="./data", split="train", transform=T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    val_ds   = SegDataset(root="./data", split="val",   transform=T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=cfg["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(cfg["num_classes"], cfg["backbone"], cfg["pretrained"]).to(cfg["device"])
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # 1段階目：ヘッドのみ学習
    freeze_encoder(model)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr_head"])

    start_epoch = 0
    if cfg["resume"]:
        ckpt = torch.load(cfg["resume"], map_location=cfg["device"])
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    best_loss = float("inf")
    for epoch in range(start_epoch, cfg["total_epochs"]):
        if epoch == cfg["freeze_epochs"]:
            # 2段階目：fine-tune
            unfreeze_encoder(model)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr_finetune"])

        tr_loss = train_epoch(model, train_loader, criterion, optimizer, cfg["device"])
        val_loss = validate(model, val_loader, criterion, cfg["device"])
        writer.add_scalar("Loss/train", tr_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss, epoch)
        logger.info(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": cfg,
            }, Path(cfg["save_dir"]) / "best.pth")

    writer.close()

if __name__ == "__main__":
    # JSON上書き例: python train.py config.json
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            CFG.update(json.load(f))
    main(CFG)