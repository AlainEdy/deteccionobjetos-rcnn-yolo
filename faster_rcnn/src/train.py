"""
Fine-tuning de Faster R-CNN en un dataset personalizado
========================================================
Adapta el modelo pre-entrenado en COCO a tus propias clases.

Estructura esperada del dataset (formato COCO):
    data/
      images/
        train/  *.jpg
        val/    *.jpg
      annotations/
        instances_train.json
        instances_val.json
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF


# ── Dataset personalizado ─────────────────────────────────────────────────────
class CustomCOCODataset(Dataset):
    """
    Dataset compatible con anotaciones en formato COCO JSON.

    Parámetros
    ----------
    images_dir   : carpeta con las imágenes
    ann_file     : ruta al JSON de anotaciones COCO
    transforms   : transformaciones de data augmentation
    """

    def __init__(self, images_dir: str, ann_file: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(ann_file) as f:
            data = json.load(f)

        # Mapeo id → nombre de archivo
        self.images = {img["id"]: img for img in data["images"]}

        # Mapeo image_id → lista de anotaciones
        self.anns = {}
        for ann in data["annotations"]:
            iid = ann["image_id"]
            self.anns.setdefault(iid, []).append(ann)

        # Mapeo category_id → índice de clase (1-indexado, 0 = fondo)
        self.cat_map = {
            cat["id"]: i + 1
            for i, cat in enumerate(data["categories"])
        }
        self.class_names = ["__background__"] + [c["name"] for c in data["categories"]]
        self.image_ids = list(self.images.keys())

        print(f"[Dataset] {len(self.image_ids)} imágenes, {len(self.class_names)-1} clases")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = self.images_dir / img_info["file_name"]

        img = Image.open(img_path).convert("RGB")
        img_tensor = TF.to_tensor(img)

        anns = self.anns.get(img_id, [])
        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_map[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        target = {
            "boxes":    torch.tensor(boxes,   dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels":   torch.tensor(labels,  dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area":     torch.tensor(areas,   dtype=torch.float32),
            "iscrowd":  torch.tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target


# ── Construcción del modelo fine-tuning ──────────────────────────────────────
def build_model(num_classes: int) -> torch.nn.Module:
    """
    Carga Faster R-CNN pre-entrenado y reemplaza la cabeza de clasificación
    para adaptarse a `num_classes` clases personalizadas.

    num_classes debe incluir el fondo → clases_tuyas + 1
    """
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    )
    # Reemplazar el predictor final (in_features proviene de la capa anterior)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ── Entrenamiento ─────────────────────────────────────────────────────────────
def train_one_epoch(model, optimizer, loader, device, epoch):
    model.train()
    total_loss = 0.0

    for step, (images, targets) in enumerate(loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward — el modelo calcula las pérdidas internamente
        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses.item()

        if step % 10 == 0:
            breakdown = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"  Epoch {epoch} | Step {step:>4}/{len(loader)}  |  {breakdown}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluación simplificada: calcula la pérdida en validación."""
    model.train()   # necesario para obtener pérdidas
    val_loss = 0.0
    for images, targets in loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        val_loss += sum(loss_dict.values()).item()
    return val_loss / len(loader)


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Script de entrenamiento completo ─────────────────────────────────────────
def train(
    train_images_dir: str,
    train_ann_file:   str,
    val_images_dir:   str,
    val_ann_file:     str,
    output_dir:       str = "outputs/checkpoints",
    num_epochs:       int = 10,
    batch_size:       int = 2,
    lr:               float = 0.005,
    lr_step_size:     int = 3,
    lr_gamma:         float = 0.1,
    num_workers:      int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Dispositivo: {device}")

    # Datasets
    train_ds = CustomCOCODataset(train_images_dir, train_ann_file)
    val_ds   = CustomCOCODataset(val_images_dir,   val_ann_file)
    num_classes = len(train_ds.class_names)  # incluye fondo

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn
    )

    # Modelo
    model = build_model(num_classes).to(device)
    print(f"[Train] Modelo listo — {num_classes} clases: {train_ds.class_names}")

    # Optimizador + scheduler
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'═'*55}")
        print(f"  EPOCH {epoch}/{num_epochs}  |  LR = {scheduler.get_last_lr()}")
        print(f"{'═'*55}")

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_loss   = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"\n  → Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        # Guardar checkpoint
        ckpt_path = Path(output_dir) / f"faster_rcnn_epoch{epoch:02d}.pth"
        torch.save({
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "train_loss":  train_loss,
            "val_loss":    val_loss,
            "class_names": train_ds.class_names,
        }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = Path(output_dir) / "best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Mejor modelo guardado → {best_path}")

    print(f"\n[Train] Entrenamiento completo. Mejor val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    # ── Ejemplo de uso ────────────────────────────────────────────────────────
    train(
        train_images_dir="data/images/train",
        train_ann_file="data/annotations/instances_train.json",
        val_images_dir="data/images/val",
        val_ann_file="data/annotations/instances_val.json",
        output_dir="outputs/checkpoints",
        num_epochs=10,
        batch_size=2,
        lr=0.005,
    )
