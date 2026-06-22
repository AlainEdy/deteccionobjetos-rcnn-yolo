"""
Evaluación de Faster R-CNN con métricas COCO (mAP)
===================================================
Calcula mAP@0.5, mAP@0.5:0.95, AR y otras métricas estándar.
"""

import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("[Eval] pycocotools no disponible — se usará evaluación simplificada")


# ── Evaluación con pycocotools (oficial) ──────────────────────────────────────
class COCOEvaluator:
    """
    Evaluador oficial COCO. Calcula:
      - mAP @ IoU=0.50:0.95
      - mAP @ IoU=0.50  (PASCAL VOC)
      - mAP @ IoU=0.75
      - AR @  1 / 10 / 100 detecciones
    """

    def __init__(self, ann_file: str):
        if not HAS_PYCOCOTOOLS:
            raise ImportError("Instala pycocotools: pip install pycocotools")
        self.coco_gt   = COCO(ann_file)
        self.results   = []
        self.image_ids = []

    def update(self, image_ids, outputs):
        """
        Agrega predicciones al evaluador.

        image_ids : list[int]   — IDs de imagen del batch
        outputs   : list[dict]  — salida de model(images)
        """
        for img_id, out in zip(image_ids, outputs):
            boxes  = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()
            labels = out["labels"].cpu().numpy()

            # Convertir xyxy → xywh (formato COCO)
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                self.results.append({
                    "image_id":    int(img_id),
                    "category_id": int(label),
                    "bbox":        [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    "score":       float(score),
                })
            self.image_ids.append(int(img_id))

    def summarize(self):
        """Calcula y muestra las métricas finales."""
        if not self.results:
            print("[Eval] Sin resultados para evaluar.")
            return {}

        coco_dt  = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.params.imgIds = list(set(self.image_ids))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {
            "mAP_50_95": float(stats[0]),
            "mAP_50":    float(stats[1]),
            "mAP_75":    float(stats[2]),
            "AR_1":      float(stats[6]),
            "AR_10":     float(stats[7]),
            "AR_100":    float(stats[8]),
        }


# ── Evaluación simplificada (sin pycocotools) ─────────────────────────────────
def compute_iou(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """IoU entre box1 (M,4) y box2 (N,4)."""
    x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Área bajo la curva Precision-Recall (interpolación en 11 puntos)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += (p.max() if len(p) else 0.0)
    return ap / 11.0


def evaluate_map(
    predictions: list,
    ground_truths: list,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> dict:
    """
    Calcula mAP simplificado para una lista de predicciones y GT.

    predictions   : list de dict {'boxes', 'scores', 'labels'}
    ground_truths : list de dict {'boxes', 'labels'}
    """
    aps = []

    for cls in range(1, num_classes):
        tp_all, fp_all, scores_all = [], [], []
        total_gt = 0

        for pred, gt in zip(predictions, ground_truths):
            gt_mask   = gt["labels"] == cls
            pred_mask = pred["labels"] == cls

            gt_boxes   = gt["boxes"][gt_mask]
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]

            total_gt += len(gt_boxes)

            if len(pred_boxes) == 0:
                continue

            order = np.argsort(-pred_scores)
            pred_boxes  = pred_boxes[order]
            pred_scores = pred_scores[order]
            scores_all.extend(pred_scores.tolist())

            matched = np.zeros(len(gt_boxes), dtype=bool)

            for pb in pred_boxes:
                if len(gt_boxes) == 0:
                    tp_all.append(0); fp_all.append(1)
                    continue
                ious = compute_iou(pb[None], gt_boxes)[0]
                best_idx = np.argmax(ious)
                if ious[best_idx] >= iou_threshold and not matched[best_idx]:
                    tp_all.append(1); fp_all.append(0)
                    matched[best_idx] = True
                else:
                    tp_all.append(0); fp_all.append(1)

        if total_gt == 0:
            continue

        order = np.argsort(-np.array(scores_all)) if scores_all else []
        tp = np.array(tp_all)[order] if len(order) else np.array([])
        fp = np.array(fp_all)[order] if len(order) else np.array([])

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls    = tp_cum / total_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        aps.append(compute_ap(recalls, precisions))

    mAP = float(np.mean(aps)) if aps else 0.0
    return {"mAP_50": mAP, "per_class_AP": aps}


# ── Evaluación en un DataLoader ───────────────────────────────────────────────
@torch.no_grad()
def run_evaluation(
    model,
    loader: DataLoader,
    device: torch.device,
    ann_file: str = None,
    confidence_threshold: float = 0.5,
) -> dict:
    """
    Ejecuta el modelo sobre `loader` y devuelve métricas.

    Si `ann_file` está disponible y pycocotools instalado → usa COCOeval.
    En caso contrario → evaluación simplificada.
    """
    model.eval()
    all_preds, all_gts = [], []
    evaluator = COCOEvaluator(ann_file) if (HAS_PYCOCOTOOLS and ann_file) else None

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            keep = out["scores"] >= confidence_threshold

            pred = {
                "boxes":  out["boxes"][keep].cpu().numpy(),
                "scores": out["scores"][keep].cpu().numpy(),
                "labels": out["labels"][keep].cpu().numpy(),
            }
            gt = {
                "boxes":  tgt["boxes"].numpy(),
                "labels": tgt["labels"].numpy(),
            }
            all_preds.append(pred)
            all_gts.append(gt)

            if evaluator:
                img_ids = tgt["image_id"].tolist()
                evaluator.update(img_ids, [{
                    "boxes":  out["boxes"][keep],
                    "scores": out["scores"][keep],
                    "labels": out["labels"][keep],
                }])

    if evaluator:
        print("\n[Eval] Métricas COCO oficiales:")
        return evaluator.summarize()
    else:
        num_classes = max(
            max(p["labels"].max() for p in all_preds if len(p["labels"])),
            max(g["labels"].max() for g in all_gts   if len(g["labels"])),
        ) + 1
        metrics = evaluate_map(all_preds, all_gts, int(num_classes))
        print(f"\n[Eval] mAP@0.50 = {metrics['mAP_50']:.4f}")
        return metrics


if __name__ == "__main__":
    print("Módulo de evaluación listo.")
    print("Usa run_evaluation(model, loader, device) para obtener métricas.")
