"""
Utilidades: visualización, métricas IoU y helpers generales
============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
from pathlib import Path
from typing import Optional
import torch


# ── Generación de paleta de colores ──────────────────────────────────────────
def generate_color_palette(n: int) -> list:
    """Genera N colores distintos en formato BGR para OpenCV."""
    palette = []
    for i in range(n):
        hue = i / n
        rgb = hsv_to_rgb([hue, 0.8, 0.9])
        bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        palette.append(bgr)
    return palette


# ── IoU entre dos cajas ───────────────────────────────────────────────────────
def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calcula IoU entre dos bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


# ── Visualización matplotlib (mejor calidad que OpenCV) ───────────────────────
def visualize_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    masks: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    show: bool = True,
) -> plt.Figure:
    """
    Visualiza detecciones con matplotlib (alta calidad para reportes).

    Parámetros
    ----------
    image       : BGR numpy array
    boxes       : (N,4) int array  [x1,y1,x2,y2]
    scores      : (N,) float array
    labels      : (N,) int array
    class_names : list de nombres (incluyendo __background__ en [0])
    masks       : (N,H,W) binario, opcional (Mask R-CNN)
    output_path : guardar figura si se especifica
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    palette = generate_color_palette(max(len(class_names), 10))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(rgb)

    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        color_bgr = palette[label % len(palette)]
        color_rgb = tuple(c/255 for c in reversed(color_bgr))

        # Máscara semitransparente
        if masks is not None and i < len(masks):
            mask_overlay = np.zeros((*rgb.shape[:2], 4))
            mask_overlay[masks[i] == 1] = [*color_rgb, 0.45]
            ax.imshow(mask_overlay)

        # Bounding box
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color_rgb, facecolor="none"
        )
        ax.add_patch(rect)

        # Etiqueta
        name = class_names[label] if label < len(class_names) else str(label)
        ax.text(
            x1, y1 - 5, f"{name}: {score:.2f}",
            color="white", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.2", fc=color_rgb, ec="none", alpha=0.85)
        )

    n_obj = len(boxes)
    avg_conf = scores.mean() if n_obj else 0
    ax.set_title(
        f"Faster R-CNN  ·  {n_obj} objetos detectados  ·  conf. media: {avg_conf:.2f}",
        fontsize=12, pad=10
    )
    ax.axis("off")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Figura guardada → {output_path}")

    if show:
        plt.show()

    return fig


# ── Comparativa de variantes ──────────────────────────────────────────────────
def compare_models(
    image: np.ndarray,
    results: dict,
    output_path: Optional[str] = None,
    figsize: tuple = (18, 6),
) -> None:
    """
    Visualiza resultados de múltiples variantes en columnas.

    results = {
        "Faster R-CNN v1": {"annotated": img_bgr, "time_ms": 120, "n": 5},
        "Faster R-CNN v2": {"annotated": img_bgr, "time_ms": 180, "n": 6},
        "Mask R-CNN":      {"annotated": img_bgr, "time_ms": 250, "n": 6},
    }
    """
    n = len(results) + 1
    fig, axes = plt.subplots(1, n, figsize=figsize)

    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    for ax, (name, res) in zip(axes[1:], results.items()):
        ax.imshow(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\n{res['n']} objs | {res['time_ms']:.0f}ms", fontsize=10)
        ax.axis("off")

    plt.suptitle("Comparativa de variantes R-CNN", fontsize=13, y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Comparativa guardada → {output_path}")

    plt.show()


# ── Curva Precision-Recall ────────────────────────────────────────────────────
def plot_precision_recall(
    recalls: np.ndarray,
    precisions: np.ndarray,
    ap: float,
    class_name: str = "",
    output_path: Optional[str] = None,
) -> None:
    """Dibuja la curva Precision-Recall para una clase."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recalls, precisions, where="post", color="#378ADD", linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.2, color="#378ADD")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_title(f"Curva P-R  {class_name}  |  AP = {ap:.3f}", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Redimensionar manteniendo aspecto ─────────────────────────────────────────
def resize_keep_aspect(
    image: np.ndarray,
    max_side: int = 800,
) -> np.ndarray:
    """Redimensiona una imagen manteniendo la proporción."""
    h, w = image.shape[:2]
    scale = min(max_side / h, max_side / w, 1.0)
    if scale == 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ── Resumen estadístico de detecciones ───────────────────────────────────────
def detection_summary(result: dict, class_names: list) -> dict:
    """
    Devuelve un resumen estadístico del resultado de detección.
    """
    counts = {}
    for label, score in zip(result["labels"], result["scores"]):
        name = class_names[label] if label < len(class_names) else str(label)
        if name not in counts:
            counts[name] = {"count": 0, "scores": []}
        counts[name]["count"] += 1
        counts[name]["scores"].append(float(score))

    summary = {}
    for name, data in counts.items():
        summary[name] = {
            "count":    data["count"],
            "mean_conf": np.mean(data["scores"]),
            "max_conf":  np.max(data["scores"]),
        }
    return summary
