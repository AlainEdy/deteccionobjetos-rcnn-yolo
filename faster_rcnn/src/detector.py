"""
Faster R-CNN Object Detector
============================
Detección de objetos usando Faster R-CNN con backbone ResNet-50 FPN
pre-entrenado en COCO (80 clases).

Arquitectura:
  Imagen → Backbone CNN (ResNet50-FPN) → RPN → RoI Pooling → Clasificador + BBox Regressor
"""

import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import time
from pathlib import Path
from typing import Optional


# ── Clases COCO ────────────────────────────────────────────────────────────────
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Paleta de colores BGR para cada clase
np.random.seed(42)
COLORS = np.random.randint(50, 220, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


class FasterRCNNDetector:
    """
    Detector de objetos basado en Faster R-CNN.

    Parámetros
    ----------
    model_variant : str
        'v1'  → FasterRCNN_ResNet50_FPN   (más rápido)
        'v2'  → FasterRCNN_ResNet50_FPN_V2 (más preciso)
        'mask'→ Mask R-CNN (segmentación de instancias)
    confidence_threshold : float
        Umbral mínimo de confianza para conservar una detección (0–1).
    nms_iou_threshold : float
        Umbral IoU para Non-Maximum Suppression.
    device : str | None
        'cuda', 'mps', 'cpu' o None (auto-detect).
    """

    def __init__(
        self,
        model_variant: str = "v2",
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.conf_thresh = confidence_threshold
        self.nms_iou    = nms_iou_threshold
        self.variant    = model_variant

        # ── Dispositivo ───────────────────────────────────────────────────────
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        print(f"[Detector] Usando dispositivo: {self.device}")

        # ── Modelo ────────────────────────────────────────────────────────────
        self.model = self._load_model()
        self.model.eval()
        self.model.to(self.device)
        print(f"[Detector] Modelo '{model_variant}' listo ✓")

    # ── Carga del modelo ──────────────────────────────────────────────────────
    def _load_model(self):
        print(f"[Detector] Cargando pesos pre-entrenados COCO…")
        if self.variant == "v1":
            return fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
            )
        elif self.variant == "v2":
            return fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            )
        elif self.variant == "mask":
            return maskrcnn_resnet50_fpn(
                weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
            )
        else:
            raise ValueError(f"Variante desconocida: {self.variant}. Usa 'v1', 'v2' o 'mask'.")

    # ── Pre-procesamiento ─────────────────────────────────────────────────────
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """BGR (OpenCV) → RGB tensor normalizado [0, 1]."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(rgb)          # HWC → CHW, uint8→float32/255
        return tensor.to(self.device)

    # ── NMS manual (opcional, torchvision ya lo aplica internamente) ──────────
    def _apply_nms(self, boxes, scores, labels):
        """Aplica NMS adicional por clase si se desea mayor control."""
        from torchvision.ops import nms
        keep_all = []
        for cls in labels.unique():
            mask = labels == cls
            idx  = nms(boxes[mask], scores[mask], self.nms_iou)
            keep_all.append(mask.nonzero(as_tuple=True)[0][idx])
        if not keep_all:
            return torch.tensor([], dtype=torch.long)
        return torch.cat(keep_all)

    # ── Inferencia ────────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray) -> dict:
        """
        Ejecuta detección sobre una imagen BGR (numpy array).

        Retorna
        -------
        dict con claves:
            boxes      : np.ndarray (N, 4)  [x1, y1, x2, y2]
            scores     : np.ndarray (N,)
            labels     : np.ndarray (N,)    índices COCO
            class_names: list[str]
            masks      : np.ndarray|None    (N, H, W) si es Mask R-CNN
            time_ms    : float
        """
        tensor = self._preprocess(image)

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model([tensor])
        t1 = time.perf_counter()

        out = outputs[0]
        boxes  = out["boxes"].cpu()
        scores = out["scores"].cpu()
        labels = out["labels"].cpu()

        # Filtrar por confianza
        keep = scores >= self.conf_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Máscaras (solo Mask R-CNN)
        masks = None
        if "masks" in out:
            masks = out["masks"].cpu()[keep].squeeze(1).numpy()
            masks = (masks > 0.5).astype(np.uint8)

        class_names = [
            COCO_CLASSES[l] if l < len(COCO_CLASSES) else "???"
            for l in labels.numpy()
        ]

        return {
            "boxes":       boxes.numpy().astype(int),
            "scores":      scores.numpy(),
            "labels":      labels.numpy(),
            "class_names": class_names,
            "masks":       masks,
            "time_ms":     (t1 - t0) * 1000,
        }

    # ── Dibujo de resultados ──────────────────────────────────────────────────
    def draw(self, image: np.ndarray, result: dict, alpha: float = 0.4) -> np.ndarray:
        """
        Dibuja bounding boxes, etiquetas y máscaras (si existen) sobre la imagen.

        Parámetros
        ----------
        image  : imagen BGR original
        result : salida de self.detect()
        alpha  : transparencia de máscaras de segmentación
        """
        canvas = image.copy()
        overlay = canvas.copy()

        for i, (box, score, label, name) in enumerate(zip(
            result["boxes"], result["scores"],
            result["labels"], result["class_names"]
        )):
            if name in ("N/A", "__background__"):
                continue

            color = tuple(int(c) for c in COLORS[label % len(COLORS)])
            x1, y1, x2, y2 = box

            # Máscara de segmentación (Mask R-CNN)
            if result["masks"] is not None:
                mask = result["masks"][i]
                overlay[mask == 1] = color

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con fondo
            label_text = f"{name}: {score:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            cv2.rectangle(canvas, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                canvas, label_text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

        # Mezclar overlay de máscaras
        if result["masks"] is not None:
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

        # Info general
        info = (f"Objetos: {len(result['boxes'])}  |  "
                f"Tiempo: {result['time_ms']:.1f} ms  |  "
                f"Modelo: Faster R-CNN {self.variant.upper()}")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (30, 30, 30), -1)
        cv2.putText(canvas, info, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        return canvas

    # ── Procesar un archivo de imagen ─────────────────────────────────────────
    def process_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> dict:
        """Carga, detecta, dibuja y (opcionalmente) guarda/muestra el resultado."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo leer: {image_path}")

        result = self.detect(img)
        annotated = self.draw(img, result)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            print(f"[Detector] Imagen guardada → {output_path}")

        if show:
            cv2.imshow("Faster R-CNN Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self._print_results(result, image_path)
        return result

    # ── Procesar video ────────────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """Detecta objetos frame a frame en un video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        print(f"[Video] {total} frames a {fps:.1f} FPS — procesando…")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            result    = self.detect(frame)
            annotated = self.draw(frame, result)

            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow("Faster R-CNN — Video", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Video] Interrumpido por el usuario.")
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  Frame {frame_idx}/{total}  |  {result['time_ms']:.1f} ms/frame")

        cap.release()
        if writer:
            writer.release()
            print(f"[Video] Guardado → {output_path}")
        cv2.destroyAllWindows()

    # ── Resumen de resultados ─────────────────────────────────────────────────
    def _print_results(self, result: dict, source: str = "") -> None:
        print(f"\n{'─'*55}")
        print(f"  Fuente : {source}")
        print(f"  Modelo : Faster R-CNN {self.variant.upper()} (COCO)")
        print(f"  Tiempo : {result['time_ms']:.1f} ms")
        print(f"  Objetos detectados: {len(result['boxes'])}")
        print(f"{'─'*55}")
        for name, score, box in zip(
            result["class_names"], result["scores"], result["boxes"]
        ):
            x1, y1, x2, y2 = box
            print(f"  [{score:.3f}]  {name:<20}  box=[{x1},{y1},{x2},{y2}]")
        print(f"{'─'*55}\n")
