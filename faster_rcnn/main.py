"""
main.py — Punto de entrada principal
=====================================
Detección de objetos con Faster R-CNN / Mask R-CNN

Uso rápido
----------
  # Imagen única
  python main.py --image ¿

  # Video
  python main.py --video clip.mp4 --output outputs/clip_det.mp4

  # Comparar variantes
  python main.py --image foto.jpg --compare

  # Mask R-CNN (segmentación)
  python main.py --image foto.jpg --model mask

  # Ajustar umbral y variante
  python main.py --image foto.jpg --model v2 --confidence 0.6 --max-detections 30
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch

# Importar módulos del proyecto
sys.path.insert(0, str(Path(__file__).parent))
from src.detector import FasterRCNNDetector, COCO_CLASSES
from utils.visualization import (
    visualize_detections,
    compare_models,
    resize_keep_aspect,
    detection_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Detección de objetos con Faster R-CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image",        type=str,   help="Ruta a imagen de entrada")
    p.add_argument("--video",        type=str,   help="Ruta a video de entrada")
    p.add_argument("--output",       type=str,   help="Ruta de salida (imagen o video)")
    p.add_argument("--model",        type=str,   default="v2",
                   choices=["v1", "v2", "mask"],
                   help="Variante del modelo: v1 | v2 | mask (default: v2)")
    p.add_argument("--confidence",   type=float, default=0.5,
                   help="Umbral de confianza 0-1 (default: 0.5)")
    p.add_argument("--nms-iou",      type=float, default=0.5,
                   help="Umbral NMS IoU (default: 0.5)")
    p.add_argument("--max-side",     type=int,   default=1024,
                   help="Lado máximo de la imagen de entrada (default: 1024)")
    p.add_argument("--compare",      action="store_true",
                   help="Comparar los 3 modelos en la misma imagen")
    p.add_argument("--no-display",   action="store_true",
                   help="No mostrar ventana de visualización")
    p.add_argument("--device",       type=str,   default=None,
                   help="Forzar dispositivo: cuda | mps | cpu")
    p.add_argument("--max-frames",   type=int,   default=None,
                   help="Límite de frames a procesar en video")
    return p.parse_args()


# ── Detección en imagen única ─────────────────────────────────────────────────
def run_image(args):
    img = cv2.imread(args.image)
    if img is None:
        print(f"[Error] No se pudo leer: {args.image}")
        sys.exit(1)

    img = resize_keep_aspect(img, args.max_side)

    if args.compare:
        _run_compare(img, args)
        return

    detector = FasterRCNNDetector(
        model_variant=args.model,
        confidence_threshold=args.confidence,
        nms_iou_threshold=args.nms_iou,
        device=args.device,
    )

    result = detector.detect(img)
    annotated = detector.draw(img, result)

    # Guardar si se especificó salida
    output = args.output or f"outputs/det_{Path(args.image).stem}.jpg"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output, annotated)
    print(f"[Main] Resultado guardado → {output}")

    # Visualización matplotlib de alta calidad
    viz_path = str(Path(output).parent / f"{Path(output).stem}_hq.png")
    visualize_detections(
        img,
        result["boxes"], result["scores"],
        result["labels"], COCO_CLASSES,
        masks=result.get("masks"),
        output_path=viz_path,
        show=not args.no_display,
    )

    # Resumen por clase
    summary = detection_summary(result, COCO_CLASSES)
    print("\n[Resumen por clase]")
    print(f"  {'Clase':<20} {'N':>4}  {'Conf. Media':>12}  {'Conf. Máx':>10}")
    print(f"  {'─'*52}")
    for cls_name, data in sorted(summary.items(), key=lambda x: -x[1]["count"]):
        print(f"  {cls_name:<20} {data['count']:>4}  {data['mean_conf']:>12.3f}  {data['max_conf']:>10.3f}")


# ── Comparar variantes ────────────────────────────────────────────────────────
def _run_compare(img, args):
    variants = [("v1", "Faster R-CNN v1"), ("v2", "Faster R-CNN v2"), ("mask", "Mask R-CNN")]
    results = {}

    for variant_id, variant_name in variants:
        print(f"\n{'─'*45}")
        print(f"  Ejecutando {variant_name}…")
        det = FasterRCNNDetector(
            model_variant=variant_id,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        t0 = time.perf_counter()
        result = det.detect(img)
        elapsed = (time.perf_counter() - t0) * 1000
        annotated = det.draw(img, result)

        results[variant_name] = {
            "annotated": annotated,
            "time_ms":   elapsed,
            "n":         len(result["boxes"]),
        }
        print(f"  → {len(result['boxes'])} objetos en {elapsed:.1f} ms")

    output = args.output or f"outputs/compare_{Path(args.image).stem}.png"
    compare_models(img, results, output_path=output, figsize=(18, 6))


# ── Detección en video ────────────────────────────────────────────────────────
def run_video(args):
    detector = FasterRCNNDetector(
        model_variant=args.model,
        confidence_threshold=args.confidence,
        nms_iou_threshold=args.nms_iou,
        device=args.device,
    )

    output = args.output or f"outputs/det_{Path(args.video).stem}.mp4"
    detector.process_video(
        video_path=args.video,
        output_path=output,
        show=not args.no_display,
        max_frames=args.max_frames,
    )


# ── Demo rápida sin argumentos ────────────────────────────────────────────────
def run_demo():
    """Demo con imagen de prueba descargada de la web."""
    import urllib.request
    import os

    url      = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Shopping_Center_Magna_Plaza_Amsterdam_2012.jpg/1280px-Shopping_Center_Magna_Plaza_Amsterdam_2012.jpg"
    img_path = "data/images/demo.jpg"
    Path("data/images").mkdir(parents=True, exist_ok=True)

    if not os.path.exists(img_path):
        print(f"[Demo] Descargando imagen de prueba…")
        try:
            urllib.request.urlretrieve(url, img_path)
        except Exception:
            print("[Demo] No se pudo descargar. Coloca una imagen en data/images/demo.jpg")
            return

    detector = FasterRCNNDetector(model_variant="v2", confidence_threshold=0.5)
    result   = detector.process_image(img_path, output_path="outputs/demo_result.jpg", show=True)

    print("\n[Demo] Detecciones completadas.")
    print(f"       Objetos encontrados: {len(result['boxes'])}")
    print(f"       Tiempo de inferencia: {result['time_ms']:.1f} ms")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if args.video:
        run_video(args)
    elif args.image:
        run_image(args)
    else:
        print("[Main] Sin argumentos — ejecutando demo…\n")
        run_demo()
