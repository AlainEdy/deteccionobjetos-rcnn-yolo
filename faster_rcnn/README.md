# Detección de Objetos con Faster R-CNN

Implementación completa de detección de objetos usando **Faster R-CNN**, **Faster R-CNN v2** y **Mask R-CNN** con PyTorch y torchvision. Incluye inferencia, fine-tuning, evaluación mAP y visualización.

---

## Arquitectura del Proyecto

```
faster_rcnn/
├── main.py                    # Punto de entrada principal
├── requirements.txt
├── src/
│   ├── detector.py            # Clase FasterRCNNDetector (inferencia + dibujo)
│   ├── train.py               # Fine-tuning en dataset personalizado (COCO format)
│   └── evaluate.py            # Evaluación mAP con pycocotools
├── utils/
│   └── visualization.py       # Visualización matplotlib, comparativas, P-R curves
├── data/
│   ├── images/                # Imágenes de entrada
│   └── annotations/           # JSONs COCO para fine-tuning
├── outputs/                   # Resultados y checkpoints
└── notebooks/
    └── exploracion.ipynb      # Notebook interactivo
```

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# PyTorch con CUDA (si tienes GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Uso Rápido

### Detección en imagen
```bash
# Básico (modelo v2, umbral 0.5)
python main.py --image foto.jpg

# Mask R-CNN (segmentación de instancias)
python main.py --image foto.jpg --model mask

# Umbral personalizado
python main.py --image foto.jpg --confidence 0.7

# Comparar las 3 variantes
python main.py --image foto.jpg --compare

# Sin ventana emergente, guardar resultado
python main.py --image foto.jpg --output results/out.jpg --no-display
```

### Detección en video
```bash
python main.py --video clip.mp4 --output outputs/clip_det.mp4
python main.py --video clip.mp4 --model mask --max-frames 100
```

### Demo automática
```bash
python main.py   # descarga imagen y ejecuta demo
```

---

## Uso desde Python

```python
from src.detector import FasterRCNNDetector
import cv2

# Cargar detector
detector = FasterRCNNDetector(
    model_variant='v2',          # 'v1' | 'v2' | 'mask'
    confidence_threshold=0.5,    # filtrar por confianza
    nms_iou_threshold=0.5,       # NMS
    device=None,                 # auto-detect (cuda/mps/cpu)
)

# Detectar en imagen
img    = cv2.imread('foto.jpg')
result = detector.detect(img)

# result = {
#   'boxes':       np.ndarray (N, 4)  [x1, y1, x2, y2]
#   'scores':      np.ndarray (N,)
#   'labels':      np.ndarray (N,)
#   'class_names': list[str]
#   'masks':       np.ndarray (N,H,W) | None
#   'time_ms':     float
# }

# Dibujar y guardar
annotated = detector.draw(img, result)
cv2.imwrite('resultado.jpg', annotated)

# Procesar un archivo completo de una vez
detector.process_image('foto.jpg', output_path='out.jpg', show=True)

# Procesar video
detector.process_video('clip.mp4', output_path='clip_det.mp4')
```

---

## Variantes de Modelo

| Variante | Backbone | mAP@0.5:0.95 | Velocidad | Segmentación |
|----------|----------|-------------|-----------|--------------|
| `v1` | ResNet-50 FPN | 37.0 | ★★★★☆ | No |
| `v2` | ResNet-50 FPN v2 | 46.7 | ★★★☆☆ | No |
| `mask` | ResNet-50 FPN | 44.4 | ★★☆☆☆ | ✓ (máscaras por píxel) |

---

## Arquitectura Faster R-CNN

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────┐
│   Imagen     │───▶│  Backbone CNN     │───▶│ Feature Maps │
│ (H×W×3)     │    │  ResNet50 + FPN   │    │ (multi-scale)│
└──────────────┘    └───────────────────┘    └──────┬───────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │ RPN                 │
                                          │ (Region Proposal    │
                                          │  Network)           │
                                          │ ─ anchors 3 escalas │
                                          │ ─ ~2000 proposals   │
                                          └─────────┬──────────┘
                                                    │
                                          ┌─────────▼──────────┐
                                          │ RoI Pooling / Align │
                                          │ (7×7 fijo)          │
                                          └─────────┬──────────┘
                                                    │
                              ┌─────────────────────┼──────────────────────┐
                              │                     │                      │
                    ┌─────────▼──────┐   ┌──────────▼──────┐   ┌──────────▼──────┐
                    │   Clasificador  │   │  BBox Regressor  │   │ Máscara (Mask)  │
                    │   N clases + bg │   │  4 coords ×clase │   │  (Mask R-CNN)   │
                    └────────────────┘   └─────────────────┘   └─────────────────┘
```

### Componentes clave

- **Backbone (ResNet50-FPN):** Extrae feature maps multi-escala. La Feature Pyramid Network combina capas de distintas resoluciones para detectar objetos de cualquier tamaño.

- **RPN:** Red convolucional ligera que propone regiones candidatas. Usa anchors de distintas escalas y ratios de aspecto. Genera ~2000 proposals por imagen.

- **RoI Pooling/Align:** Normaliza cada proposal a un tensor 7×7 independientemente de su tamaño original.

- **Cabeza de clasificación:** Dos ramas lineales — una predice la clase (cross-entropy loss) y otra refina las coordenadas del bounding box (Smooth L1 loss).

- **NMS:** Post-proceso que elimina detecciones redundantes conservando la de mayor score.

---

## Fine-tuning en Dataset Propio

### 1. Preparar anotaciones (formato COCO)
```json
{
  "images":      [{"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480}],
  "categories":  [{"id": 1, "name": "gato"}, {"id": 2, "name": "perro"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [100, 150, 80, 60], "area": 4800, "iscrowd": 0}]
}
```

Herramientas recomendadas para anotar: **CVAT**, **Label Studio**, **Roboflow**.

### 2. Entrenar
```python
from src.train import train

train(
    train_images_dir='data/images/train',
    train_ann_file='data/annotations/instances_train.json',
    val_images_dir='data/images/val',
    val_ann_file='data/annotations/instances_val.json',
    num_epochs=15,
    batch_size=2,
    lr=0.005,
)
```

### 3. Inferencia con modelo entrenado
```python
import torch
from src.train import build_model

model = build_model(num_classes=3)  # tus clases + fondo
ckpt  = torch.load('outputs/checkpoints/best_model.pth')
model.load_state_dict(ckpt)
model.eval()
```

---

## Evaluación mAP

```python
from src.evaluate import run_evaluation
from torch.utils.data import DataLoader
from src.train import CustomCOCODataset, collate_fn

val_ds     = CustomCOCODataset('data/images/val', 'data/annotations/instances_val.json')
val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn)

metrics = run_evaluation(
    model=detector.model,
    loader=val_loader,
    device=detector.device,
    ann_file='data/annotations/instances_val.json',  # opcional, activa pycocotools
)
# → {'mAP_50_95': 0.412, 'mAP_50': 0.621, 'mAP_75': 0.443, ...}
```

---

## Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| Python | 3.9+ | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | — | 6 GB (RTX 3060+) |
| Almacenamiento | 2 GB | 5 GB |

---

## Notebook Interactivo

```bash
cd notebooks
jupyter notebook exploracion.ipynb
```

Incluye: carga de imágenes, visualización de detecciones, comparativa de variantes, análisis de arquitectura y esquema de fine-tuning.

---

## Referencias

- Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (NeurIPS 2015)
- He et al., *Mask R-CNN* (ICCV 2017)
- Lin et al., *Feature Pyramid Networks for Object Detection* (CVPR 2017)
- [torchvision detection models](https://pytorch.org/vision/stable/models.html#object-detection)
