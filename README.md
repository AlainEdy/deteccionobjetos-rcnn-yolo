# Detección de Objetos en Tiempo Real e Imágenes Estáticas

Este repositorio contiene las pruebas y el análisis comparativo de rendimiento entre dos de las arquitecturas más utilizadas en visión por computadora: **YOLO (v8/v11n) de Ultralytics** y **Faster R-CNN de Torchvision**. Las pruebas evalúan el comportamiento de los modelos frente a imágenes estáticas y flujos de video en tiempo real (Webcam) utilizando procesamiento por CPU.

---

## Resumen de Hallazgos y Métricas

El análisis de los resultados demuestra el balance clásico en visión por computadora: **Velocidad vs. Precisión** y **Análisis Estático vs. Análisis Dinámico (Video)**.

| Métrica / Comportamiento | Faster R-CNN (`torchvision`) | YOLOv8 / YOLO11 (`ultralytics`) |
| :--- | :--- | :--- |
| **Confianza (Imagen Estática)** | Alta / Excelente (**100%**) | Alta / Buena (**93%**) |
| **Confianza (Video en Vivo)** | Alta / Estable (**86.8% - 99.1%**) | Moderada / Fluctuante (**53% - 91%**) |
| **Velocidad de Inferencia (CPU)**| Inviable para tiempo real (**1.2 FPS**) | Excelente para tiempo real (**36.8 FPS**) |
| **Estabilidad de la Caja** | Muy estable (Rígida, sin parpadeo) | Inestable (Efecto de temblor/parpadeo) |
| **Tipo de Arquitectura** | Dos etapas (*Two-stage detector*) | Una sola etapa (*One-stage detector*) |

---

## Análisis Técnico del Comportamiento

### 1. Fase Estática (Imágenes Fijas)
Al procesar imágenes fijas (como las pruebas con el *Golden Retriever*), el tiempo de cómputo por fotograma no afecta la experiencia del usuario.
* **Faster R-CNN**: Al proponer regiones de interés antes de clasificar, extrae características más profundas obteniendo una confianza perfecta del **1.00 (100%)**.
* **YOLO11n**: Como versión "nano", optimiza la ligereza matemática. Consigue un excelente **93%** de confianza de forma inmediata, ideal para hardware limitado.

### 2. Fase Dinámica (Video en Tiempo Real)
En flujos de video por cámara web, la métrica crítica son los **Fotogramas por Segundo (FPS)**.
* **Faster R-CNN (`torchvision_resnet50`)**: Da como resultado **1.2 FPS**. El procesador (CPU) se satura calculando las regiones propuestas por la red interna ResNet50 en cada cuadro. La pantalla se actualiza cada segundo, siendo inviable para producción en tiempo real sin GPU.
* **YOLOv8 (`ultralytics`)**: Alcanza **36.8 FPS**. Su arquitectura procesa la imagen completa en una sola pasada (*You Only Look Once*), superando con creces el estándar de fluidez (30 FPS) nativamente en CPU.

---

## El Fenómeno del "Parpadeo" en YOLOv8

Durante las pruebas en video con YOLOv8, se observa que **el recuadro de detección tiembla o parpadea constantemente**, a diferencia de la rigidez de Faster R-CNN. Esto ocurre debido a dos factores:

1. **Caída y Fluctuación de Confianza:** En video, la confianza de YOLOv8 para el objeto `cell phone` cae al **53%** (frente al 86.8% de Faster R-CNN). Cualquier micro-cambio de iluminación o movimiento hace que la confianza baje momentáneamente del umbral mínimo de visualización, provocando que la caja desaparezca y reaparezca ("parpadee").
2. **Falta de Consistencia Temporal:** El script analiza cada fotograma como una foto aislada sin memoria del cuadro anterior. Al ser una red más pequeña, es más sensible al ruido del sensor de la cámara web, variando las coordenadas de la caja a cada instante.

---

## Tecnologías Utilizadas

### Implementación Faster R-CNN
```python
from torchvision.models.detection import fastercnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
```

### Implementación YOLO
```python
from ultralytics import YOLO
```

---
