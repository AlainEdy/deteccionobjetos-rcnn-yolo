# Detector de Objetos YOLO11n — Local

Aplicación web de detección de objetos que corre 100% en tu máquina.
Sin internet (después de la instalación), sin API key, completamente gratis.

## Requisitos

- Windows 10/11
- Python 3.8 o superior → https://www.python.org/downloads/
  ⚠️ Al instalar Python, marca la casilla **"Add Python to PATH"**

## Cómo usar

1. Descomprime esta carpeta donde quieras
2. Haz doble clic en **iniciar.bat**
3. La primera vez instala dependencias y descarga el modelo (~6 MB) — tarda ~1 minuto
4. Se abre el navegador en http://localhost:5000 automáticamente
5. Sube una imagen y haz clic en "Detectar"

## Estructura del proyecto

```
yolo_detector/
│
├── app.py              ← Servidor Flask + lógica YOLO
├── requirements.txt    ← Dependencias Python
├── iniciar.bat         ← Lanzador para Windows
│
├── templates/
│   └── index.html      ← Interfaz web
│
└── static/
    ├── uploads/        ← Imágenes temporales de entrada
    └── results/        ← Imágenes temporales con anotaciones
```

## Parámetros ajustables

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| Umbral de confianza | Mínimo de certeza para mostrar una detección | 30% |
| Umbral IoU (NMS) | Controla la superposición entre boxes | 45% |

- **Confianza alta (70-90%)**: menos detecciones pero más precisas
- **Confianza baja (10-30%)**: detecta más cosas, incluyendo objetos dudosos
- **IoU bajo**: elimina más boxes superpuestos
- **IoU alto**: permite más boxes superpuestos

## Modelo usado

**YOLO11n** (nano) — el más liviano de la familia YOLO11:
- Tamaño: ~6 MB
- Velocidad: muy rápida en CPU
- 80 clases del dataset COCO (personas, vehículos, animales, objetos comunes)

Si quieres más precisión (y tienes buena GPU), cambia en app.py:
```python
model = YOLO("yolo11s.pt")   # small  ~22 MB
model = YOLO("yolo11m.pt")   # medium ~52 MB
model = YOLO("yolo11l.pt")   # large  ~87 MB
```

## Solución de problemas

**"Python no encontrado"**
→ Reinstala Python marcando "Add Python to PATH"

**La instalación falla**
→ Abre CMD como administrador y corre: `pip install ultralytics flask opencv-python`

**El navegador no se abre solo**
→ Abre manualmente: http://localhost:5000

**Lento en la primera detección**
→ Normal, YOLO carga el modelo en memoria. Las siguientes son más rápidas.
