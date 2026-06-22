import os
import uuid
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2

app = Flask(__name__)

# ── Configuración ────────────────────────────────────────────────────────────
UPLOAD_FOLDER  = Path("static/uploads")
RESULTS_FOLDER = Path("static/results")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_SIZE_MB  = 10

# ── Cargar modelo YOLO (se descarga automáticamente la primera vez) ──────────
print("Cargando modelo YOLO11n...")
model = YOLO("yolo11n.pt")   # ~6 MB, se descarga una sola vez
print("Modelo listo.")

# ── Clases en español ────────────────────────────────────────────────────────
LABELS_ES = {
    0:"persona", 1:"bicicleta", 2:"auto", 3:"moto", 4:"avión",
    5:"autobús", 6:"tren", 7:"camión", 8:"barco", 9:"semáforo",
    10:"boca de incendio", 11:"señal de stop", 12:"parquímetro",
    13:"banco/banca", 14:"pájaro", 15:"gato", 16:"perro", 17:"caballo",
    18:"oveja", 19:"vaca", 20:"elefante", 21:"oso", 22:"cebra",
    23:"jirafa", 24:"mochila", 25:"paraguas", 26:"bolso", 27:"corbata",
    28:"maleta", 29:"frisbee", 30:"esquís", 31:"snowboard",
    32:"pelota deportiva", 33:"cometa", 34:"béisbol bat",
    35:"béisbol guante", 36:"skateboard", 37:"tabla de surf",
    38:"raqueta de tenis", 39:"botella", 40:"copa de vino",
    41:"taza", 42:"tenedor", 43:"cuchillo", 44:"cuchara", 45:"tazón",
    46:"banana", 47:"manzana", 48:"sándwich", 49:"naranja",
    50:"brócoli", 51:"zanahoria", 52:"hot dog", 53:"pizza",
    54:"dona", 55:"pastel", 56:"silla", 57:"sofá", 58:"planta",
    59:"cama", 60:"mesa de comedor", 61:"inodoro", 62:"tv/monitor",
    63:"laptop", 64:"mouse", 65:"control remoto", 66:"teclado",
    67:"teléfono celular", 68:"microondas", 69:"horno",
    70:"tostadora", 71:"lavabo", 72:"refrigerador", 73:"libro",
    74:"reloj", 75:"florero", 76:"tijeras", 77:"osito de peluche",
    78:"secador", 79:"cepillo de dientes"
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    # ── Validar archivo ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Formato no soportado: {ext}"}), 400

    # ── Guardar imagen ───────────────────────────────────────────────────────
    uid       = uuid.uuid4().hex
    img_path  = UPLOAD_FOLDER  / f"{uid}{ext}"
    out_path  = RESULTS_FOLDER / f"{uid}_result.jpg"

    file.save(img_path)

    # Verificar tamaño
    if img_path.stat().st_size > MAX_SIZE_MB * 1024 * 1024:
        img_path.unlink()
        return jsonify({"error": f"Imagen demasiado grande (máx {MAX_SIZE_MB} MB)"}), 400

    # ── Parámetros de detección ──────────────────────────────────────────────
    conf_thresh = float(request.form.get("conf", 0.3))
    iou_thresh  = float(request.form.get("iou",  0.45))

    # ── Inferencia YOLO ──────────────────────────────────────────────────────
    results = model.predict(
        source=str(img_path),
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False
    )
    result = results[0]

    # ── Leer imagen con OpenCV para dibujar ──────────────────────────────────
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]

    # Paleta de colores BGR por clase
    PALETTE = [
        (205, 95,  24),  (86, 110, 15),  (23, 117, 185), (147, 53, 153),
        (23, 110,  14),  (30,  60, 185), (23,  95, 179), (180, 95,  23),
    ]

    detections = []
    boxes = result.boxes

    for box in boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        label_en = result.names.get(cls_id, str(cls_id))
        label_es = LABELS_ES.get(cls_id, label_en)
        color    = PALETTE[cls_id % len(PALETTE)]

        # Dibujar bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Etiqueta con fondo
        tag      = f"{label_es} {confidence*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty       = y1 - 6 if y1 > th + 8 else y1 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), color, -1)
        cv2.putText(img, tag, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Datos normalizados para el frontend
        detections.append({
            "label":      label_es,
            "label_en":   label_en,
            "confidence": round(confidence, 3),
            "bbox": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "xn": round(x1/W, 4), "yn": round(y1/H, 4),
                "wn": round((x2-x1)/W, 4), "hn": round((y2-y1)/H, 4),
            }
        })

    # ── Guardar imagen con anotaciones ────────────────────────────────────────
    cv2.imwrite(str(out_path), img)

    # ── Convertir resultado a base64 para enviar al frontend ──────────────────
    with open(out_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Limpiar archivos temporales
    img_path.unlink(missing_ok=True)
    out_path.unlink(missing_ok=True)

    return jsonify({
        "detections": detections,
        "image_b64":  img_b64,
        "total":      len(detections),
        "img_size":   {"w": W, "h": H}
    })


if __name__ == "__main__":
    print("\n========================================")
    print("  Detector YOLO11n corriendo en local")
    print("  Abre: http://localhost:5000")
    print("========================================\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
