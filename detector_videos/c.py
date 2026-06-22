import cv2
import torch
import torchvision
import time
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def seleccionar_camara():
    camaras_disponibles = []
    print("\nBuscando cámaras disponibles en el dispositivo (esto puede tomar unos segundos)...")
    for i in range(5):  # Revisar los primeros 5 índices
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DSHOW suele ser más estable en Windows
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                camaras_disponibles.append(i)
            cap.release()

    if not camaras_disponibles:
        print("No se detectaron cámaras en el dispositivo.")
        return None

    if len(camaras_disponibles) == 1:
        print(f"-> Se detectó 1 cámara (índice {camaras_disponibles[0]}). Seleccionada por defecto.\n")
        return camaras_disponibles[0]

    print("Cámaras detectadas:")
    for idx in camaras_disponibles:
        print(f"  [{idx}] Cámara {idx}")
    
    while True:
        try:
            seleccion = int(input(f"Ingresa el índice de la cámara que deseas usar {camaras_disponibles}: "))
            if seleccion in camaras_disponibles:
                print(f"Cámara {seleccion} seleccionada.\n")
                return seleccion
            else:
                print("Índice no válido. Intenta de nuevo.")
        except ValueError:
            print("Por favor, ingresa un número válido.")

def main():
    print("Cargando el modelo Faster R-CNN (la primera vez puede tardar en descargar los pesos)...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizando dispositivo para inferencia: {device}")
    model.to(device)
    model.eval() 
    
    # 1. Seleccionar cámara dinámicamente
    video_source = seleccionar_camara()
    if video_source is None:
        return
        
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {video_source}")
        return

    # 2. Configurar y mejorar la ventana
    window_name = 'Deteccion de Objetos - Faster R-CNN'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720) # Ventana grande y redimensionable

    print("Iniciando detección. Presiona 'q' en la ventana de video para salir.")
    umbral_confianza = 0.7 

    while True:
        start_time = time.time() # Iniciar cronómetro de FPS
        
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame. Saliendo...")
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_img = F.to_tensor(rgb_frame).to(device)
        
        with torch.no_grad():
            predictions = model([tensor_img])
            
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= umbral_confianza:
                box = boxes[i].astype(int)
                label_idx = labels[i]
                score = scores[i]
                
                if label_idx < len(COCO_INSTANCE_CATEGORY_NAMES):
                    class_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                else:
                    class_name = f"Clase {label_idx}"
                
                # Dibujar cajón más estilizado (Relleno para el texto)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                texto = f"{class_name}: {score*100:.1f}%"
                
                # Fondo del texto para mayor visibilidad
                (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (box[0], box[1] - h - 10), (box[0] + w, box[1]), (0, 255, 0), -1)
                cv2.putText(frame, texto, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
        # 3. Calcular y mostrar FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time) if end_time - start_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
