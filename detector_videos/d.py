import cv2
import time
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

def seleccionar_camara():
    camaras_disponibles = []
    print("\nBuscando cámaras disponibles en el dispositivo (esto puede tomar unos segundos)...")
    # Buscar en los primeros índices comunes
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # CAP_DSHOW es más estable en Windows para inicialización rápida
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
    print("Cargando el modelo YOLOv8n...")
    model = YOLO('yolov8n.pt') 
    
    # 1. Llamar a la función de selección de cámara
    video_source = seleccionar_camara()
    if video_source is None:
        return
        
    cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara {video_source}")
        return

    # 2. Mejorar la ventana (redimensionable y con mejor tamaño inicial)
    window_name = 'Deteccion de Objetos en Tiempo Real - YOLOv8'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("Iniciando detección con YOLO. Presiona 'q' en la ventana para salir.")
    
    while True:
        start_time = time.time() # Iniciar cronómetro para medir los FPS
        
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame. Saliendo...")
            break
            
        results = model(frame, stream=True, verbose=False)
        
        for r in results:
            annotated_frame = r.plot()
            
            # 3. Calcular y mostrar los FPS en la pantalla
            end_time = time.time()
            fps = 1 / (end_time - start_time) if end_time - start_time > 0 else 0
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            cv2.imshow(window_name, annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
