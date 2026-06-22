# Actividad 6 - Detección de Objetos en Tiempo Real

Este proyecto contiene dos implementaciones de detección de objetos en tiempo real utilizando la cámara del dispositivo mediante Python y OpenCV. Se abordan dos aproximaciones clave en la Visión Artificial: **Faster R-CNN** y **YOLOv8**.

## Archivos Principales

### 1. `c.py` - Implementación con Faster R-CNN (PyTorch / Torchvision)
Esta solución utiliza la arquitectura **Faster R-CNN con un backbone ResNet-50 FPN**, provista por PyTorch. Es un detector de "dos etapas" (two-stage detector), ofreciendo por lo general una alta precisión a expensas de un mayor costo computacional (menos fotogramas por segundo, especialmente si se corre sin una tarjeta gráfica GPU).

**Partes importantes del código:**
*   **Carga del modelo (`fasterrcnn_resnet50_fpn`)**: Se instancia la arquitectura cargando los pesos por defecto (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`), entrenados previamente con el dataset COCO (capaz de detectar 91 tipos de objetos).
*   **Preprocesamiento (`F.to_tensor`)**: OpenCV captura los fotogramas (frames) en formato de color BGR. Para que el modelo lo procese correctamente, se convierte a RGB (`cv2.cvtColor`) y luego se convierte a un tensor de PyTorch (que adicionalmente normaliza los valores de los píxeles a un rango de 0 a 1).
*   **Inferencia Manual**: Se inyecta la imagen al modelo (`model([tensor_img])`) bajo el entorno `torch.no_grad()` para ahorrar memoria RAM y VRAM.
*   **Post-procesamiento y Filtro**: El código itera manualmente sobre el diccionario de predicciones filtrando solo aquellos objetos cuyo nivel de confianza (`scores`) supere el 70% (`umbral_confianza = 0.7`).
*   **Anotación con OpenCV**: Las cajas y textos de las clases detectadas se renderizan geométricamente utilizando las funciones matemáticas y de dibujo de OpenCV (`cv2.rectangle` y `cv2.putText`).

---

### 2. `d.py` - Implementación con YOLOv8 (Ultralytics)
Esta solución utiliza la famosa arquitectura **YOLO (You Only Look Once)**, usando su iteración de estado del arte **YOLOv8n (nano)** mediante la librería oficial `ultralytics`. Es un detector "de una etapa" optimizado para ofrecer extrema velocidad computacional, logrando una tasa de FPS sumamente alta, ideal para escenarios de tiempo real.

**Partes importantes del código:**
*   **Alta Abstracción (`YOLO('yolov8n.pt')`)**: Ultralytics empaqueta la complejidad. Bastan dos líneas de código para cargar el modelo e inicializar sus pesos preentrenados (se descargan automáticamente si no existen).
*   **Inferencia en Flujo (`stream=True`)**: Se realiza llamando al objeto instanciado: `model(frame, stream=True)`. Este parámetro le indica al modelo que use generadores para procesar un flujo infinito de imágenes continuas, liberando basura de la memoria constantemente.
*   **Anotación Automática (`r.plot()`)**: La librería posee su propio motor de dibujado. Al ejecutar `plot()` sobre el resultado, la librería devuelve el frame original de la cámara ya modificado y pintado con los "bounding boxes", las clases y las estadísticas. Evita tener que usar complejas funciones matemáticas de OpenCV de forma manual.

---

## Características Comunes (En Ambos Scripts)
Para enriquecer la aplicación y mejorar la experiencia de uso, ambos archivos comparten las siguientes rutinas de código:

1.  **Selección Dinámica de Cámara (`seleccionar_camara()`)**: Se construyó un bloque de código que escanea las ranuras locales del sistema mediante el framework de Windows `cv2.CAP_DSHOW`. Si el dispositivo tiene más de una cámara conectada (por ejemplo, integrada vs externa), mostrará una lista interactiva en la consola para elegir cuál usar.
2.  **Medidor de FPS (Fotogramas Por Segundo)**: Usando la librería `time`, se toma una "foto" del reloj del sistema antes de procesar un frame (`start_time`), y otra al finalizar de dibujarlo (`end_time`). El inverso del tiempo transcurrido equivale a la velocidad en tiempo real: `fps = 1 / (end_time - start_time)`.
3.  **Ventana Ergonómica Mejorada**: Para evitar las ventanas pequeñas de OpenCV que no se pueden estirar, se usa `cv2.namedWindow(..., cv2.WINDOW_NORMAL)` y `cv2.resizeWindow(...)`, creando una interfaz de video de alta resolución (`1280x720`) totalmente redimensionable por el usuario.

## Requisitos de Instalación
Para poder ejecutar el código, debes abrir tu consola/terminal y asegurarte de tener instaladas las siguientes dependencias:

```bash
pip install torch torchvision ultralytics opencv-python
```

## ¿Cómo ejecutar la aplicación?
1. Abre tu terminal.
2. Navega hasta el directorio donde guardaste los archivos.
3. Ejecuta el archivo de tu elección:

```bash
python c.py  # Para correr la aproximación con Faster R-CNN
# O bien:
python d.py  # Para correr la aproximación súper veloz de YOLO
```
4. Observa tu terminal en caso de que te pida seleccionar un número de cámara.
5. Para terminar la ejecución y cerrar la ventana, **selecciona la ventana de video y presiona la tecla `q`**.
