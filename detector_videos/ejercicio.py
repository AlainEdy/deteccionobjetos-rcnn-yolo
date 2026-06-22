import cv2
import tkinter as tk
from tkinter import filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def calcular_iou(boxGT, boxPred):
    """
    Calcula la Intersección sobre la Unión (IoU) dadas dos cajas.
    Las cajas vienen en formato (x, y, w, h)
    """
    # Convertir de (x, y, ancho, alto) a (x1, y1, x2, y2)
    x1_gt, y1_gt, x2_gt, y2_gt = boxGT[0], boxGT[1], boxGT[0]+boxGT[2], boxGT[1]+boxGT[3]
    x1_pr, y1_pr, x2_pr, y2_pr = boxPred[0], boxPred[1], boxPred[0]+boxPred[2], boxPred[1]+boxPred[3]

    # Encontrar las coordenadas del rectángulo de intersección
    xA = max(x1_gt, x1_pr)
    yA = max(y1_gt, y1_pr)
    xB = min(x2_gt, x2_pr)
    yB = min(y2_gt, y2_pr)

    # Calcular el área de intersección (si no se superponen, el área es 0)
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calcular el área de ambas cajas por separado
    areaGT = (x2_gt - x1_gt) * (y2_gt - y1_gt)
    areaPred = (x2_pr - x1_pr) * (y2_pr - y1_pr)

    # Calcular el área de unión (Área A + Área B - Intersección)
    unionArea = areaGT + areaPred - interArea

    # Calcular el IoU
    iou = interArea / float(unionArea) if unionArea > 0 else 0.0

    return iou, interArea, unionArea, areaGT, areaPred, (xA, yA, xB, yB)

def main():
    print("Iniciando aplicación interactiva para cálculo de IoU y mAP...")
    
    # 1. Crear ventana raíz de Tkinter (oculta por ahora)
    root = tk.Tk()
    root.withdraw() 
    
    # Seleccionar la imagen
    file_path = filedialog.askopenfilename(
        title="Selecciona una imagen para el cálculo de IoU",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    if not file_path:
        print("Operación cancelada. No se seleccionó ninguna imagen.")
        root.destroy()
        return

    # 2. Cargar imagen con OpenCV
    img = cv2.imread(file_path)
    if img is None:
        print("Error: No se pudo cargar la imagen seleccionada.")
        root.destroy()
        return

    # Escalar la imagen si es muy grande para que quepa en la pantalla al dibujar
    h, w = img.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    img_disp = img.copy()

    # 3. Dibujar Bounding Box Real (Ground Truth)
    print("\nPASO 1: Dibuja el 'Ground Truth'")
    print("-> Haz clic, arrastra para enmarcar un objeto y presiona la tecla ESPACIO o ENTER para confirmar.")
    gt_box = cv2.selectROI("1. Dibuja Ground Truth (Caja Real - Ej: Verde)", img_disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("1. Dibuja Ground Truth (Caja Real - Ej: Verde)")

    # Dibujarlo permanentemente en la imagen temporal para tener referencia al dibujar el segundo
    cv2.rectangle(img_disp, (int(gt_box[0]), int(gt_box[1])), 
                  (int(gt_box[0]+gt_box[2]), int(gt_box[1]+gt_box[3])), (0, 255, 0), 2)

    # 4. Dibujar Bounding Box de la Predicción
    print("\nPASO 2: Dibuja la 'Predicción'")
    print("-> Imagina que eres el modelo de IA. Dibuja otra caja sobre el mismo objeto (puedes fallar un poco a propósito). Presiona ESPACIO al terminar.")
    pred_box = cv2.selectROI("2. Dibuja Prediccion (Caja del Modelo - Ej: Rojo)", img_disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("2. Dibuja Prediccion (Caja del Modelo - Ej: Rojo)")

    # Validar que se dibujaron áreas válidas
    if gt_box[2] == 0 or gt_box[3] == 0 or pred_box[2] == 0 or pred_box[3] == 0:
         print("Error: Una de las cajas se dibujó sin área (0x0). Abortando.")
         root.destroy()
         return

    # 5. Realizar Cálculos
    iou, interArea, unionArea, areaGT, areaPred, interRect = calcular_iou(gt_box, pred_box)

    # Convertir BGR (OpenCV) a RGB (Matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Crear una capa superpuesta para pintar el área de intersección en amarillo
    overlay = img_rgb.copy()
    if interArea > 0:
        xA, yA, xB, yB = [int(v) for v in interRect]
        cv2.rectangle(overlay, (xA, yA), (xB, yB), (255, 255, 0), -1) # Relleno amarillo
    
    # Dibujar contornos
    cv2.rectangle(overlay, (int(gt_box[0]), int(gt_box[1])), 
                  (int(gt_box[0]+gt_box[2]), int(gt_box[1]+gt_box[3])), (0, 255, 0), 3)
    cv2.rectangle(overlay, (int(pred_box[0]), int(pred_box[1])), 
                  (int(pred_box[0]+pred_box[2]), int(pred_box[1]+pred_box[3])), (255, 0, 0), 3)

    # Fusionar con transparencia
    alpha = 0.5
    img_final = cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0)

    # ==========================================
    # 6. Crear la UI Final con Matplotlib y Tkinter (Scrollbar)
    # ==========================================
    root.deiconify() # Volver a mostrar la ventana principal
    root.title('Cálculo e Interpretación de IoU y mAP')
    root.geometry("1300x700")
    root.configure(bg="#f0f0f0")

    # Frame izquierdo para el gráfico (Se expande)
    frame_izq = tk.Frame(root, bg="#f0f0f0")
    frame_izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Frame derecho para el texto (Ancho fijo)
    frame_der = tk.Frame(root, bg="#f0f0f0", width=450)
    frame_der.pack(side=tk.RIGHT, fill=tk.Y, padx=20, pady=20)
    frame_der.pack_propagate(False) # Evitar que cambie de tamaño por el contenido

    # --- Configurar Matplotlib en el panel izquierdo ---
    fig = plt.figure(figsize=(8, 8))
    ax_img = plt.subplot(1, 1, 1)
    ax_img.imshow(img_final)
    ax_img.set_title(f"IoU Calculado: {iou*100:.2f}%", fontsize=16, fontweight='bold', color='darkblue')
    ax_img.axis('off')
    
    # Leyenda de colores
    legend_elements = [
        Patch(facecolor='none', edgecolor=(0,1,0), linewidth=3, label='Ground Truth (Real)'),
        Patch(facecolor='none', edgecolor=(1,0,0), linewidth=3, label='Predicción (Modelo IA)'),
        Patch(facecolor=(1,1,0,0.5), label='Área de Intersección')
    ]
    ax_img.legend(handles=legend_elements, loc='lower right', fontsize=11)
    plt.tight_layout()

    # Integrar el Canvas de Matplotlib en Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame_izq)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Configurar el panel de texto con Scroll ---
    # Quité los emojis que causaban advertencias en tu consola y rediseñé el panel
    texto_explicativo = (
        "1. CÁLCULO DE IoU (Intersection over Union)\n"
        "-----------------------------------------------------------\n"
        f"-> Área del Ground Truth (Caja Verde): {areaGT} píxeles²\n"
        f"-> Área de la Predicción (Caja Roja):  {areaPred} píxeles²\n"
        f"-> Área de Intersección (Amarillo):    {interArea} píxeles²\n\n"
        
        "Fórmula Matemática:\n"
        "  IoU = (Área de Intersección) / (Área de Unión)\n"
        f"  Área de Unión = {areaGT} + {areaPred} - {interArea} = {unionArea} px²\n\n"
        
        f"  IoU = {interArea} / {unionArea} = {iou:.4f}\n\n"
        
        "INTERPRETACIÓN DEL IoU:\n"
        "Sirve para saber qué tan acertada fue la predicción del modelo.\n"
        "Dependiendo del umbral que elijas (usualmente 0.5 o 50%):\n"
        f"  Evaluación en tu ejemplo: {'¡VERDADERO POSITIVO! (Acierto)' if iou >= 0.5 else 'FALSO POSITIVO (Error)'}\n"
        "  (Porque el IoU calculado es " + ("mayor" if iou >= 0.5 else "menor") + " a 0.5)\n\n\n"
        
        "2. ¿CÓMO SE RELACIONA ESTO CON EL mAP?\n"
        "-----------------------------------------------------------\n"
        "El mAP (mean Average Precision) es la métrica reina para evaluar modelos\n"
        "como YOLO o Faster R-CNN en un conjunto de datos completo.\n\n"
        
        "Pasos de cómo se utiliza el IoU para obtener el mAP:\n"
        "1. Para cada imagen, el modelo calcula múltiples IoUs usando este \n"
        "   mismo método matemático.\n"
        "2. Según el IoU, se cuentan cuántos fueron Verdaderos Positivos y\n"
        "   cuántos fueron Falsos Positivos.\n"
        "3. Con esos números se calcula la Precisión (Calidad del modelo) y\n"
        "   el Recall (Cantidad que logró encontrar de la realidad).\n"
        "4. Se grafica la curva Precisión vs Recall y se calcula su 'Área\n"
        "   Bajo la Curva'. Eso se llama AP (Average Precision) de una clase.\n"
        "5. ¡El mAP es simplemente el promedio del AP de todas las clases!\n\n"
        "Nota: Verás métricas como mAP@0.5, esto significa que el modelo\n"
        "exigió que el IoU fuera mayor a 0.5 para considerarlo correcto.\n\n"
        "Conceptos Adicionales:\n"
        "- Recall: ¿Qué porcentaje de los objetos reales encontró la IA?\n"
        "- Precision: ¿Qué porcentaje de lo que la IA encontró era correcto?\n"
    )

    lbl_titulo = tk.Label(frame_der, text="Análisis y Cálculos", font=("Arial", 16, "bold"), bg="#f0f0f0")
    lbl_titulo.pack(pady=(0, 10))

    # Widget de texto nativo con scroll bar (ScrolledText)
    txt_scroll = scrolledtext.ScrolledText(frame_der, wrap=tk.WORD, font=("Consolas", 12), bg="#ffffff", fg="#333333", padx=10, pady=10)
    txt_scroll.insert(tk.INSERT, texto_explicativo)
    txt_scroll.config(state=tk.DISABLED) # Evita que el usuario edite el texto
    txt_scroll.pack(fill=tk.BOTH, expand=True)

    # Asegurar que la ejecución de Python termine correctamente al cerrar la ventana
    def on_closing():
        root.quit()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main()
