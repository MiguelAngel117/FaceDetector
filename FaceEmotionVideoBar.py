from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variables y configuraciones iniciales
x = [0, 1, 2, 3, 4, 5, 6]
y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
my_colors = 'rgbykmc'
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Cargar modelos
prototxtPath = r"./face_detector/deploy.prototxt"
weightsPath = r"./face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
emotionModel = load_model("./Reconocimiento.h5")

# Función para predecir emociones
def predict_emotion(frame, faceNet, emotionModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

            pred = emotionModel.predict(face)
            preds.append(pred[0])

    return (locs, preds)

# Inicializar la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("Detección de Emociones en Tiempo Real")
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=1)

camera_label = tk.Label(root)
camera_label.grid(row=0, column=0, sticky="nsew")

# Configuración de la gráfica
figura1 = plt.figure(figsize=(4, 2.5))
bar1 = plt.bar(x, y, color=my_colors, tick_label=classes)
plt.xticks(rotation=45)
canvas = FigureCanvasTkAgg(figura1, master=root)
graph_widget = canvas.get_tk_widget()
graph_widget.grid(row=0, column=1, sticky="nsew")

# Función para actualizar el frame
def update_frame():
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=640)
    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        label = "{}: {:.2f}%".format(classes[np.argmax(pred)], max(pred) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # Actualización de la gráfica
        plt.cla()
        bars = plt.bar(x, pred, color=my_colors, tick_label=classes)
        plt.ylim([0, 1])
        plt.xticks(rotation=45)
        canvas.draw()

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)
    root.after(10, update_frame)

# Función para detener el programa
def stop_program():
    cam.release()
    root.quit()

stop_button = tk.Button(root, text="Detener", command=stop_program, 
                        bg="light blue", fg="white", 
                        font=("Helvetica", 12, "bold"))
stop_button.grid(row=1, column=1, sticky="sw")
# Inicia la actualización y la interfaz gráfica
update_frame()
root.mainloop()