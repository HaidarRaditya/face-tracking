import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

try:
    font_ubuntu = ImageFont.truetype("Ubuntu-Regular.ttf", 20)
except IOError:
    print("Peringatan: File 'Ubuntu-Regular.ttf' tidak ditemukan di folder! Menggunakan font default.")
    font_ubuntu = ImageFont.load_default()

if not cap.isOpened():
    print("Error: Tidak dapat mengakses webcam.")
    exit()

print("Webcam aktif! Tekan tombol 'q' pada keyboard untuk keluar.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Gagal menerima frame dari webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    wajah = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in wajah:
        padding = 30  
        x_new = max(0, x - padding)
        y_new = max(0, y - padding)
        w_new = w + (padding * 2)
        h_new = h + (padding * 2)

        cv2.rectangle(
            frame,
            (x_new, y_new),
            (x_new + w_new, y_new + h_new),
            (0, 255, 0),    
            2               
        )

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for (x, y, w, h) in wajah:
        x_new = max(0, x - 30)
        y_new = max(0, y - 30)
        
        text_position = (x_new, max(0, y_new - 25))
        
        draw.text(text_position, "admin", font=font_ubuntu, fill=(0, 255, 0))

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("Deteksi Wajah Real-Time (Font Ubuntu)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
