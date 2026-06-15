import time
import cv2
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

app = FastAPI(title="API Deteksi Objek - UAS Image Processing")

print("Loading model YOLOv8...")
model = YOLO("yolov8n.pt")
print("Model siap!")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    start_time = time.time()
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Prediksi dengan YOLO
    results = model(img)
    inference_time = time.time() - start_time

    # Ambil confidence score
    boxes = results[0].boxes
    scores = boxes.conf.tolist() if len(boxes) > 0 else []
    avg_conf = sum(scores) / len(scores) if len(scores) > 0 else 0.0

    # EKSTRAK KETERANGAN OBJEK
    names = model.names # Kamus nama objek bawaan YOLO
    detected_counts = {}
    
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
        
    detail_teks = []
    for name, count in detected_counts.items():
        detail_teks.append(f"{count} {name}")
        
    keterangan_objek = ", ".join(detail_teks) if detail_teks else "Tidak ada objek yang terdeteksi"

    # Ubah gambar ke Base64
    hasil_bgr = results[0].plot()
    _, buffer = cv2.imencode('.jpg', hasil_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "waktu_inferensi_detik": round(inference_time, 4),
        "confidence_score_persen": round(avg_conf * 100, 2),
        "keterangan_objek": keterangan_objek,
        "gambar_hasil_b64": img_base64
    }