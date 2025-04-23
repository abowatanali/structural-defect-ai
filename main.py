from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()

# CORS setup to allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for production, replace with your Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (you can use your custom .pt model here)
model = YOLO("yolov8n.pt")  # or "best.pt" if you have a trained model

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_filename)
    result = results[0]

    detected_classes = [model.names[int(cls)] for cls in result.boxes.cls]
    confidences = [float(conf) for conf in result.boxes.conf]

    # Delete the temporary file
    os.remove(temp_filename)

    # Format response to match frontend expectation
    if detected_classes:
        message = f"Detected: {detected_classes[0]} - Confidence: {confidences[0]*100:.2f}%"
    else:
        message = "No defects detected."

    return { "result": message }