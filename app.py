from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf
import numpy as np
import cv2
import io
import gdown
from PIL import Image

app = FastAPI()

# =========================
# 🔥 CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔥 Model Config
# =========================
FILE_ID    = "1U_yALQDMHepzu-B3J7Wq5DSbLzZVrBy5"
MODEL_PATH = "dfu_model_final.h5"

CLASS_NAMES = ['Healthy', 'Grade1', 'Grade2', 'Grade3', 'Grade4']
IMG_SIZE    = 224
THRESHOLD   = 0.60

# =========================
# 🔥 Load Model
# =========================
def load_model():
    if not os.path.isfile(MODEL_PATH):
        print("⬇ Downloading model...")
        gdown.download(
            url=f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t",
            output=MODEL_PATH,
            quiet=False
        )
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"Downloaded: {size_mb:.1f} MB")
        if size_mb < 1:
            os.remove(MODEL_PATH)
            raise RuntimeError("فشل التحميل — تأكد إن Drive Public")
        print("✅ Model downloaded")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded")

    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    model(dummy)

    return model

model = load_model()

# =========================
# 🔥 Skin Detection
# =========================
def is_foot_image(img_array, skin_threshold=0.25):
    img_ycrcb  = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    lower_ycc  = np.array([0,   133, 77],  dtype=np.uint8)
    upper_ycc  = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycc, upper_ycc)

    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0,  15,  50],  dtype=np.uint8)
    upper_hsv = np.array([25, 200, 255], dtype=np.uint8)
    mask_hsv  = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    combined   = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    skin_ratio = np.sum(combined > 0) / (img_array.shape[0] * img_array.shape[1])

    return skin_ratio >= skin_threshold, float(skin_ratio)

# =========================
# 🔥 Preprocess
# =========================
def preprocess_image(img_array):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# 🔥 Routes
# =========================

@app.get("/")
def root():
    return {"status": "API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents  = await file.read()
        if len(contents) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large")
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # === Skin Detection ===
    is_skin, skin_ratio = is_foot_image(img_array)
    if not is_skin:
        return {
            "status"    : "rejected",
            "prediction": "Not a foot",
            "confidence": 0.0,
            "skin_ratio": round(skin_ratio, 3),
            "message"   : "الصورة ليست قدم"
        }

    # === Model ===
    img_processed = preprocess_image(img_array)
    preds         = model(img_processed, training=False).numpy()[0]
    confidence    = float(np.max(preds))
    pred_index    = int(np.argmax(preds))
    pred_class    = CLASS_NAMES[pred_index]

    # === Uncertainty Check ===
    top2 = np.sort(preds)[-2:]
    if (top2[1] - top2[0]) < 0.10:
        return {
            "status"    : "uncertain",
            "prediction": "Undefined",
            "confidence": round(confidence, 3),
            "skin_ratio": round(skin_ratio, 3),
            "message"   : "الموديل غير متأكد"
        }

    if confidence < THRESHOLD:
        return {
            "status"    : "uncertain",
            "prediction": "Undefined",
            "confidence": round(confidence, 3),
            "skin_ratio": round(skin_ratio, 3),
            "message"   : "الثقة منخفضة"
        }

    return {
        "status"    : "ok",
        "prediction": pred_class,
        "confidence": round(confidence, 3),
        "skin_ratio": round(skin_ratio, 3),
    }
