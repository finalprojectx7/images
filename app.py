from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
import os
import gdown
from PIL import Image

app = FastAPI()

# CORS عشان Flutter يقدر يكلم الـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔥 رابط الموديل على Drive
# =========================
# السطر 1
MODEL_URL  = "https://drive.google.com/file/d/1TMUub0CxlvumUkEuPYgA27H1N47KN-YZ/view?usp=sharing"

# السطر 2  
MODEL_PATH = "dfu_model_final.h5"

# =========================
# 🔥 تحميل الموديل
# =========================
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded ✅")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded ✅")

# =========================
# 🔥 الإعدادات
# =========================
CLASS_NAMES = ['Healthy', 'Grade1', 'Grade2', 'Grade3', 'Grade4']
IMG_SIZE    = 224
THRESHOLD   = 0.60

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
# 🔥 Preprocessing
# =========================
def preprocess_image(img_array):
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# 🔥 API Endpoints
# =========================

@app.get("/")
def root():
    return {"message": "DFU API is running ✅"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents  = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(pil_image)

    # === خطوة 1: Skin Detection ===
    is_skin, skin_ratio = is_foot_image(img_array)

    if not is_skin:
        return {
            "status"    : "rejected",
            "prediction": "Not a foot image",
            "message"   : "الصورة مش صورة قدم — صوّر القدم بشكل واضح",
            "confidence": 0.0,
            "skin_ratio": round(skin_ratio, 3)
        }

    # === خطوة 2: موديل DFU ===
    img_processed = preprocess_image(img_array)
    preds         = model.predict(img_processed, verbose=0)[0]
    confidence    = float(np.max(preds))
    pred_class    = CLASS_NAMES[int(np.argmax(preds))]

    # === خطوة 3: Confidence Threshold ===
    if confidence < THRESHOLD:
        return {
            "status"    : "uncertain",
            "prediction": "Undefined",
            "message"   : "الصورة مش واضحة — صوّر تاني في إضاءة أحسن",
            "confidence": round(confidence, 3),
            "skin_ratio": round(skin_ratio, 3)
        }

    return {
        "status"    : "ok",
        "prediction": pred_class,
        "message"   : "تم التشخيص بنجاح",
        "confidence": round(confidence, 3),
        "skin_ratio": round(skin_ratio, 3),
        "all_probs" : {
            CLASS_NAMES[i]: round(float(preds[i]), 3)
            for i in range(len(CLASS_NAMES))
        }
    }
