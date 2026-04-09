from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn.functional as F
import os
import gdown

app = FastAPI()

# =========================
# 🔥 رابط الموديل
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1U10r0RQZF1Sj7lTmHVW4A3zMeYn0QKAu"
MODEL_PATH = "model.pth"

# =========================
# 🔥 تحميل الموديل
# =========================
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    print("Model downloaded ✅")

# =========================
# 🔥 إنشاء الموديل
# =========================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

# =========================
# 🔥 تحميل الـ checkpoint
# =========================
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 🔥 الكلاسات
class_names = checkpoint["class_names"]

# =========================
# 🔥 transforms
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# 🔥 API endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)
        confidence_value = float(confidence.item())

        sorted_probs = torch.sort(probs, descending=True)
        top1 = sorted_probs.values[0][0].item()
        top2 = sorted_probs.values[0][1].item()

    # =========================
    # 🔥 Logic متوازن
    # =========================

    THRESHOLD = 0.3   # مهم جدًا ↓ (علشان موديلك ضعيف)
    MARGIN = 0.08     # فرق صغير

    # ❌ لو ضعيف جدًا
    if confidence_value < THRESHOLD:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value
        }

    # ❌ لو محتار جدًا
    if (top1 - top2) < MARGIN:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value
        }

    # =========================
    # ✅ النتيجة
    # =========================
    return {
        "prediction": class_names[pred.item()],
        "confidence": confidence_value
    }
