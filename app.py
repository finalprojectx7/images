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

        # 🔥 أعلى احتمالين
        sorted_probs = torch.sort(probs, descending=True)
        top1 = sorted_probs.values[0][0].item()
        top2 = sorted_probs.values[0][1].item()

    # =========================
    # 🔥 Logic ذكي (مناسب للموديل)
    # =========================

    THRESHOLD = 0.45   # أقل شوية علشان موديلك ضعيف نسبيًا
    MARGIN = 0.20      # لازم يكون الفرق واضح

    # ❌ لو confidence ضعيف
    if confidence_value < THRESHOLD:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value
        }

    # ❌ لو الموديل محتار
    if (top1 - top2) < MARGIN:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value
        }

    # ❌ لو مش Normal وconfidence مش عالي
    if pred.item() != class_names.index("Normal") and confidence_value < 0.6:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value
        }

    # =========================
    # ✅ النتيجة النهائية
    # =========================
    return {
        "prediction": class_names[pred.item()],
        "confidence": confidence_value
    }
