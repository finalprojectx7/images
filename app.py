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
# 🔥 رابط الموديل الجديد
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1wgQSkAcDPGTmNtlMx_BF3Js7x05MLD9u"
MODEL_PATH = "model.pth"

# =========================
# 🔥 تحميل الموديل
# =========================
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)
    print("Model downloaded ✅")

# =========================
# 🔥 إنشاء الموديل (EfficientNet B3)
# =========================
model = models.efficientnet_b3(weights=None)

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 5)
)

# =========================
# 🔥 تحميل الـ checkpoint
# =========================
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 🔥 الكلاسات
class_names = checkpoint["classes"]

# =========================
# 🔥 transforms (نفس التدريب)
# =========================
transform = transforms.Compose([
    transforms.Resize((300, 300)),
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
    # 🔥 Undefined System (احترافي)
    # =========================

    THRESHOLD = 0.65   # متوازن
    MARGIN = 0.20      # فرق واضح

    # ❌ لو مش واثق
    if confidence_value < THRESHOLD:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value,
            "reason": "Low confidence"
        }

    # ❌ لو محتار بين كلاسين
    if (top1 - top2) < MARGIN:
        return {
            "prediction": "Undefined",
            "confidence": confidence_value,
            "reason": "Model confused between classes"
        }

    # =========================
    # ✅ النتيجة
    # =========================
    return {
        "prediction": class_names[pred.item()],
        "confidence": confidence_value
    }
