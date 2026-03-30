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
MODEL_URL = "https://drive.google.com/uc?id=10zAXshLCaBI0up4NG1z_zbiHCmO3LCYg"
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
# 🔥 تحميل الـ checkpoint صح
# =========================
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

# 🔥 دي أهم نقطة (حل المشكلة بتاعتك)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

# 🔥 الكلاسات من الموديل نفسه
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

    return {
        "prediction": class_names[pred.item()],
        "confidence": float(confidence.item())
    }
