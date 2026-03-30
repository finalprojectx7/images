from fastapi import FastAPI, File, UploadFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn.functional as F
import requests
import os

app = FastAPI()

# 🔥 رابط الموديل
MODEL_URL = "https://drive.google.com/uc?id=10zAXshLCaBI0up4NG1z_zbiHCmO3LCYg"

MODEL_PATH = "model.pth"

# 🔥 تحميل الموديل لو مش موجود
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded ✅")

# 🔥 تحميل الموديل
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 5)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

class_names = ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4', 'Normal']

# 🔥 transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# 🔥 API endpoint
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
