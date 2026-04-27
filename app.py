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
# Model Link
# =========================
MODEL_URL = "https://drive.google.com/uc?id=1U10r0RQZF1Sj7lTmHVW4A3zMeYn0QKAu"
MODEL_PATH = "model.pth"

# =========================
# Download model
# =========================
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(
        url=MODEL_URL,
        output=MODEL_PATH,
        quiet=False
    )
    print("Model downloaded ✅")


# =========================
# Load model
# =========================
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

class_names = checkpoint["class_names"]


# =========================
# Image preprocessing
# =========================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


# =========================
# Health check
# =========================
@app.get("/")
def home():
    return {
        "status":"API Running"
    }


# =========================
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(
        io.BytesIO(await file.read())
    ).convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = F.softmax(
            output,
            dim=1
        )

        confidence, pred = torch.max(
            probs,
            1
        )

        confidence_value = float(
            confidence.item()
        )

        # أعلى احتمالين
        top2 = torch.topk(
            probs,
            2
        ).values[0]

    # =========================
    # Undefined only if confused
    # =========================
    MARGIN = 0.10

    if (top2[0]-top2[1]).item() < MARGIN:
        return {
            "prediction":"Undefined",
            "confidence":confidence_value,
            "reason":"Model confused"
        }

    # =========================
    # Final prediction
    # =========================
    return {
        "prediction":class_names[pred.item()],
        "confidence":confidence_value
    }
