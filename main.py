from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
import io

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = [
    "pizza",
    "hamburger",
    "sushi",
    "ice_cream",
    "fried_rice",
    "ramen",
    "steak",
    "chicken_curry",
    "spaghetti_bolognese",
    "caesar_salad"
]

model = models.resnet18(pretrained=None)
model.fc = nn.Sequential(
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512,10)
)
model.load_state_dict(
    torch.load('best_model101.pth',map_location=device)
)
model.to(device)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
model.eval()
def pred_label(image):
  image_tensor = test_transforms(image).to(device)
  with torch.inference_mode():
    pred = torch.softmax(model(image_tensor.unsqueeze(dim=0)),dim=1)
  return {'name':classes[pred.argmax(dim=1).item()],'confidence':f'{(pred.max().item()*100):.2f}%'}



@app.get("/")
def home():
    return {"message": "Food101 API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return pred_label(image)
