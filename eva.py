import torch
from torchvision import transforms
from torch.nn import functional as F
from transformers import ViTModel
from PIL import Image
import requests

# DINO Transforms
T = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Get images from Figure 11
urls = [
    'https://github.com/google/dreambooth/blob/main/dataset/rc_car/03.jpg?raw=true', # reference from Fig 11
    'https://github.com/google/dreambooth/blob/main/dataset/rc_car/02.jpg?raw=true'# Real Sample from Fig 11
]
images = [
    T(Image.open(requests.get(url, stream=True).raw))
    for url in urls
]
inputs = torch.stack(images) # (2, 3, 224, 224). Batchsize = 2

# Load DINO ViT-S/16
model = ViTModel.from_pretrained('facebook/dino-vits16')

# Get DINO features
with torch.no_grad():
    outputs = model(inputs)

last_hidden_states = outputs.last_hidden_state # ViT backbone features
emb_img1, emb_img2 = last_hidden_states[0, 0], last_hidden_states[1, 0] # Get cls token (0-th token) for each img
metric = F.cosine_similarity(emb_img1, emb_img2, dim=0)
print(f'''
DINO Score
Expected: 0.770
Calculated: {metric.item():.3f}''')