import os
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# -------------------------------
# CONFIG
# -------------------------------
IMAGE_FOLDER = "processed_images"
EMBEDDINGS_FILE = "embeddings.pkl"
FILENAMES_FILE = "filenames.pkl"
BATCH_SIZE = 64   # adjust based on GPU (32 if low VRAM)

# -------------------------------
# DEVICE
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# MODEL (ResNet50 backbone)
# -------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# remove classification head
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()
model.to(device)

# -------------------------------
# TRANSFORM
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# LOAD IMAGE PATHS
# -------------------------------
filenames = []
for file in os.listdir(IMAGE_FOLDER):
    path = os.path.join(IMAGE_FOLDER, file)
    filenames.append(path)

print(f"Total images found: {len(filenames)}")

# -------------------------------
# FEATURE EXTRACTION (BATCHED)
# -------------------------------
features = []

with torch.no_grad():
    for i in tqdm(range(0, len(filenames), BATCH_SIZE), desc="Extracting features"):
        batch_paths = filenames[i:i+BATCH_SIZE]
        batch_images = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = transform(img)
                batch_images.append(img)
            except:
                continue

        if len(batch_images) == 0:
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        outputs = model(batch_tensor)  # shape: (B, 2048, 1, 1)
        outputs = outputs.view(outputs.size(0), -1)  # flatten

        outputs = outputs.cpu().numpy()

        # normalize (important for similarity search)
        norms = np.linalg.norm(outputs, axis=1, keepdims=True)
        outputs = outputs / norms

        features.extend(outputs)

features = np.array(features)

# -------------------------------
# SAVE
# -------------------------------
pickle.dump(features, open(EMBEDDINGS_FILE, "wb"))
pickle.dump(filenames, open(FILENAMES_FILE, "wb"))

print("\n✅ Done!")
print(f"Feature shape: {features.shape}")
print("Saved:", EMBEDDINGS_FILE, FILENAMES_FILE)