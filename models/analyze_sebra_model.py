import os
import glob
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt
import cv2


# =========================
# FIND LATEST RUN AUTOMATICALLY
# =========================
RUNS_DIR = "runs"
run_folders = sorted(glob.glob(os.path.join(RUNS_DIR, "*")))

if len(run_folders) == 0:
    raise FileNotFoundError("No runs found")

LATEST_RUN = run_folders[-1]
MODEL_PATH = os.path.join(LATEST_RUN, "best_model.pth")

print("Using run:", LATEST_RUN)
print("Model:", MODEL_PATH)


# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATA
# =========================
DATA_DIR = os.path.abspath("split_data")

tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=tfms)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)


# =========================
# MODEL
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


# =========================
# GRAD-CAM
# =========================
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.gradients = None
        self.activations = None

        layer.register_forward_hook(self.forward_hook)
        layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, m, i, o):
        self.activations = o

    def backward_hook(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x):
        self.model.zero_grad()

        logits = self.model(x)
        class_idx = torch.argmax(logits, dim=1)

        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


cam_gen = GradCAM(model, model.layer4[-1])


# =========================
# OUTPUT DIR
# =========================
OUT_DIR = os.path.join(LATEST_RUN, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# SAVE FUNCTION
# =========================
def save_cam(img_tensor, cam, path):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225])
    img = img + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    cam = cv2.resize(cam, (224, 224))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = heat[..., ::-1] / 255.0

    overlay = img + 0.6 * heat
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(path, overlay)


# =========================
# EVALUATION LOOP (FIXED)
# =========================
all_preds = []
all_labels = []

for i, (x, y) in enumerate(val_loader):

    x = x.to(device)
    y = y.to(device)

    # --------------------
    # inference (no grad)
    # --------------------
    with torch.no_grad():
        logits = model(x)
        preds = torch.argmax(logits, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(y.cpu().numpy())

    # --------------------
    # Grad-CAM (grad enabled)
    # --------------------
    for j in range(x.size(0)):

        img = x[j:j+1]

        cam = cam_gen.generate(img)

        save_cam(
            img[0],
            cam,
            os.path.join(OUT_DIR, f"gradcam_{i}_{j}.png")
        )


# =========================
# METRICS
# =========================
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds)

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)


with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
    f.write(str(cm))
    f.write("\n\n")
    f.write(report)

print("\nSaved to:", OUT_DIR)