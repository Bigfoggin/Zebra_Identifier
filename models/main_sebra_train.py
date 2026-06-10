import os
import random
import json
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2


# =========================
# RUN DIRECTORY (NO OVERWRITE)
# =========================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = os.path.join("runs", timestamp)

os.makedirs(RUN_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(RUN_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(RUN_DIR, "best_model.pth")
METRICS_PATH = os.path.join(RUN_DIR, "metrics.json")


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATA
# =========================
DATA_DIR = os.path.abspath("split_data")

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tfms)

print("Class mapping:", train_ds.class_to_idx)

pos_class = train_ds.class_to_idx["y"]  # adjust if needed


train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=min(8, os.cpu_count() or 4),
                          pin_memory=True)

val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=min(8, os.cpu_count() or 4),
                        pin_memory=True)


# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


# =========================
# LOSS (IMBALANCE AWARE)
# =========================
class_weights = torch.tensor([1.0, 3.54], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)


# =========================
# OPTIMIZER
# =========================
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# =========================
# METRICS
# =========================
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# =========================
# EVAL
# =========================
def evaluate(model, loader):
    model.eval()

    preds_all, labels_all = [], []
    loss_sum, total = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

    metrics = compute_metrics(labels_all, preds_all)
    metrics["loss"] = loss_sum / total
    return metrics


# =========================
# TRAIN
# =========================
EPOCHS = 10
best_score = -1
history = []

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    val_metrics = evaluate(model, val_loader)

    # prioritize recall (your requirement)
    score = 0.6 * val_metrics["f1"] + 0.4 * val_metrics["recall"]

    print("\n--- Epoch Summary ---")
    print(val_metrics)
    print("Score:", score)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss / len(train_loader),
        **val_metrics,
        "score": score
    })

    if score > best_score:
        best_score = score
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("✔ Best model saved")

    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": best_score
    }, CHECKPOINT_PATH)


# save logs
with open(METRICS_PATH, "w") as f:
    json.dump(history, f, indent=4)

print("\nTraining complete. Run saved in:", RUN_DIR)


# =========================
# PROBABILITY THRESHOLD TUNING
# =========================
def get_probs(model, loader):
    model.eval()

    probs, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]

            probs.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(probs), np.array(labels)


def tune_threshold(probs, labels):
    best_t, best_score = 0.5, -1

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)

        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        score = 0.7 * recall + 0.3 * precision

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


probs, labels = get_probs(model, val_loader)
best_threshold, thr_score = tune_threshold(probs, labels)

print("\n✔ Best threshold:", best_threshold)
print("✔ Threshold score:", thr_score)


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
        self.model.eval()

        logits = self.model(x)
        class_idx = logits.argmax(dim=1)

        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam


def show_cam(img_tensor, cam):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225])
    img = img + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap[..., ::-1] / 255.0

    overlay = 0.6 * heatmap + img

    plt.imshow(overlay)
    plt.axis("off")
    plt.show()


target_layer = model.layer4[-1]
cam = GradCAM(model, target_layer)

x, y = next(iter(val_loader))
x = x[:1].to(device)

heatmap = cam.generate(x)
show_cam(x[0], heatmap)