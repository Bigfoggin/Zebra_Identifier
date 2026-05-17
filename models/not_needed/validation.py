import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def evaluate_model():
    # --- 1. Path & Device Settings ---
    # Matches your folder structure: models/sebra_classifier/
    model_path = "models/sebra_classifier/final_model.pth"
    data_dir = "split_data/val"
    
    # Using CPU to bypass the "Old NVIDIA Driver" error on the cluster
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        return

    val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = val_dataset.classes # Expected: ['n', 'y']

    # --- 3. Initialize Model Architecture ---
    # We must build the 'brain' structure BEFORE loading the weights
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2) # 2 classes: Zebra (y) vs No Zebra (n)
    model = model.to(device)

    # --- 4. Load Trained Weights ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # --- 5. Run Inference ---
    print("Running validation images through the model...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 6. Generate & Save Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted (Model Guess)')
    plt.ylabel('Actual (Truth)')
    plt.title('Zebra Identifier Confusion Matrix')
    
    save_path = 'confusion_matrix.png'
    plt.savefig(save_path)
    print(f"Success! Confusion Matrix saved as {save_path}")

    # --- 7. Print Detailed Text Report ---
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_model()