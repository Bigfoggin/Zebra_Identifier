import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def evaluate_weighted_model():
    # --- 1. Path & Device Settings ---
    # Updated to point to your new weighted model folder
    model_path = "models/sebra_weighted/final_model.pth"
    data_dir = "split_data/val"
    
    # Using CPU for evaluation is usually safer on cluster login nodes
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists(data_dir):
        print(f"Error: Validation directory not found at {data_dir}")
        return

    val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    # We use num_workers=4 here; since it's just validation, we don't need 32 CPUs
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    class_names = val_dataset.classes # Should be ['n', 'y']

    # --- 3. Initialize Model Architecture ---
    # Matches the sebra_weighted.py architecture
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    model = model.to(device)

    # --- 4. Load Trained Weights ---
    if not os.path.exists(model_path):
        print(f"Error: Weighted model file not found at {model_path}")
        print("Did the training job finish successfully?")
        return
        
    print(f"Loading weights from {model_path}...")
    # map_location=device ensures it loads on CPU even if trained on GPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # --- 5. Run Inference ---
    print(f"Evaluating {len(val_dataset)} images from the validation set...")
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Using Green to distinguish from old model
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted (Model Guess)')
    plt.ylabel('Actual (Truth)')
    plt.title('Weighted Zebra Identifier - Confusion Matrix')
    
    save_path = 'weighted_confusion_matrix.png'
    plt.savefig(save_path)
    print(f"\nSuccess! Confusion Matrix saved as {save_path}")

    # --- 7. Print Detailed Text Report ---
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT (WEIGHTED MODEL)")
    print("="*30)
    # This report is crucial because it will show the 'Recall' for the 'y' class
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_weighted_model()