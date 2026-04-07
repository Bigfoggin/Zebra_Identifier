import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def evaluate_model():
    # Settings
    data_dir = os.path.abspath("split_data/val")
    model_path = "models/sebra_classificator/final_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Data
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    class_names = val_dataset.classes # ['n', 'y']

    # 2. Load Model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # 3. Collect Predictions
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # 5. Visualize with Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted (Model Guess)')
    plt.ylabel('Actual (Truth)')
    plt.title('Zebra Identifier Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as confusion_matrix.png")

    # 6. Detailed Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    evaluate_model()