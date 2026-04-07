import os
import shutil
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def export_errors():
    # 1. Setup
    model_path = "models/sebra_classifier/final_model.pth"
    data_dir = "split_data/val"
    error_dir = "error_analysis"
    device = torch.device("cpu")

    # 2. Model & Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    # We set batch_size=1 to easily map image paths
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Create folders for the images
    os.makedirs(f"{error_dir}/missed_zebras", exist_ok=True) # Truth: y, Pred: n
    os.makedirs(f"{error_dir}/false_alarms", exist_ok=True)  # Truth: n, Pred: y

    print("Analyzing images...")
    
    # dataset.samples is a list of (path, class_index)
    image_paths = [s[0] for s in dataset.samples]

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            true_label = labels.item()
            pred_label = preds.item()
            img_path = image_paths[i]
            filename = os.path.basename(img_path)

            # MISSED ZEBRA (False Negative)
            if true_label == 1 and pred_label == 0:
                shutil.copy(img_path, f"{error_dir}/missed_zebras/{filename}")
            
            # FALSE ALARM (False Positive)
            elif true_label == 0 and pred_label == 1:
                shutil.copy(img_path, f"{error_dir}/false_alarms/{filename}")

    print(f"Done! Check the '{error_dir}' folder to see the images.")

if __name__ == "__main__":
    export_errors()