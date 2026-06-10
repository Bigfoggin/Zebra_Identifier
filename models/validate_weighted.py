import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


def evaluate_weighted_model():
    # --- 1. Path & Device Settings ---
    model_path = "models/sebra_weighted/final_model.pth"

    # Using CPU for evaluation is usually safer on cluster login nodes
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Data Transforms ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- 5. Evaluate Each Dataset Split ---
    for split in ["train", "val", "test"]:

        data_dir = f"split_data/{split}"

        if not os.path.exists(data_dir):
            print(f"\nWarning: {data_dir} not found. Skipping {split}.")
            continue

        print("\n" + "=" * 50)
        print(f"EVALUATING {split.upper()} SET")
        print("=" * 50)

        dataset = datasets.ImageFolder(
            data_dir,
            transform=data_transforms
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )

        class_names = dataset.classes  # e.g. ['n', 'y']

        all_preds = []
        all_labels = []

        print(f"Evaluating {len(dataset)} images from the {split} set...")

        # --- 6. Run Inference ---
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # --- 7. Generate & Save Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Greens',
            xticklabels=class_names,
            yticklabels=class_names
        )

        plt.xlabel("Predicted (Model Guess)")
        plt.ylabel("Actual (Truth)")
        plt.title(f"Weighted Zebra Identifier - {split.capitalize()} Confusion Matrix")

        save_path = f"{split}_weighted_confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Success! Confusion Matrix saved as {save_path}")

        # --- 8. Print Classification Report ---
        print("\n" + "=" * 30)
        print(f"CLASSIFICATION REPORT ({split.upper()})")
        print("=" * 30)

        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=class_names
            )
        )


if __name__ == "__main__":
    evaluate_weighted_model()