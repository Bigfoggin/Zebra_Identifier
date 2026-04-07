import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    # --- Configuration & Paths ---
    data_dir = os.path.abspath("split_data")
    checkpoint_path = "models/sebra_classifier/checkpoint.pth"
    final_model_path = "models/sebra_classifier/final_model.pth"
    os.makedirs("models/sebra_classifier", exist_ok=True)

    print(f"--- Step 1: Locating Data ---")
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at {data_dir}")
        return

    # --- Transforms & Data Loading ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"--- Step 2: Loading Datasets ---")
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
    
    # Change this in your .py file to use the 32 CPUs you reserved:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    # --- Model Setup ---
    print(f"--- Step 3: Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Checkpoint Loading (The "Resume" Logic) ---
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Resuming...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch + 1}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Training Loop ---
    num_epochs = 10
    print(f"\n--- Step 4: Training ---")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=running_loss/len(train_loader))
        
        # Validation
        model.eval()
        val_corrects = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
        
        val_acc = val_corrects.double() / len(val_dataset)
        print(f"Epoch {epoch+1} Complete. Accuracy: {val_acc:.4f}")

        # --- Save Checkpoint ---
        print(f"Saving checkpoint for epoch {epoch+1}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_path)

    # --- Final Save ---
    torch.save(model.state_dict(), final_model_path)
    print(f"Full training finished. Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()