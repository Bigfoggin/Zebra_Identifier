import os
import random
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as TF
from tqdm import tqdm  # Ensure this is installed: pip install tqdm

def augment_dataset(base_path):
    categories = ['y', 'n']
    
    # Outer loop for categories (y/n)
    for cat in categories:
        folder_path = os.path.join(base_path, cat)
        if not os.path.exists(folder_path):
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Inner loop with progress bar
        # desc: Label shown on the left of the bar
        # unit: The item being counted
        for filename in tqdm(files, desc=f"Augmenting class '{cat}'", unit="img"):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                # Use tqdm.write to avoid breaking the progress bar layout
                tqdm.write(f"Skipping {filename}: {e}")
                continue
                
            name, ext = os.path.splitext(filename)

            # 1. GAUSSIAN BLUR
            blur_img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
            blur_img.save(os.path.join(folder_path, f"{name}_blur{ext}"))

            # 2. GRAYSCALE
            gray_img = img.convert('L').convert('RGB')
            gray_img.save(os.path.join(folder_path, f"{name}_gray{ext}"))

            # 3. SHADOW & CONTRAST
            enhancer_b = ImageEnhance.Brightness(img)
            shadow_img = enhancer_b.enhance(0.5)
            enhancer_c = ImageEnhance.Contrast(shadow_img)
            shadow_img = enhancer_c.enhance(1.4)
            shadow_img.save(os.path.join(folder_path, f"{name}_shadow{ext}"))

            # 4. SKEW / AFFINE TRANSFORM
            skew_img = TF.affine(
                img, 
                angle=0, 
                translate=[0, 0], 
                scale=1.0, 
                shear=[random.randint(10, 15), random.randint(0, 5)],
                fill=128
            )
            skew_img.save(os.path.join(folder_path, f"{name}_skew{ext}"))

if __name__ == "__main__":
    train_path = "split_data/train"
    if os.path.exists(train_path):
        augment_dataset(train_path)
    else:
        print(f"Error: Path {train_path} not found.")