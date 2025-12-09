#%%
# %pip install openimages
# 
# %%
# Download Open Images subset
import os
from dataset_utils import download_openimages_subset, preview_imagefolder

output = download_openimages_subset(
    classes=["Dog", "Cat", "Car", "Tree", "Bicycle"],
    max_samples=1000,
    export_dir=r"C:\Users\glenm\Downloads\_git\DATA37000\data\bigdata\open_images_subset"
)

preview_imagefolder(output)
print("Download Done.")

#%%
# Exploratory Data Analysis
import os
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import seaborn as sns

# Path to downloaded dataset
DATA_DIR = r"C:\Users\glenm\Downloads\_git\DATA37000\data\bigdata\open_images_subset"

# Helper: get class folders
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print("Classes found:", classes)

# 1. Show examples of images
def show_examples_per_class(classes, n=3):
    fig, axes = plt.subplots(len(classes), n, figsize=(n*3, len(classes)*3))
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(DATA_DIR, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        for j in range(n):
            img_path = os.path.join(cls_dir, images[j])
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(cls)
    plt.tight_layout()
    plt.show()

show_examples_per_class(classes, n=3)

# 2. Class distribution
counts = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in classes}
print("Class counts:", counts)

plt.figure(figsize=(8,5))
sns.barplot(x=list(counts.keys()), y=list(counts.values()))
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.show()

# 3. Image sizes & quality
sizes = []
for cls in classes:
    cls_dir = os.path.join(DATA_DIR, cls)
    images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in images[:50]:  # sample first 50 per class for speed
        img_path = os.path.join(cls_dir, f)
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)  # (width, height)
        except Exception as e:
            print(f"Error opening {img_path}: {e}")

# Convert to width/height lists
widths = [w for w,h in sizes]
heights = [h for w,h in sizes]

plt.figure(figsize=(6,4))
sns.scatterplot(x=widths, y=heights)
plt.xlabel("Width")
plt.ylabel("Height")
plt.title("Image Size Distribution (sampled)")
plt.show()

print(f"Average width: {sum(widths)/len(widths):.1f}, Average height: {sum(heights)/len(heights):.1f}")
print(f"Min size: {min(widths)}x{min(heights)}, Max size: {max(widths)}x{max(heights)}")

#%%


