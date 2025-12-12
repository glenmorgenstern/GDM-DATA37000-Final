#%%
# %pip install openimages
# 
# %%
# Download Open Images subset
# import os
# from dataset_utils import download_openimages_subset, preview_imagefolder

# output = download_openimages_subset(
#     classes=["Dog", "Cat", "Car", "Tree", "Bicycle"],
#     max_samples=1000,
#     export_dir=r"C:\Users\glenm\Downloads\_git\DATA37000\data\bigdata\open_images_subset"
# )

# preview_imagefolder(output)
# print("Download Done.")

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
# Define transforms
from torchvision import transforms

# ImageNet normalization stats (used for pretrained models)
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

# Training transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # normalize size
    transforms.RandomHorizontalFlip(),  # simple augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Validation/test transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

#%%
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

DATA_DIR = r"C:\Users\glenm\Downloads\_git\DATA37000\data\bigdata\open_images_subset"

# Load full dataset with training transforms
full_dataset = ImageFolder(DATA_DIR, transform=train_transforms)

# Split into train/val (80/20)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply val transforms to validation set
val_dataset.dataset.transform = val_transforms

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


# %%
# Define simple CNN baseline model
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # assuming input 224x224
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x112x112
        x = self.pool(F.relu(self.conv2(x)))  # 64x56x56
        x = self.pool(F.relu(self.conv3(x)))  # 128x28x28
        x = x.view(-1, 128 * 28 * 28)         # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
# %%
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%")
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    print(f"Validation Acc: {val_acc:.2f}%")
# %%
