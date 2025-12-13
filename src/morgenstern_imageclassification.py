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

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Transforms
# -----------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# -----------------------------
# Dataset + Loaders
# -----------------------------
DATA_DIR = r"C:\Users\glenm\Downloads\_git\DATA37000\data\bigdata\open_images_subset"

full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transforms

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

class_names = val_dataset.dataset.classes
num_classes = len(class_names)

# -----------------------------
# Baseline CNN
# -----------------------------
class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*28*28, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128*28*28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -----------------------------
# ResNet18 Transfer Learning
# -----------------------------
def build_resnet18(num_classes=5):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

# -----------------------------
# Training function
# -----------------------------
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, patience=5, save_path="model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc, counter = 0, 0
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
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
        train_accs.append(train_acc)

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
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    return train_accs, val_accs

# -----------------------------
# Evaluation: Confusion Matrix + ROC/AUC
# -----------------------------
def evaluate_model(model, val_loader, model_name="Model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ROC/AUC (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(y_scores)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# -----------------------------
# Train and Evaluate Both Models
# -----------------------------
baseline_model = BaselineCNN(num_classes=num_classes)
baseline_train_accs, baseline_val_accs = train_model(baseline_model, train_loader, val_loader,
                                                     num_epochs=10, save_path="baseline_best.pth")

resnet_model = build_resnet18(num_classes=num_classes)
resnet_train_accs, resnet_val_accs = train_model(resnet_model, train_loader, val_loader,
                                                 num_epochs=15, save_path="resnet_best.pth")

# Plot training/validation curves
plt.figure(figsize=(8,6))
plt.plot(range(1, len(baseline_train_accs)+1), baseline_train_accs, label="Baseline Train")
plt.plot(range(1, len(baseline_val_accs)+1), baseline_val_accs, label="Baseline Val")
plt.plot(range(1, len(resnet_train_accs)+1), resnet_train_accs, label="ResNet18 Train")
plt.plot(range(1, len(resnet_val_accs)+1), resnet_val_accs, label="ResNet18 Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy Curves")
plt.legend()
plt.grid(True)
plt.show()

# Reload best weights before evaluation
baseline_model.load_state_dict(torch.load("baseline_best.pth", weights_only=True))
evaluate_model(baseline_model, val_loader, model_name="Baseline CNN")

resnet_model.load_state_dict(torch.load("resnet_best.pth", weights_only=True))
evaluate_model(resnet_model, val_loader, model_name="ResNet18")
# %%
