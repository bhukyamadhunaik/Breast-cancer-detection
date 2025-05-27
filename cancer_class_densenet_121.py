import os
import glob
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models#provides all the pre trained models.
from PIL import Image
from pathlib import Path
from tqdm import tqdm
#99
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
breast_img_paths = glob.glob('../cancer_class/data/BreaKHis_v1/histology_slides/breast/**/*.png', recursive=True)
benign_images, malignant_images = [], []

for img in breast_img_paths:
    img_name = Path(img).name
    if img_name[4] == 'B':
        benign_images.append(img)
    else:
        malignant_images.append(img)

print(f"Total examples: {len(breast_img_paths)}")#7908 samples
print(f"Benign: {len(benign_images)}, Malignant: {len(malignant_images)}")#Malignant 5640

all_images = benign_images + malignant_images
all_labels = [0] * len(benign_images) + [1] * len(malignant_images)  # 0: Benign, 1: Malignant

#0.2 fraction for validation
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Data Augmentation for train data
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),#Fixed Input size
    transforms.RandomHorizontalFlip(),#Flipping some images horizontally
    transforms.RandomRotation(10),#rotating images by 10 degree
    transforms.ToTensor(),#converts the image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# to have 0 mean and 1 varience, values set as per ImageNet dataset
])

#Data processing for test data
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class
class BreakHisDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):#image paths corresponding labels provided
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.loader = default_loader  # Default image loader from torchvision

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.loader(self.image_paths[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Creating Dataset and DataLoader instances for train and validation sets
train_dataset = BreakHisDataset(train_images, train_labels, transform=transform_train)
val_dataset = BreakHisDataset(val_images, val_labels, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#shuffle is only required for train so that training happens effctively.

# Load and modify pretrained DenseNet121 model
model = models.densenet121(pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)  # Changing final layer to 2 classes (Benign and Malignant)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function with tqdm progress bar
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  #Model set to train Mode
        running_loss = 0.0#Loss for current epoch considering all batches

        # Initialize tqdm progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

                # Update tqdm progress bar with current loss
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

# Evaluation function
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

# Train the model
num_epochs = 10
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Save the fine-tuned model
model_save_path = "fine_tuned_densenet121.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'")


# model = models.densenet121(pretrained=False)
# model.classifier = nn.Linear(model.classifier.in_features, 2)
# model.load_state_dict(torch.load(model_save_path))
# model = model.to(device)
# model.eval()