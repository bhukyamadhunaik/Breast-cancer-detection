import os
import glob
from sklearn.model_selection import train_test_split 
from torchvision.datasets.folder import default_loader#for loading the image
from torch.utils.data import Dataset, DataLoader#for loading the dataset in batches
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path
from tqdm import tqdm
#97-98 approx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


breast_img_paths = glob.glob('../cancer_class/data/BreaKHis_v1/histology_slides/breast/**/*.png', recursive=True)
benign_images, malignant_images = [], []

for img in breast_img_paths:
    img_name = Path(img).name
    if img_name[4] == 'B':
        benign_images.append(img)
    else:
        malignant_images.append(img)

print(f"Total examples: {len(breast_img_paths)}")
print(f"Benign: {len(benign_images)}, Malignant: {len(malignant_images)}")

all_images = benign_images + malignant_images
all_labels = [0] * len(benign_images) + [1] * len(malignant_images)  # 0: Benign, 1: Malignant

# Train-test split
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# Define image transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #they correspond to the mean and standard deviation of the ImageNet dataset.
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Custom Dataset class
class BreakHisDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
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

# Create Dataset and DataLoader instances for train and validation sets
train_dataset = BreakHisDataset(train_images, train_labels, transform=transform_train)
val_dataset = BreakHisDataset(val_images, val_labels, transform=transform_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load and modify pretrained VGG16 model
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)  # Change final layer to have 2 output neurons for (Benign and Malignant)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function with tqdm progress bar
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

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

        # Validation phase after completion of each Epoch
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
model_save_path = "fine_tuned_vgg16.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to '{model_save_path}'")


# model = models.vgg16(pretrained=False)
# model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
# model.load_state_dict(torch.load(model_save_path))
# model = model.to(device)
# model.eval()