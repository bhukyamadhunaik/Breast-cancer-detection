import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, vgg16, densenet121
from torchvision.models import ResNet50_Weights, VGG16_Weights, DenseNet121_Weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
from pathlib import Path
import glob
import numpy as np
from tqdm import tqdm
import os
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


breast_img_paths = glob.glob('../cancer_class/data/BreaKHis_v1/histology_slides/breast/**/*.png', recursive=True)
benign_images, malignant_images = [], []

for img in breast_img_paths:
    img_name = Path(img).name
    if img_name[4] == 'B':
        benign_images.append(img)
    else:
        malignant_images.append(img)

all_images = benign_images + malignant_images
all_labels = [0] * len(benign_images) + [1] * len(malignant_images)
# 0: Benign, 1: Malignant

# Train-test split
train_images, val_images, train_labels, val_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# applies the transformations on the fly to each image as it is loaded during the data loading process.

#Loading fine-tuned resnet50
model_resnet = resnet50(weights=None)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 2)#modifying the final layer
model_resnet.load_state_dict(torch.load("fine_tuned_resnet50.pth",map_location=torch.device('cpu') ,weights_only=True))
model_resnet = model_resnet.to(device)

#Loading fine tuned vgg-16
model_vgg = vgg16(weights=None)
model_vgg.classifier[6] = nn.Linear(model_vgg.classifier[6].in_features, 2)
model_vgg.load_state_dict(torch.load("fine_tuned_vgg16.pth", map_location=torch.device('cpu'), weights_only=True))
model_vgg = model_vgg.to(device)

class ModifiedDenseNet(nn.Module):
    def __init__(self, original_model):
        super(ModifiedDenseNet, self).__init__()
        self.features = original_model.features#Contains the convolutional feature extractor of the DenseNet model.
        self.classifier = nn.Linear(original_model.classifier.in_features, 2)#Modified for binary classification

    def forward(self, x):
        features = self.features(x)#feature Extraction
        out = F.relu(features, inplace=False)#Applying RELU to feature map
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        out = self.classifier(out)
        return out

#Loading denset121 fine tuned model
original_densenet = densenet121(weights=None)
model_densenet = ModifiedDenseNet(original_densenet)
model_densenet.load_state_dict(torch.load("fine_tuned_densenet121.pth", map_location=torch.device('cpu'), weights_only=True))
model_densenet = model_densenet.to(device)

#Set models to evaluation mode
models = [model_resnet, model_densenet, model_vgg]
target_layers = [model_resnet.layer4, model_densenet.features, model_vgg.features[-1]]

# Define Grad-CAM generation function wrt individual pre-trained model
def generate_individual_gradcams(model, image_tensor, target_layer):
    activations, gradients = [], []
    #activations will store activations of target layer during forward pass
    #gradients will store the gradients of model parameters computed during the backward pass.

    #capture activations in target layer
    def forward_hook(module, input, output):
        activations.append(output)

    #captures the gradients flowing through the target layer during the backward pass.
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer.register_forward_hook(forward_hook)#Attach forward hook to the target layer
    target_layer.register_full_backward_hook(backward_hook)#Attach backward hook to the target layer

    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()#predicted class ouput by the given model


    model.zero_grad()
    output[0, class_idx].backward()#gradients of output class score wrt to feature maps obtained in 
    #last convulation layers
    activation = activations[0].detach()
    gradient = gradients[0].detach()
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3])

    for i in range(pooled_gradients.size(0)):
        activation[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=1).squeeze().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)#RELU
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1#Normalise
    
   
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)
    heatmap_resized = F.interpolate(heatmap_tensor, size=(224, 224), mode='bilinear', align_corners=False)
    heatmap_resized = heatmap_resized.squeeze().numpy()  # Convert back to NumPy array

    return heatmap_resized

#Generating GradCamRes=gradCam1+gradCam2+gradCam3 for given image using all 3 pretrained models
def generate_combined_gradcam(image_tensor, models, target_layers):
    gradcams = [generate_individual_gradcams(model, image_tensor, layer) for model, layer in zip(models, target_layers)]
    combined_gradcam = sum(gradcams)
    return combined_gradcam

#Generating Grad-CAM maps for all images and save them
# gradcam_data = []
#for image_path, label in tqdm(zip(all_images, all_labels), desc="Generating Grad-CAMs", total=len(all_images)):
#image_tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
#gradcam_res = generate_combined_gradcam(image_tensor, models, target_layers)
#gradcam_data.append((gradcam_res, label))#-->gradCamRes,corresponding true_label

#Save Grad-CAM data to file
# with open("gradcam_data.pkl", "wb") as f:
#     pickle.dump(gradcam_data, f)
# print("Grad-CAM data saved to 'gradcam_data.pkl'.")


# Step 2: Load Grad-CAM data for training
class GradCamDataset(Dataset):
    def __init__(self, gradcam_data):
        self.gradcam_data = gradcam_data

    def __len__(self):
        return len(self.gradcam_data)

    def __getitem__(self, idx):
        gradcam_res, label = self.gradcam_data[idx]
        gradcam_res = torch.tensor(gradcam_res, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return gradcam_res, label

# Load preprocessed Grad-CAM data
with open("gradcam_data.pkl", "rb") as f:
    gradcam_data = pickle.load(f)
print(f"Loaded Grad-CAM data size: {len(gradcam_data)}")
print(f"Sample Grad-CAM shape: {gradcam_data[0][0].shape}")  # Assuming gradcam_data contains (gradcam_res, True_label)

# Split Grad-CAM data into training and validation sets
train_data = gradcam_data[:int(0.8 * len(gradcam_data))]#0 to 80
val_data = gradcam_data[int(0.8 * len(gradcam_data)):]#80 to 100

train_dataset = GradCamDataset(train_data)
val_dataset = GradCamDataset(val_data)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Step 3: Final Classifier for predicting from gradCamRes
class FinalClassifier(nn.Module):
    def __init__(self, input_height, input_width):#height and width of each input image
        super(FinalClassifier, self).__init__()
        self.fc = nn.Sequential(
            #1st Layer
            nn.Linear(input_height * input_width, 512),  # More neurons
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(),  # Activation Function
            nn.Dropout(0.5),  # Dropout for Regularization

            #2nd Layer
            nn.Linear(512, 256),  # Additional Layer
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),  # Activation Function
            nn.Dropout(0.5),  # Dropout for Regularization

            #3rd Layer
            nn.Linear(256, 128),  # Additional Layer
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),  # Activation Function
            nn.Dropout(0.3),  # Dropout for Regularization

            #Final Classifiying Layer
            nn.Linear(128, 2)  # Output Layer
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        return self.fc(x)

#Determine Grad-CAM size dynamically
sample_gradcam, _ = train_dataset[0]
input_height, input_width = sample_gradcam.shape[1:]  # Get height and width

final_classifier = FinalClassifier(input_height=input_height, input_width=input_width).to(device)

# Training the classifier
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(final_classifier.parameters(), lr=0.0001)
optimizer = torch.optim.AdamW(final_classifier.parameters(), lr=0.0001, weight_decay=1e-5)

epoches = 10

for epoch in range(epoches):
    final_classifier.train()
    running_loss = 0.0

    for gradcam_res, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epoches}"):
        gradcam_res, labels = gradcam_res.to(device), torch.tensor(labels).to(device)

        outputs = final_classifier(gradcam_res)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * gradcam_res.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{epoches}], Loss: {epoch_loss:.4f}")

# Save the trained classifier, trained model file which classifies based on the resultant gradcam
torch.save(final_classifier.state_dict(), "final_classifier.pth")
print("Model saved as 'final_classifier.pth'.")

# Evaluate the classifier
final_classifier.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for gradcam_res, labels in val_loader:
        gradcam_res, labels = gradcam_res.to(device), torch.tensor(labels).to(device)
        outputs = final_classifier(gradcam_res)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Final Classification Accuracy: {accuracy * 100:.2f}%")