Breast Cancer Classification using Transfer Learning and Grad-CAM

This project aims to classify breast cancer histopathology images into **Benign** and **Malignant** categories using transfer learning with popular CNN architectures (DenseNet-121, ResNet-50, VGG-16). In addition, we leverage **Grad-CAM** (Gradient-weighted Class Activation Mapping) for visual explanation and combine outputs from multiple models to train a final classifier on Grad-CAM heatmaps.

Project Structure
- `cancer_class_densenet_121.py`: Training using DenseNet-121.
- `cancer_class_resnet_50.py`: Training using ResNet-50.
- `cancer_class_vgg_16.py`: Training using VGG-16.
- `cancer_class_further.py`: Combines Grad-CAM outputs from all three models to train a final classifier.

Dataset
Name: BreaKHis Dataset
Description: Contains microscopic images of breast tumor tissue samples.
Classes: Benign (label `0`) and Malignant (label `1`)

Models Used
Pretrained **DenseNet-121**, **ResNet-50**, and **VGG-16**
Final MLP Classifier trained on combined Grad-CAM maps from the above models

Grad-CAM Visualization:
-Grad-CAM heatmaps are generated from the last convolutional layer of each model.
- Combined Grad-CAM output is used as an input to a classifier to improve interpretability and performance.

Training & Evaluation:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam / AdamW
- Epochs: 10
- Augmentations: Resizing, Horizontal/Vertical Flip, Rotation, Normalization (based on ImageNet)

Final Classification Architecture:
The final classifier is a feed-forward neural network trained on combined Grad-CAM heatmaps with:
- Multiple dense layers
- Batch normalization
- ReLU activations
- Dropout regularization

Output Models:
- `fine_tuned_densenet121.pth`
- `fine_tuned_resnet50.pth`
- `fine_tuned_vgg16.pth`
- `final_classifier.pth`

Dependencies:
- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- tqdm
- PIL (Pillow)
- NumPy

```bash
pip install torch torchvision scikit-learn tqdm pillow numpy

Future Improvements:
Add GUI for visualization
Use more ensemble techniques
Explore other interpretability techniques like LIME or SHAP

Acknowledgements:
Dataset: BreaKHis Dataset
Transfer learning models from PyTorch Model Zoo
