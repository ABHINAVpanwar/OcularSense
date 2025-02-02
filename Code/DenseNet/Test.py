import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Validation dataset path
val_path = "C:/Users/Abhinav/OneDrive/Documents/MAJOR PROJECT/Dataset/TEST"

# Transformation for validation data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load validation dataset
val_dataset = datasets.ImageFolder(val_path, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Class names
class_names = list(val_dataset.class_to_idx.keys())

# Load the saved model
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
num_classes = len(class_names)
num_features = model.classifier.in_features
model.classifier = torch.nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load('C:/Users/Abhinav/OneDrive/Documents/MAJOR PROJECT/Code/DenseNet/best_densenet_model.pth'))
model = model.to(device)
model.eval()

# Initialize metrics
all_labels = []
all_preds = []
all_probs = []

# Perform inference
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names)
print("Classification Report:")
print(report)

# Plot ROC Curves for each class
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    fpr, tpr, _ = roc_curve(all_labels == i, all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Class-wise Accuracy
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, accuracy in enumerate(class_accuracy):
    print(f"Class {class_names[i]} Accuracy: {accuracy:.2f}")

# Overall Accuracy
overall_accuracy = np.sum(conf_matrix.diagonal()) / np.sum(conf_matrix)
print(f"Overall Accuracy: {overall_accuracy:.2f}")