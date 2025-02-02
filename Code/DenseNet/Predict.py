import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Step 1: Define the model architecture
model = models.densenet121(pretrained=False)  # Use DenseNet-121
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 4)  # Set the output layer to match your class count

# Step 2: Load the trained weights
model.load_state_dict(torch.load('C:/Users/Abhinav/OneDrive/Documents/MAJOR PROJECT/Code/DenseNet/best_densenet_model.pth'))
model.eval()  # Set the model to evaluation mode

# Step 3: Preprocess the image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),         # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ImageNet
    ])
    image = Image.open(img_path).convert('RGB')  # Open and ensure the image is in RGB mode
    return transform(image).unsqueeze(0)  # Add batch dimension

# Step 4: Make a prediction
def predict_image_class(img_path):
    img_tensor = preprocess_image(img_path)  # Preprocess the image
    
    # Get the model's predictions
    with torch.no_grad():  # Disable gradient calculations
        predictions = model(img_tensor)
    
    # Get the predicted class (highest probability)
    predicted_class = torch.argmax(predictions, dim=1).item()
    
    return predicted_class, predictions

# Step 5: Map predicted class to label
def map_class_to_label(predicted_class):
    class_labels = ['CNV', 'DME', 'DRUNSEN', 'NORMAL']  # Replace with your actual class labels
    return class_labels[predicted_class]

# Example usage
img_path = 'C:/Users/Abhinav/OneDrive/Documents/MAJOR PROJECT/Dataset/TEST/DRUSEN/DRUNSEN  (5).jpeg'  # Path to the image

# Predict the class of the image
predicted_class, predictions = predict_image_class(img_path)

# Map the predicted class to the label
predicted_label = map_class_to_label(predicted_class)

# Output the prediction results
print(f"Predicted label: {predicted_label}")  # Human-readable label
print(f"Prediction probabilities: {predictions}")  # Class probabilities