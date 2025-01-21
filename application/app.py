from flask import Flask, jsonify, request, render_template, send_file, redirect, url_for
import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Directory for storing uploaded images and results
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploaded_data')
RESULTS_FILE = os.path.join(app.root_path, 'classification_report.txt')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

# Initialize model
model_path = os.path.join(app.root_path, 'best_resnet_model.pth')
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model.load_state_dict(torch.load(model_path))
model.eval()

# Preprocessing for images
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Prediction function
def predict_image_class(img_path):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        predictions = model(img_tensor)
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = torch.softmax(predictions, dim=1)[0, predicted_class].item() * 100
    return predicted_class, confidence

# Map class index to labels
def map_class_to_label(predicted_class):
    class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    return class_labels[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []

    if request.method == 'POST':
        # Clear previous uploaded files
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Clear or initialize results file
        with open(RESULTS_FILE, 'w') as f:
            f.write("=============================\n")
            f.write("OCT Image Classification Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=============================\n")
            f.write("\nFilename                     | Predicted Label | Confidence\n")
            f.write("------------------------------------------------------------\n")

        if 'images' not in request.files:
            return render_template('index.html', error='No file part', results=None)

        files = request.files.getlist('images')

        for file in files:
            if file.filename == '':
                continue

            # Save the file in the upload folder
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Predict the class
            predicted_class, confidence = predict_image_class(file_path)
            predicted_label = map_class_to_label(predicted_class)

            # Append result
            results.append({
                'filename': file.filename,
                'predicted_label': predicted_label,
                'confidence': f"{confidence:.2f}%"
            })

            # Log result in the results file
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{file.filename:<30} | {predicted_label:<15} | {confidence:.2f}%\n")

        with open(RESULTS_FILE, 'a') as f:
            f.write("------------------------------------------------------------\n")
            f.write(f"Total Predictions: {len(results)}\n")
            f.write("------------------------------------------------------------\n")

    return render_template('index.html', results=results)

@app.route('/download')
def download_file():
    return send_file(RESULTS_FILE, as_attachment=True)

@app.route('/uploaded_images/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
