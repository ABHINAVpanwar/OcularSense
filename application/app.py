from flask import Flask, jsonify, request, render_template, send_file, redirect, url_for
import os
from datetime import datetime
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'best_densenet_model.pth')

# Directory for storing uploaded images and results
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploaded_data')
RESULTS_FILE = os.path.join(app.root_path, 'classification_report.txt')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Add this line to fix the KeyError

# Step 1: Define the model architecture
model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 4)

# Step 2: Load the trained weights
model.load_state_dict(torch.load(model_path))
model.eval()

# Step 3: Preprocess the image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(img_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Step 4: Predict the image class
def predict_image_class(img_path):
    img_tensor = preprocess_image(img_path)
    with torch.no_grad():
        predictions = model(img_tensor)
    predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = torch.softmax(predictions, dim=1)[0, predicted_class].item() * 100
    if confidence < 70:
        return -1, confidence
    return predicted_class, confidence

# Step 5: Map predicted class to label
def map_class_to_label(predicted_class):
    class_labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    if predicted_class not in range(len(class_labels)):  # Handle invalid index
        return "OCT_UNKNOWN"
    else:
        return class_labels[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []

    if request.method == 'POST':
        # Clear previous files in upload folder
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            os.remove(file_path)

        # Clear the predictions file
        with open(RESULTS_FILE, 'w') as f:
            f.write("=============================" + "\n")
            f.write("OCT Image Classification Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=============================" + "\n")
            f.write("\nFilename                     | Predicted Label | Confidence\n")
            f.write("------------------------------------------------------------\n")

        if 'images' not in request.files:
            return render_template('index.html', error='No file part', results=None)
        
        files = request.files.getlist('images')

        for file in files:
            if file.filename == '':
                continue
            
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Predict the class
            predicted_class, confidence = predict_image_class(file_path)
            predicted_label = map_class_to_label(predicted_class)

            # Append results for display
            results.append({
                'filename': file.filename,
                'predicted_label': predicted_label,
                'confidence': f"{confidence:.2f}%"
            })

            # Append prediction to the text file
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{file.filename:<30} | {predicted_label:<15} | {confidence:.2f}%\n")

        with open(RESULTS_FILE, 'a') as f:
            f.write("------------------------------------------------------------\n")
            f.write(f"Total Predictions: {len(results)}\n")
            f.write("------------------------------------------------------------\n")

    return render_template('index.html', results=results)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form.get('username')
    password = request.form.get('password')

    if username == "admin" and password == "123":
        return jsonify({"message": "Login Successful"})
    else:
        return jsonify({"message": "Invalid Credentials"}), 401

@app.route('/download')
def download_file():
    return send_file(RESULTS_FILE, as_attachment=True)

@app.route('/uploaded_images/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
