# Save this file as app.py

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import numpy as np
import io
import os
import cv2
from model import CNN

# --- 1. SETUP ---

# Load the trained model
print("Loading EMNIST model...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# This should be the model trained on EMNIST with 62 classes.
model_path = r'C:/Users/Matthew/Documents/EMNIST/Handwritten-digit-recogniser-extended/scripts/models/best_model_fold_5.pt' 

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please place your trained .pt file here.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from {model_path} and set to evaluation mode.")

# EMNIST ByClass mapping: 0-9, A-Z, a-z
# The model outputs an index from 0-61. This maps it to a character.
label_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# --- 2. GRAD-CAM IMPLEMENTATION ---
# This function generates a heatmap showing which parts of the image the model focused on.

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Use the specific class index for the backward pass
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        output[0][class_idx].backward(retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        
        for i in range(pooled_gradients.shape[0]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        return heatmap

# Initialise Grad-CAM on the last convolutional layer of the model
grad_cam = GradCAM(model=model, target_layer=model.conv3)


def generate_heatmap_image(heatmap_data, original_size=(28, 28)):
    heatmap_resized = cv2.resize(heatmap_data, original_size)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Convert to PNG bytes
    is_success, buffer = cv2.imencode(".png", heatmap_colored)
    if not is_success:
        return None
    return io.BytesIO(buffer)


# --- 3. FLASK APP ---

app = Flask(__name__, static_url_path='', static_folder='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # --- Image Preprocessing ---
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Preprocessing: crop, resize, pad
        image_np = np.array(image)
        coords = np.argwhere(image_np > 50)
        if coords.size == 0:
            return jsonify({'error': 'No character found in image'})

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        cropped = image_np[y0:y1, x0:x1]

        cropped_img = Image.fromarray(cropped)
        cropped_img.thumbnail((20, 20), Image.LANCZOS)
        
        padded_img = Image.new('L', (28, 28), 0)
        paste_x = (28 - cropped_img.width) // 2
        paste_y = (28 - cropped_img.height) // 2
        padded_img.paste(cropped_img, (paste_x, paste_y))

        # --- Prediction Logic ---
        padded_np = np.array(padded_img).astype(np.float32) / 255.0
        mean, std = 0.1307, 0.3081
        normed = (padded_np - mean) / std
        tensor = torch.tensor(normed).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            probabilities = F.softmax(output, dim=1)
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities, 3)

        # --- Generate Grad-CAM ---
        top_prediction_index = top3_indices[0][0].item()
        heatmap_data = grad_cam.generate(tensor, class_idx=top_prediction_index)
        heatmap_img = cv2.resize(heatmap_data, (280, 280)) # Resize to canvas size
        heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap_img), cv2.COLORMAP_JET)
        
        import base64
        _, buffer = cv2.imencode('.png', heatmap_img)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')

        # --- Format Response ---
        predictions = []
        for i in range(top3_probs.shape[1]):
            prob = top3_probs[0][i].item()
            idx = top3_indices[0][i].item()
            predictions.append({
                'label': label_map[idx],
                'confidence': f"{prob:.4f}"
            })
        
        return jsonify({
            'top_predictions': predictions,
            'heatmap': f"data:image/png;base64,{heatmap_base64}"
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/log_correction', methods=['POST'])
def log_correction():
    data = request.json
    image_b64 = data['image']
    correct_label = data['label']
    
    # Log the correction to a file
    with open("corrections.log", "a") as f:
        # Save the base64 image data and the correct label
        f.write(f"{correct_label},{image_b64}\n")
        
    print(f"Logged correction: Label={correct_label}")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)