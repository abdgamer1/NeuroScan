import os
import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the model file name
MODEL_PATH = 'model.h5'

# Check if the model already exists, if not, download it
if not os.path.exists(MODEL_PATH):
    model_url = "https://drive.google.com/uc?id=1eKanoLsBZQ2c61Z5X0LW8_E6GBip8KGL"
    print("Model not found. Downloading...")
    gdown.download(model_url, MODEL_PATH, quiet=False)
else:
    print("Model already exists, skipping download.")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def predict(image):
    """Preprocess image and make prediction."""
    # Resize image to match the model's input size
    image = image.resize((224, 224))  # You can adjust this if needed
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get the prediction
    prediction = model.predict(img_array)
    return prediction.tolist()

@app.route('/predict', methods=['POST'])
def upload_image():
    """Handle image upload and return prediction."""
    try:
        # Get the image from the request
        file = request.files['file']
        image = Image.open(file)  # Open the image using PIL

        # Get the prediction
        prediction = predict(image)
        
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
