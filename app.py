import gdown
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Download the model from Google Drive
model_url = "https://drive.google.com/uc?id=1eKanoLsBZQ2c61Z5X0LW8_E6GBip8KGL"
gdown.download(model_url, 'model.h5', quiet=False)

# Load the model
model = tf.keras.models.load_model('model.h5')

def predict(image):
    # Resize the image to match the model's input size
    image = image.resize((224, 224))  # You can adjust the size if necessary
    img_array = np.array(image) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get the prediction
    prediction = model.predict(img_array)
    return prediction.tolist()

@app.route('/predict', methods=['POST'])
def upload_image():
    # Get the image from the request
    file = request.files['file']
    image = Image.open(file)  # Open the image using PIL

    # Get the prediction
    prediction = predict(image)
    
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
