from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Only images are allowed
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the saved deepfake detector model
model_path = 'deepfake_detector_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    """Return True if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(224, 224)):
    """Read and preprocess the image file for model prediction."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0  # Normalize pixel values
    return image

def predict_image(img_path):
    """Run prediction on the preprocessed image."""
    img = preprocess_image(img_path)
    if img is None:
        return "Error reading image", 0.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)[0]
    # Assuming threshold 0.5 for binary classification
    if prediction[0] > 0.5:
        label = "Real"
        confidence = prediction[0]
    else:
        label = "Fake"
        confidence = 1 - prediction[0]
    return label, confidence

@app.route('/')
def index():
    """Serve the main index page."""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """Serve the favicon to avoid 404 errors."""
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict_route():
    """Endpoint to handle image upload and prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            label, confidence = predict_image(file_path)
            
            # Clean up: remove the temporary file after prediction
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Convert confidence to a standard Python float to ensure JSON serialization.
            return jsonify({'label': label, 'confidence': float(confidence)})
        else:
            return jsonify({'error': 'File type not allowed'}), 400

    except Exception as e:
        print("Error in /predict route:", e)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler to catch unhandled exceptions and return JSON."""
    print(f"Unhandled Exception: {e}")
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
