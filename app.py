# app.py
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    try:
        file = request.files['image']
        
        # Save the file temporarily
        file_path = "temp_image.jpg"
        file.save(file_path)
        
        # Load and preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Make prediction
        preds = model.predict(x)
        predictions = decode_predictions(preds, top=3)[0]
        
        # Format results
        results = [
            {'label': label, 'probability': float(prob) * 100}
            for (_, label, prob) in predictions
        ]
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)