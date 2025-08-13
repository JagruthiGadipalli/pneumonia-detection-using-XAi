from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__, template_folder='templates')
CORS(app)

# Load model with error handling
try:
    model = tf.keras.models.load_model('pneumonia_xray_model.h5')
    print("Model loaded successfully")
    # Print model summary to verify layers (optional, for debugging)
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def get_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv_layer'):  # Updated layer name
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)  # Add small epsilon to avoid division by zero
        return heatmap
    except Exception as e:
        print(f"GradCAM error: {e}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
            
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        probability = float(prediction[0][0])
        result = 'Pneumonia' if probability > 0.5 else 'Normal'
        
        heatmap = get_gradcam_heatmap(img_array, model)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        superimposed_img = heatmap * 0.4 + np.array(img)
        superimposed_img = np.uint8(superimposed_img)
        
        _, img_encoded = cv2.imencode('.png', superimposed_img)
        heatmap_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        return jsonify({
            'prediction': result,
            'probability': probability,
            'heatmap': f'data:image/png;base64,{heatmap_base64}'
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)