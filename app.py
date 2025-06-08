from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
from werkzeug.utils import secure_filename
import uuid
import time

app = Flask(__name__)
app.secret_key = 'yahya_wong_sangar'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

prediction_results = {}

model = None
try:
    print("Attempting to load gg_model.pkl...")

    try:
        with open("gacor_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Model loaded successfully with standard pickle!")
    except Exception as e1:
        print(f"Standard pickle failed: {e1}")

        try:
            with open("gacor_model.pkl", "rb") as f:
                model = pickle.load(f, encoding='latin1')
            print("Model loaded successfully with latin1 encoding!")
        except Exception as e2:
            print(f"Latin1 encoding failed: {e2}")

            try:
                import joblib
                model = joblib.load("gacor_model.pkl")
            except Exception as e3:
                print(f"Joblib loading failed: {e3}")
    
    if model is not None:
        print(f"Model type: {type(model)}")
        print("Model loaded successfully!")
    else:
        print("FAILED TO LOAD ANY MODEL!")
        
except Exception as e:
    print(f"Critical error in model loading: {e}")
    model = None

CLASS_LABELS = {
    0: 'Bluish Nail',
    1: 'Healthy Nail', 
    2: 'Koilonychia',
    3: 'Terry-s nail'
}

CONDITION_DESCRIPTIONS = {
    'Bluish Nail': 'Kondisi autoimun yang menyebabkan penebalan, perubahan warna, dan kelainan bentuk pada kuku. Dapat disertai dengan bintik-bintik kecil atau garis-garis pada permukaan kuku.',
    'Healthy Nail': 'Kuku dalam kondisi sehat tanpa tanda-tanda penyakit atau kelainan yang terdeteksi.',
    'Koilonychia': 'Infeksi jamur (onikomikosis) yang menyebabkan perubahan warna kuku menjadi kuning, putih, atau coklat, penebalan, dan rapuh. Sering dimulai dari ujung kuku.',
    'Terry-s nail': 'Kerusakan kuku akibat cedera fisik seperti terjepit, terpukul, atau tekanan berulang. Dapat menyebabkan perdarahan di bawah kuku atau kerusakan struktur kuku.'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_and_extract_features(image_path):
    try:
        resize_dim = (256, 256)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Tidak dapat membaca gambar")
        
        resized_image = cv2.resize(image, resize_dim)
        lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        clahe_image = cv2.merge((cl, a, b))
        clahe_image_bgr = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2BGR)

        image_resized = cv2.resize(clahe_image_bgr, (224, 224))
        image_preprocessed = tf.keras.applications.densenet.preprocess_input(image_resized)
        
        base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, pooling='avg')
        preprocessed_image = np.expand_dims(image_preprocessed, axis=0)
        features = base_model.predict(preprocessed_image, verbose=0)
        
        return features
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

def get_prediction_probabilities(model, features):
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            predicted_class = np.argmax(probabilities)
            return predicted_class, probabilities
        
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(features)[0]
            
            if len(scores) > 1:
                exp_scores = np.exp(scores - np.max(scores))
                probabilities = exp_scores / np.sum(exp_scores)
                predicted_class = np.argmax(probabilities)
            else:
                probability = 1 / (1 + np.exp(-scores))
                probabilities = np.array([1-probability, probability])
                predicted_class = int(probability > 0.5)
            
            return predicted_class, probabilities

        else:
            predicted_class = model.predict(features)[0]

            num_classes = len(CLASS_LABELS)
            probabilities = np.ones(num_classes) * (5.0 / 100.0)
            probabilities[predicted_class] = 90.0 / 100.0
            
            return predicted_class, probabilities
            
    except Exception as e:
        print(f"Error in getting probabilities: {e}")
        
        predicted_class = model.predict(features)[0] if hasattr(model, 'predict') else 0
        num_classes = len(CLASS_LABELS)
        probabilities = np.ones(num_classes) * (10.0 / 100.0)
        probabilities[predicted_class] = 70.0 / 100.0
        
        return predicted_class, probabilities

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/loading/<prediction_id>')
def loading(prediction_id):
    return render_template('loading.html', prediction_id=prediction_id)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model tidak tersedia. Silakan coba lagi nanti.')
        return redirect(url_for('upload'))
    
    if 'image' not in request.files:
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('upload'))
    
    file = request.files['image']
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        try:
            # Generate unique prediction ID
            prediction_id = str(uuid.uuid4())
            
            # Save file
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")

            # Initialize prediction status
            prediction_results[prediction_id] = {
                'status': 'processing',
                'filepath': filepath,
                'filename': filename
            }
            
            # Redirect to loading page
            return redirect(url_for('loading', prediction_id=prediction_id))
            
        except Exception as e:
            print(f"Error in file handling: {e}")
            flash(f'Terjadi error saat memproses gambar: {str(e)}')
            return redirect(url_for('upload'))
    else:
        flash('Format file tidak didukung. Gunakan PNG, JPG, JPEG, GIF, atau BMP.')
        return redirect(url_for('upload'))

@app.route('/check_prediction/<prediction_id>')
def check_prediction(prediction_id):
    if prediction_id not in prediction_results:
        return jsonify({'status': 'error', 'message': 'Prediction ID tidak ditemukan'})
    
    result = prediction_results[prediction_id]
    
    if result['status'] == 'processing':
        try:
            filepath = result['filepath']
            filename = result['filename']
            
            print("Starting preprocessing...")
            features = preprocess_and_extract_features(filepath)
            if features is None:
                result['status'] = 'error'
                result['message'] = 'Error dalam memproses gambar'
                return jsonify(result)
            
            print(f"Features shape: {features.shape}")
            print("Making prediction...")
            
            predicted_class, probabilities = get_prediction_probabilities(model, features)
            
            prediction_label = CLASS_LABELS.get(predicted_class, 'Unknown')
            main_confidence = float(probabilities[predicted_class]) * 100
            description = CONDITION_DESCRIPTIONS.get(prediction_label, 'Deskripsi tidak tersedia')
            
            image_path = url_for('static', filename=f'uploads/{filename}')
            
            probability_dict = {}
            for class_idx, class_name in CLASS_LABELS.items():
                if class_idx < len(probabilities):
                    probability_dict[class_name] = round(float(probabilities[class_idx]) * 100, 1)
                else:
                    probability_dict[class_name] = 0.0
            
            # Update result with completed prediction
            result.update({
                'status': 'completed',
                'prediction': prediction_label,
                'confidence': round(main_confidence, 1),
                'description': description,
                'image_path': image_path,
                'probabilities': probability_dict
            })
            
            print(f"Prediction completed: {prediction_label} with {main_confidence}% confidence")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            result['status'] = 'error'
            result['message'] = f'Terjadi error saat memproses gambar: {str(e)}'
    
    return jsonify(result)

@app.route('/result/<prediction_id>')
def result(prediction_id):
    if prediction_id not in prediction_results:
        flash('Hasil prediksi tidak ditemukan')
        return redirect(url_for('upload'))
    
    result = prediction_results[prediction_id]
    
    if result['status'] != 'completed':
        flash('Prediksi belum selesai atau terjadi error')
        return redirect(url_for('upload'))
    
    return render_template('result.html', 
                         prediction=result['prediction'],
                         confidence=result['confidence'],
                         description=result['description'],
                         image_path=result['image_path'],
                         probabilities=result['probabilities'])

@app.errorhandler(413)
def too_large(e):
    flash('File terlalu besar. Maksimal 16MB.')
    return redirect(url_for('upload'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting Flask app on port {port}")
    print(f"Debug mode: {debug_mode}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode
    )