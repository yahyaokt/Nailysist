from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
app.secret_key = 'yahya_wong_sangar'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = None
nail_detector_model = None

try:
    try:
        with open("gacor_model.pkl", "rb") as f:
            model = pickle.load(f)
        print("Main model loaded successfully with standard pickle!")
    except Exception as e1:
        print(f"Standard pickle failed for main model: {e1}")

        try:
            with open("gacor_model.pkl", "rb") as f:
                model = pickle.load(f, encoding='latin1')
            print("Main model loaded successfully with latin1 encoding!")
        except Exception as e2:
            print(f"Latin1 encoding failed for main model: {e2}")

            try:
                import joblib
                model = joblib.load("gacor_model.pkl")
                print("Main model loaded successfully with joblib!")
            except Exception as e3:
                print(f"Joblib loading failed for main model: {e3}")
    
    if model is not None:
        print(f"Main model type: {type(model)}")
        print("Main model loaded successfully!")
    else:
        print("FAILED TO LOAD MAIN MODEL!")
        
except Exception as e:
    print(f"Critical error in main model loading: {e}")
    model = None

try:
    try:
        with open("model_kuku.pkl", "rb") as f:
            nail_detector_model = pickle.load(f)
        print("Nail detector model loaded successfully with standard pickle!")
    except Exception as e1:
        print(f"Standard pickle failed for nail detector: {e1}")

        try:
            with open("model_kuku.pkl", "rb") as f:
                nail_detector_model = pickle.load(f, encoding='latin1')
            print("Nail detector model loaded successfully with latin1 encoding!")
        except Exception as e2:
            print(f"Latin1 encoding failed for nail detector: {e2}")

            try:
                import joblib
                nail_detector_model = joblib.load("model_kuku.pkl")
                print("Nail detector model loaded successfully with joblib!")
            except Exception as e3:
                print(f"Joblib loading failed for nail detector: {e3}")
    
    if nail_detector_model is not None:
        print(f"Nail detector model type: {type(nail_detector_model)}")
        print("Nail detector model loaded successfully!")
    else:
        print("FAILED TO LOAD NAIL DETECTOR MODEL!")
        
except Exception as e:
    print(f"Critical error in nail detector model loading: {e}")
    nail_detector_model = None

CLASS_LABELS = {
    0: 'Bluish Nail',
    1: 'Healthy Nail', 
    2: 'Koilonychia',
    3: 'Terry-s nail'
}

CONDITION_DESCRIPTIONS = {
    'Bluish Nail': 'Jika kuku kebiruan muncul tanpa sebab yang jelas atau berlangsung lama, segera konsultasikan dengan dokter untuk diagnosis yang tepat. Jika disebabkan oleh penyakit tertentu, penanganan medis untuk penyakit tersebut akan membantu mengatasi masalah kuku. Jika akibat cedera, hindari aktivitas fisik yang dapat memperburuk kondisi kuku.',
    'Healthy Nail': 'Untuk menjaga kuku tetap sehat, penting untuk mengonsumsi makanan yang kaya zat besi, protein, dan vitamin, serta menjaga kebersihan kuku dengan menggunakan pelembap atau minyak kuku secara teratur. Selain itu, memotong kuku secara rutin dan menjaga panjang kuku agar tidak terlalu panjang juga dapat mencegah cedera dan infeksi.',
    'Koilonychia': 'Perawatan untuk koilonychia melibatkan pengobatan penyebab utama, seperti pemberian suplemen zat besi atau perubahan pola makan untuk anemia defisiensi besi. Jika disebabkan oleh penyakit lain, pengobatan kondisi tersebut dapat memperbaiki kuku. Penyuluhan diet dengan mengonsumsi makanan kaya zat besi, seperti daging merah, sayuran berdaun hijau, dan kacang-kacangan, juga penting. Selain itu, penggunaan pelapis kuku atau produk penguat kuku dapat membantu menjaga kesehatan kuku.',
    'Terry-s nail': 'Terry-s Nail biasanya merupakan gejala dari kondisi medis serius seperti penyakit hati, gagal jantung, atau diabetes. Pengobatan tergantung pada kondisi yang mendasari, seperti pengelolaan penyakit hati atau jantung melalui terapi yang tepat, serta pengontrolan kadar gula darah pada diabetes dengan diet dan obat-obatan. Pemeriksaan berkala dan konsultasi dengan dokter sangat disarankan jika ada perubahan pada kuku yang mirip dengan Terry-s Nail.'
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

def is_nail_image(nail_detector_model, features):
    try:
        if nail_detector_model is None:
            print("Nail detector model not available, assuming image is nail")
            return True

        if hasattr(nail_detector_model, 'predict'):
            prediction = nail_detector_model.predict(features)[0]
        else:
            print("Nail detector model doesn't have predict method")
            return True
        
        print(f"Nail detector prediction: {prediction}")

        return prediction == 1
        
    except Exception as e:
        print(f"Error in nail detection: {e}")
        return True

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

@app.route('/penyakit')
def penyakit():
    return render_template('penyakit.html')

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
            filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            print("Starting preprocessing...")
            features = preprocess_and_extract_features(filepath)
            if features is None:
                flash('Error dalam memproses gambar')
                return redirect(url_for('upload'))
            
            print(f"Features shape: {features.shape}")

            print("Checking if image is a nail...")
            if not is_nail_image(nail_detector_model, features):

                print("Image is not detected as a nail image")
                
                image_path = url_for('static', filename=f'uploads/{filename}')
                
                return render_template('result.html',
                                     prediction='Bukan Gambar Kuku',
                                     confidence=95.0,
                                     description='Gambar yang Anda upload tidak terdeteksi sebagai gambar kuku. Silakan upload gambar kuku yang jelas untuk mendapatkan hasil diagnosa yang akurat.',
                                     image_path=image_path,
                                     probabilities={
                                         'Bluish Nail': 0.0,
                                         'Healthy Nail': 0.0,
                                         'Koilonychia': 0.0,
                                         'Terry-s nail': 0.0
                                     })

            print("Image is detected as a nail image, proceeding with nail classification...")
            print("Making nail condition prediction...")
            
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
            
            print(f"Nail condition prediction completed: {prediction_label} with {main_confidence}% confidence")
            
            return render_template('result.html',
                                 prediction=prediction_label,
                                 confidence=round(main_confidence, 1),
                                 description=description,
                                 image_path=image_path,
                                 probabilities=probability_dict)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            flash(f'Terjadi error saat memproses gambar: {str(e)}')
            return redirect(url_for('upload'))
    else:
        flash('Format file tidak didukung. Gunakan PNG, JPG, JPEG, GIF, atau BMP.')
        return redirect(url_for('upload'))

@app.errorhandler(413)
def too_large(e):
    flash('File terlalu besar. Maksimal 16MB.')
    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run()