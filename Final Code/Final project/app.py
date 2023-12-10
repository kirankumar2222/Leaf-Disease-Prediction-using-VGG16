UPLOAD_FOLDER = 'uploads/'
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
from matplotlib.image import imread
from flask_cors import CORS, cross_origin

app = Flask(__name__,static_url_path="/static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = keras.models.load_model('Mymodel.h5')

categories=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

cors = CORS(app, resources={r"/train": {"origins": "http://localhost/home"}})

import os
import urllib.request
from flask import Flask, request, redirect, jsonify, render_template
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        return redirect(url_for('predict', filename=filename))

@app.route('/predict/<filename>')
def predict(filename):
    file_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = imread(file_path)
    resImage = resize(img, (224, 224, 3))
    prob = model.predict(np.array([resImage]))
    category_index = np.argmax(prob)
    predicted_category = categories[category_index]

    return render_template('result.html', filename=filename, predicted_category=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)
