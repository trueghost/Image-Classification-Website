import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, url_for, send_from_directory
from werkzeug.utils import secure_filename
from keras import models

app = Flask(__name__)
model = models.load_model('image_classifier.model')

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

def preprocess_image(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (32, 32))
    img = np.array([img]) / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join('static', 'images', filename)
            file.save(image_path)

            image = cv.imread(image_path)
            if image.shape[:2] != (32, 32):
                image = cv.resize(image, (32, 32))

            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)
            index = np.argmax(prediction)
            predicted_class = class_names[index]
            return render_template('index.html', predicted_class=predicted_class, image_name=filename)

    return render_template('index.html', predicted_class=None, image_name=None)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)