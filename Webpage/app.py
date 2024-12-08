import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

app = Flask(__name__, static_folder="static")

model_image = tf.keras.models.load_model('./models/dr.h5')

model_lung_cancer = tf.keras.models.load_model('./models/dr.h5')

class_labels = {
    0: "0%",
    1: "25%",
    2: "50%",
    3: "75%",
    4: "100%"
}
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def classify_image(file_path):
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img_array = np.array(img) 
    if img_array.shape[2] == 4:  
        img_array = img_array[:, :, :3]
    img_array = img_array.astype(np.uint8)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model_image.predict(img_array)
    threshold = 0.37757874193797547
    y_test_thresholded = predictions > threshold
    y_test_binary = y_test_thresholded.astype(int)
    y_test_adjusted = y_test_binary.sum(axis=1) - 1
    predicted_class = class_labels[y_test_adjusted[0]]
    stage = y_test_adjusted[0]
    return stage, predicted_class

def classify_video(file_path):
  
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return "Error opening video file"

    total_frames = 0
    total_stage = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        resized_frame = cv2.resize(frame, (224, 224))
        img_array = np.array(resized_frame)
        img_array = img_array.astype(np.uint8)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model_image.predict(img_array)

        threshold = 0.37757874193797547
        y_test_thresholded = predictions > threshold
        y_test_binary = y_test_thresholded.astype(int)
        y_test_adjusted = y_test_binary.sum(axis=1) - 1

        total_stage += y_test_adjusted[0]

    cap.release()

    if total_frames > 0:
        average_stage = total_stage / total_frames
    else:
        average_stage = 0

    predicted_class = class_labels[int(average_stage)]

    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('result.html', error='No file part')

        file = request.files['file']
        username = request.form.get('username')
        age = request.form.get('age')

        if file.filename == '':
            return render_template('result.html', error='No selected file')

        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)


            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                stage,predicted_class = classify_image(file_path)
            else:
                return render_template('result.html', error='Unsupported file format')

            return render_template('result.html', stage = stage,predicted_class=predicted_class, username=username, age=age, filename=file.filename)

    return render_template('index.html', error=None)



if __name__ == '__main__':
    app.run(debug=True)