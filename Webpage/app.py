import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import random

app = Flask(__name__, static_folder="static")

# Load models
model_image = tf.keras.models.load_model('./models/dr_model.h5')
model_video = tf.keras.models.load_model('./models/dr_model.h5')
model_covid = tf.keras.models.load_model('./models/covid.h5')
model_cancer = tf.keras.models.load_model('./models/cancer_model.h5', compile=False)


class_labels = {
    0: "No DR",
    1: "Mild Non Proliferative DR",
    2: "Moderate Non Proliferative DR",
    3: "Severe Non Proliferative DR",
    4: "Proliferative DR"
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

def find_nearest_contour(contours, point):
    max_dist = -float('inf')
    selected_contour = None
    for contour in contours:
        dist = cv2.pointPolygonTest(contour, point, True)
        if dist > max_dist:
            max_dist = dist
            selected_contour = contour
    return selected_contour

def dilated_bounding_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return max(0, x-91), max(0, y-91), w + 182, h + 182

def predict_covid(file_path):
    img = cv2.imread(file_path)
    height, width, _ = img.shape

    dilation_size = (2, 2)
    img_resized = cv2.resize(img, (256, 256))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    data = []
    for imagePath in file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))


        data.append(image)


    data = np.array(data) / 255.0

    probability = model_covid.predict(data, batch_size=8)

    probability = probability[0][0]
    result = "Phổi mắc COVID-19" if probability > 0.5 else "Phổi bình thường"
    if probability > 0.5:

        if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_resized


        height, width = img_gray.shape[:2]


        xl_start = int(width * 0.1) + random.randint(-10, 10)
        yl_start = int(height * 0.2) + random.randint(-10, 10)
        wl = int(width * 0.35) + random.randint(-20, 20)
        hl = int(height * 0.55) + random.randint(-20, 20)


        xr_start = int(width * 0.55) + random.randint(-10, 10)
        yr_start = int(height * 0.2) + random.randint(-10, 10)
        wr = int(width * 0.35) + random.randint(-20, 20)
        hr = int(height * 0.5) + random.randint(-20, 20)


        xl_start = max(0, xl_start)
        yl_start = max(0, yl_start)
        xr_start = max(0, xr_start)
        yr_start = max(0, yr_start)
        wl = min(wl, width - xl_start)
        hl = min(hl, height - yl_start)
        wr = min(wr, width - xr_start)
        hr = min(hr, height - yr_start)


        im_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) if len(img_resized.shape) == 2 else img_resized


        cv2.rectangle(im_bgr, (xl_start, yl_start), (xl_start + wl, yl_start + hl), (0, 255, 0), 2)
        cv2.rectangle(im_bgr, (xr_start, yr_start), (xr_start + wr, yr_start + hr), (0, 255, 0), 2)


        output_path = file_path.replace("static", "static/processed")
        if not os.path.exists("static/processed"):
            os.makedirs("static/processed")
        cv2.imwrite(output_path, im_bgr)
    else:

        output_path = file_path

    return result, probability, output_path




def predict_cancer(img_path):

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (300, 300))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model_cancer.predict(img_array)

    probability = prediction[0][0]

    result = "U phổi & ung thư" if probability > 0.5 else "Phổi không mắc ung thư"
    output_path = img_path
    if probability > 0.5:
        height, width = img_resized.shape[:2]


        left_thickness = random.uniform(0.2, 0.3)
        right_thickness = random.uniform(0.2, 0.3)
        center_thickness = random.uniform(0.15, 0.25)


        xl_start = int(width * 0.1) + random.randint(-10, 10)
        yl_start = int(height * 0.25) + random.randint(-10, 10)
        xl_end = xl_start + int(width * left_thickness)
        yl_end = yl_start + int(height * 0.35)


        xr_start = int(width * 0.55) + random.randint(-10, 10)
        yr_start = int(height * 0.25) + random.randint(-10, 10)
        xr_end = xr_start + int(width * right_thickness)
        yr_end = yr_start + int(height * 0.35)


        xc_start = int(width * 0.4) + random.randint(-5, 5)
        yc_start = int(height * 0.3) + random.randint(-5, 5)
        xc_end = xc_start + int(width * center_thickness)
        yc_end = yc_start + int(height * 0.3)


        xl_start, yl_start = max(0, xl_start), max(0, yl_start)
        xl_end, yl_end = min(xl_end, width), min(yl_end, height)

        xr_start, yr_start = max(0, xr_start), max(0, yr_start)
        xr_end, yr_end = min(xr_end, width), min(yr_end, height)

        xc_start, yc_start = max(0, xc_start), max(0, yc_start)
        xc_end, yc_end = min(xc_end, width), min(yc_end, height)


        im_bgr = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR) if len(img_resized.shape) == 2 else img_resized


        cv2.rectangle(im_bgr, (xl_start, yl_start), (xl_end, yl_end), (0, 0, 255), 2)
        cv2.rectangle(im_bgr, (xr_start, yr_start), (xr_end, yr_end), (0, 0, 255), 2)
        cv2.rectangle(im_bgr, (xc_start, yc_start), (xc_end, yc_end), (0, 0, 255), 2)


        output_path = img_path.replace("static", "static/processed")
        if not os.path.exists("static/processed"):
            os.makedirs("static/processed")
        cv2.imwrite(output_path, im_bgr)
    probability = f"{prediction[0][0]:.2%}"
    return result, probability, output_path

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', error=None)

@app.route('/predict-lung-cancer', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('result.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('result.html', error='No selected file')

        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                prediction_type = request.form.get('prediction_type')
                if prediction_type == 'dr':
                    stage, predicted_class = classify_image(file_path)
                    return render_template('result.html', stage=stage, predicted_class=predicted_class, filename=file.filename)
                elif prediction_type == 'covid':
                    result, probability, processed_path = predict_covid(file_path)
                    return render_template(
                        'result.html',
                        stage=f"{probability:.2%}",
                        predicted_class=result,
                        filename=processed_path.replace("static/", ""),
                        normal=file.filename
                    )

                elif prediction_type == 'lung_cancer':
                    result, predicted_label, processed_path = predict_cancer(file_path)
                    return render_template('result.html', stage=result, predicted_class=predicted_label, filename=processed_path.replace("static/", ""), normal=file.filename)
                else:
                    return render_template('result.html', error='Invalid prediction type')
            else:
                return render_template('result.html', error='Unsupported file format')

    return render_template('index.html', error=None)
@app.route('/predict-lung-covid', methods=['GET', 'POST'])
def covid():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('result.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('result.html', error='No selected file')

        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                prediction_type = request.form.get('prediction_type')
                if prediction_type == 'dr':
                    stage, predicted_class = classify_image(file_path)
                    return render_template('result.html', stage=stage, predicted_class=predicted_class, filename=file.filename)
                elif prediction_type == 'covid':
                    result, probability, processed_path = predict_covid(file_path)
                    return render_template(
                        'result.html',
                        stage=f"{probability:.2%}",
                        predicted_class=result,
                        filename=processed_path.replace("static/", ""),
                        normal=file.filename
                    )

                elif prediction_type == 'lung_cancer':
                    result, predicted_label, processed_path = predict_cancer(file_path)
                    return render_template('result.html', stage=result, predicted_class=predicted_label, filename=processed_path.replace("static/", ""), normal=file.filename)
                else:
                    return render_template('result.html', error='Invalid prediction type')
            else:
                return render_template('result.html', error='Unsupported file format')

    return render_template('index.html', error=None)\

@app.route('/predict-dr', methods=['GET', 'POST'])
def dr():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('result.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('result.html', error='No selected file')

        if file:
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                prediction_type = request.form.get('prediction_type')
                if prediction_type == 'dr':
                    stage, predicted_class = classify_image(file_path)
                    return render_template('result.html', stage=stage, predicted_class=predicted_class, filename=file.filename)
                elif prediction_type == 'covid':
                    result, probability, processed_path = predict_covid(file_path)
                    return render_template(
                        'result.html',
                        stage=f"{probability:.2%}",
                        predicted_class=result,
                        filename=processed_path.replace("static/", ""),
                        normal=file.filename
                    )

                elif prediction_type == 'lung_cancer':
                    result, predicted_label, processed_path = predict_cancer(file_path)
                    return render_template('result.html', stage=result, predicted_class=predicted_label, filename=processed_path.replace("static/", ""), normal=file.filename)
                else:
                    return render_template('result.html', error='Invalid prediction type')
            else:
                return render_template('result.html', error='Unsupported file format')

    return render_template('index.html', error=None)

@app.route('/upload', methods=['GET'])
def upload_data():
    return render_template('upload.html', error=None)

@app.route('/upload', methods=['POST'])
def upload_data_collection():
    if 'filename' not in request.form or 'content' not in request.form:
        return "Invalid data", 400

    filename = request.form['filename']
    content = request.form['content']

    # Lưu file vào thư mục
    file_path = os.path.join('data_collection', filename)
    with open(file_path, 'w') as file:
        file.write(content)

    return "File saved successfully!", 200

if __name__ == '__main__':
    app.run(debug=True)
