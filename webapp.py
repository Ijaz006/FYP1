from flask import Flask, render_template, request, url_for, Response, send_from_directory, session
import os
import cv2
import time
import torch
from werkzeug.utils import secure_filename
from ultralytics import YOLO  

import os

RESULT_FOLDER = "static/results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session usage

# # ✅ Load your trained YOLO model (change path if needed)
model = YOLO('best_246.pt')

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
RESULTS_FOLDER = os.path.join(os.getcwd(), 'runs', 'detect')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        result_path = os.path.join(RESULT_FOLDER, filename)

        file.save(file_path)  # Save uploaded file

        # Run prediction
        results = model(file_path)

        # Save result image
        results[0].save(result_path)

        # Generate URL for displaying image
        image_url = url_for("static", filename=f"results/{filename}")

        return render_template("index.html", image_path=image_url)

    return render_template("index.html", image_path=None)

@app.route("/download/<filename>")
def download_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
def get_frame():
    uploaded_filename = session.get('uploaded_filename')
    if not uploaded_filename:
        return 
    
    subfolders = [f for f in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, f))]
    if subfolders:
        latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(RESULTS_FOLDER, x)))
        video_path = os.path.join(RESULTS_FOLDER, latest_subfolder, uploaded_filename)
    else:
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_filename)

    video = cv2.VideoCapture(video_path)
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)


def generate_frames():
    while True:
        success, frame = camera.read()  # Assuming you have a camera object
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# import os
# import cv2
# import uuid
# import torch
# from flask import Flask, request, render_template, send_from_directory, Response, jsonify
# from ultralytics import YOLO  

# app = Flask(__name__)

# # Define directories
# UPLOAD_FOLDER = 'static/uploads'
# PROCESSED_FOLDER = 'static/processed'

# # Ensure directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# # ✅ Load your trained YOLO model (change path if needed)
# model = YOLO('best_246.pt')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # Generate unique filename
#     file_extension = file.filename.split('.')[-1]
#     unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
#     file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
#     file.save(file_path)

#     # Run YOLO model on image
#     results = model(file_path)

#     # Generate a unique filename for the processed image
#     processed_filename = f"{uuid.uuid4().hex}.jpg"
#     processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

#     # ✅ Save detected image properly
#     result_image = results[0].plot()  # Convert result to image
#     cv2.imwrite(processed_path, result_image)  # Save processed image

#     return jsonify({"processed_image": processed_filename})

# def generate_frames():
#     camera = cv2.VideoCapture(0)
#     try:
#         while True:
#             success, frame = camera.read()
#             if not success:
#                 break
#             else:
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#     finally:
#         camera.release()  # ✅ Release camera properly

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/display/<filename>')
# def display(filename):
#     return send_from_directory(UPLOAD_FOLDER, filename)

# @app.route('/processed/<filename>')
# def processed_image(filename):
#     return send_from_directory(PROCESSED_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)
