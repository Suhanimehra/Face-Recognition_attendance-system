from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize face detection and recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load existing model if available
if os.path.exists("trainer.yml"):
    recognizer.read("trainer.yml")

# Load or initialize labels
label_map = {}
if os.path.exists("labels.pickle"):
    with open("labels.pickle", "rb") as f:
        label_map = pickle.load(f)

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def mark_attendance(name):
    now = datetime.now()
    dt = now.strftime('%Y-%m-%d %H:%M:%S')
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        if name not in df["Name"].values:
            with open("attendance.csv", "a") as f:
                f.write(f"{name},{dt}\n")
    else:
        with open("attendance.csv", "w") as f:
            f.write("Name,Time\n")
            f.write(f"{name},{dt}\n")

def train_model():
    faces = []
    labels = []
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        for file in files:
            if allowed_file(file):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in label_map:
                    label_map[label] = len(label_map)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces_detected = face_cascade.detectMultiScale(img, 1.1, 5)
                for (x, y, w, h) in faces_detected:
                    roi = img[y:y+h, x:x+w]
                    faces.append(roi)
                    labels.append(label_map[label])
    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_map, f)

# Routes
@app.route('/')
def front():
    return render_template('front.html')

@app.route('/verify_user', methods=['POST'])
def verify_user():
    username = request.form['username'].replace(" ", "_")
    if username in label_map:
        session['verified_user'] = username
        return redirect(url_for('attendance'))
    else:
        flash('User not found. Please register first.')
        return redirect(url_for('front'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        enrollment = request.form['enrollment']
        branch = request.form['branch']
        year = request.form['year']
        email = request.form['email']

        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], name.replace(" ", "_"))
        os.makedirs(user_folder, exist_ok=True)

        user_data = {
            'Name': name,
            'Enrollment': enrollment,
            'Branch': branch,
            'Year': year,
            'Email': email
        }

        if not os.path.exists("users.csv"):
            pd.DataFrame([user_data]).to_csv("users.csv", index=False)
        else:
            df = pd.read_csv("users.csv")
            pd.concat([df, pd.DataFrame([user_data])]).to_csv("users.csv", index=False)

        return redirect(url_for('capture', username=name))

    name = request.args.get('name', '')
    enrollment = request.args.get('enrollment', '')
    return render_template('register.html', name=name, enrollment=enrollment)

@app.route('/capture/<username>')
def capture(username):
    return render_template('capture.html', username=username)

@app.route('/video_feed/<username>')
def video_feed(username):
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image/<username>')
def save_image(username):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        user_folder = os.path.join(app.config['UPLOAD_FOLDER'], username.replace(" ", "_"))
        img_count = len(os.listdir(user_folder))
        cv2.imwrite(f"{user_folder}/{img_count+1}.jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        train_model()
    cap.release()
    return redirect(url_for('attendance'))

@app.route('/attendance')
def attendance():
    if 'verified_user' not in session:
        flash('Please verify your identity first')
        return redirect(url_for('front'))
    return render_template('attendance.html', username=session['verified_user'])

@app.route('/attendance_feed')
def attendance_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                id_, conf = recognizer.predict(roi)
                if conf < 80:
                    name = next((k for k, v in label_map.items() if v == id_), "Unknown")
                    mark_attendance(name)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_attendance')
def get_attendance():
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv")
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])

@app.route('/back_to_register')
def back_to_register():
    return redirect(url_for('register'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
