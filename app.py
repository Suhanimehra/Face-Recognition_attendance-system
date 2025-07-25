from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd
import pickle
from functools import wraps
import database
from io import BytesIO
import base64

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

# Admin credentials (for demo; use env vars or DB in production)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Admin login required.')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

def student_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('student_logged_in'):
            flash('Student login required.')
            return redirect(url_for('student_login'))
        return f(*args, **kwargs)
    return decorated_function

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def mark_attendance(name, frame=None):
    now = datetime.now()
    dt = now.strftime('%Y-%m-%d %H:%M:%S')
    # Get user from DB
    user = database.get_user_by_name(name.replace("_", " "))
    class_ = user['class'] if user else ''
    user_id = user['id'] if user else None
    img_blob = None
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', frame)
        img_blob = buffer.tobytes()
    # Insert attendance in DB
    if user_id:
        database.insert_attendance(user_id, name, dt, class_, img_blob)

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

@app.route('/front')
def front_redirect():
    return redirect(url_for('front'))

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
        class_ = request.form['class']
        # Insert user in DB (registration_image will be set after capture)
        database.insert_user(name, enrollment, branch, year, class_, email, None)
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
        # Convert frame to grayscale for training
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Save image as BLOB in database
        _, buffer = cv2.imencode('.jpg', gray)
        img_blob = buffer.tobytes()
        # Get user info from database
        user = database.get_user_by_name(username.replace("_", " "))
        if user:
            # Update registration_image for user
            db = database.get_db()
            db.execute('UPDATE users SET registration_image = ? WHERE id = ?', (img_blob, user['id']))
            db.commit()
            db.close()
        else:
            # If user not in DB, fallback to old logic (should not happen)
            pass
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
                    mark_attendance(name, frame=roi)
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
    # Return attendance records from DB
    db = database.get_db()
    records = db.execute('SELECT a.id, u.name as Name, a.time as Time, a.class as Class FROM attendance a JOIN users u ON a.user_id = u.id').fetchall()
    db.close()
    return jsonify([dict(r) for r in records])

@app.route('/back_to_register')
def back_to_register():
    return redirect(url_for('register'))

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            flash('Logged in as admin.')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials.')
            return redirect(url_for('admin_login'))
    return render_template('admin_login.html')

@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin_logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    flash('Logged out successfully.')
    return redirect(url_for('admin_login'))

@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        enrollment = request.form['enrollment']
        email = request.form['email']
        user = database.get_user_by_enrollment(enrollment)
        if user and user['email'] == email:
            session['student_logged_in'] = True
            session['student_enrollment'] = enrollment
            session['student_name'] = user['name']
            flash('Logged in as student.')
            return redirect(url_for('student_dashboard'))
        flash('Invalid student credentials.')
        return redirect(url_for('student_login'))
    return render_template('student_login.html')

@app.route('/student_portal')
def student_portal():
    return render_template('student_portal.html')

@app.route('/student_dashboard')
@student_required
def student_dashboard():
    user = database.get_user_by_enrollment(session.get('student_enrollment'))
    attendance_records = []
    if user:
        attendance_records = database.get_attendance_by_user_id(user['id'])
    return render_template('student_dashboard.html', name=user['name'] if user else '', records=attendance_records)

@app.route('/student_logout')
def student_logout():
    session.pop('student_logged_in', None)
    session.pop('student_enrollment', None)
    session.pop('student_name', None)
    flash('Logged out successfully.')
    return redirect(url_for('student_login'))

if __name__ == '__main__':
    database.init_db()
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
