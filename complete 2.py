import cv2
import os
import json
import re
import threading
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, render_template
from ultralytics import YOLO
import face_recognition # type: ignore
import dlib # type: ignore
from playsound import playsound  # type: ignore # For sound alerts
import datetime
import sqlite3
import smtplib
from email.mime.text import MIMEText
import psutil  # For system health monitoring

# Define dataset directory
dataset_dir = "dataset"
frames_folder = os.path.join(dataset_dir, "frames")
data_dict_path = os.path.join(dataset_dir, "dataset.json")
detections_path = "backend/detections.json"
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(os.path.dirname(detections_path), exist_ok=True)

# Database setup
DATABASE = "proctoring.db"

def init_db():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                frame_path TEXT,
                label TEXT,
                confidence REAL
            )
        """)
        conn.commit()

init_db()

# Registered students for verification
def get_registered_students():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM students")
        return {row[0]: {"name": row[1], "email": row[2]} for row in cursor.fetchall()}

REGISTERED_STUDENTS = get_registered_students()

# Load EfficientDet model instead of YOLO
from effdet import get_efficientdet_config, EfficientDet
from effdet import DetBenchPredict
import torch

def load_efficientdet_model():
    config = get_efficientdet_config('tf_efficientdet_d0')
    net = EfficientDet(config)
    checkpoint = torch.load('models/efficientdet-d0.pth')
    net.load_state_dict(checkpoint)
    net = DetBenchPredict(net)
    net.eval()
    return net

EFFICIENTDET_MODEL = load_efficientdet_model()

# Load face detector and landmark predictor
LANDMARK_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(LANDMARK_PREDICTOR_PATH):
    raise FileNotFoundError("Landmark predictor file not found!")

FACE_DETECTOR = dlib.get_frontal_face_detector()
LANDMARK_PREDICTOR = dlib.shape_predictor(LANDMARK_PREDICTOR_PATH)

# Cache known face encodings
KNOWN_FACE_ENCODINGS = []

# Load known face encodings from the dataset
def load_known_faces():
    global KNOWN_FACE_ENCODINGS
    known_faces_dir = "dataset/faces"
    KNOWN_FACE_ENCODINGS = []  # Reset known encodings

    if not os.path.exists(known_faces_dir):
        os.makedirs(known_faces_dir)

    for file in os.listdir(known_faces_dir):
        if file.endswith(("jpg", "png")):
            img_path = os.path.join(known_faces_dir, file)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                KNOWN_FACE_ENCODINGS.append(encoding[0])

load_known_faces()

# Alert sound file
ALERT_SOUND = "alert.wav"

# Email configuration
EMAIL_HOST = "smtp.example.com"
EMAIL_PORT = 587
EMAIL_USER = "admin@example.com"
EMAIL_PASSWORD = "password"
ADMIN_EMAIL = "admin@example.com"

# Configurable alert threshold for cheating detection
CHEATING_THRESHOLD = 0.7

# Function to authenticate face
def authenticate_face(image):
    try:
        unknown_face_encodings = face_recognition.face_encodings(image)
        if not unknown_face_encodings:
            print("No face encoding found in the frame.")
            return False

        for known_face in KNOWN_FACE_ENCODINGS:
            if face_recognition.compare_faces([known_face], unknown_face_encodings[0])[0]:
                return True
        return False
    except Exception as e:
        print(f"Face authentication error: {e}")
        return False

# Function to detect gaze
def detect_gaze(landmarks):
    try:
        if landmarks.num_parts != 68:
            return "Unknown"

        left_eye = landmarks.part(36), landmarks.part(39)
        right_eye = landmarks.part(42), landmarks.part(45)

        if abs(left_eye[0].x - left_eye[1].x) > abs(right_eye[0].x - right_eye[1].x):
            return "Looking Left"
        elif abs(right_eye[0].x - right_eye[1].x) > abs(left_eye[0].x - left_eye[1].x):
            return "Looking Right"
        return "Looking Forward"
    except Exception as e:
        print(f"Gaze detection error: {e}")
        return "Unknown"

# Function to log events
def log_event(event):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO logs (timestamp, event) VALUES (?, ?)", (timestamp, event))
        conn.commit()

# Function to send email alerts
def send_email_alert(subject, message):
    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_USER
        msg["To"] = ADMIN_EMAIL

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, ADMIN_EMAIL, msg.as_string())
    except Exception as e:
        print(f"Email alert error: {e}")

# Function to perform real-time proctoring using EfficientDet
def real_time_proctoring():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face recognition and gaze detection
        face_locations = face_recognition.face_locations(rgb_frame)
        for top, right, bottom, left in face_locations:
            face_landmarks = LANDMARK_PREDICTOR(rgb_frame, dlib.rectangle(left, top, right, bottom))
            gaze_direction = detect_gaze(face_landmarks)
            cv2.putText(frame, f"Gaze: {gaze_direction}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # EfficientDet detection
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).float().unsqueeze(0) / 255
        with torch.no_grad():
            result = EFFICIENTDET_MODEL(frame_tensor)
            print(f"Detection: {result}")

        cv2.imshow("Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__":
    threading.Thread(target=real_time_proctoring).start()
    app.run(host="0.0.0.0", port=5000)
