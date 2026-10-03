import os
import sqlite3
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Cascades
CASCADE_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haarcascade_eye.xml'))
if eye_cascade.empty():
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Global State
camera = None
camera_active = True
app_mode = 'ATTENDANCE'  # 'ATTENDANCE' or 'REGISTRATION'
last_marked_person = None

# Registration State Tracking
REG_STEPS = ['CENTER', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'BLINK']
reg_step_index = 0
reg_captured_samples = []  # list of grayscale cropped faces
reg_status_msg = "Look straight at the camera"
blink_counter = 0
eyes_seen_previously = False


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def mark_attendance(name):
    global camera_active, last_marked_person
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    cursor.execute('SELECT * FROM attendance WHERE name = ? AND date = ?', (name, date_str))
    result = cursor.fetchone()

    if not result:
        cursor.execute('INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)', (name, date_str, time_str))
        conn.commit()
        print(f'[LOG] Attendance marked for {name}')

    conn.close()
    last_marked_person = name
    camera_active = False


def load_known_faces():
    known_names = []
    face_samples = []
    labels = []

    if not os.path.isdir(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)
        return None, []

    for filename in sorted(os.listdir(DATASET_DIR)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(DATASET_DIR, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        detected = face_cascade.detectMultiScale(image, 1.1, 5, minSize=(40, 40))
        if len(detected) == 0:
            continue

        x, y, w, h = detected[0]
        face_img = cv2.resize(image[y:y + h, x:x + w], (200, 200))

        # Assumes format: Name_1.jpg or Name.jpg
        raw_name = os.path.splitext(filename)[0]
        clean_name = raw_name.rsplit('_', 1)[0] if '_' in raw_name and raw_name.rsplit('_', 1)[1].isdigit() else raw_name
        clean_name = clean_name.replace('_', ' ').title()

        if clean_name not in known_names:
            known_names.append(clean_name)

        label_id = known_names.index(clean_name)
        face_samples.append(face_img)
        labels.append(label_id)

    if not face_samples:
        return None, []

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(np.array(face_samples), np.array(labels, dtype=np.int32))
    return recognizer, known_names


face_recognizer, known_names = load_known_faces()


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera


def release_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None


def generate_frames():
    global camera_active, app_mode, reg_step_index, reg_status_msg, reg_captured_samples
    global blink_counter, eyes_seen_previously
    cam = get_camera()

    while camera_active:
        success, frame = cam.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Mirror feed for natural interaction
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if app_mode == 'ATTENDANCE':
            detected_name = None
            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                name = 'Unknown'

                if face_recognizer is not None:
                    label, confidence = face_recognizer.predict(face_img)
                    if label < len(known_names) and confidence < 80:
                        name = known_names[label]
                        detected_name = name

                color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            if detected_name:
                mark_attendance(detected_name)
                time.sleep(0.5)
                break

        elif app_mode == 'REGISTRATION':
            instruction = ""
            if reg_step_index < len(REG_STEPS):
                current_step = REG_STEPS[reg_step_index]
            else:
                current_step = 'DONE'

            if current_step == 'CENTER':
                instruction = "1/5: Look Straight Center"
            elif current_step == 'LEFT':
                instruction = "2/5: Turn Face to the Left"
            elif current_step == 'RIGHT':
                instruction = "3/5: Turn Face to the Right"
            elif current_step == 'UP':
                instruction = "4/5: Tilt Face Upward"
            elif current_step == 'DOWN':
                instruction = "5/5: Tilt Face Downward"
            elif current_step == 'BLINK':
                instruction = "Bonus: Blink Your Eyes"
            elif current_step == 'DONE':
                instruction = "All Captured! Enter name on dashboard."

            reg_status_msg = instruction

            if len(faces) > 0 and current_step != 'DONE':
                x, y, w, h = faces[0]
                face_center_x = x + w / 2
                face_center_y = y + h / 2
                frame_center_x = w_frame / 2
                frame_center_y = h_frame / 2

                roi_gray = gray[y:y + h, x:x + w]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

                # Pose and Action Verification Logic
                step_passed = False

                if current_step == 'CENTER':
                    if abs(face_center_x - frame_center_x) < 80 and abs(face_center_y - frame_center_y) < 80:
                        step_passed = True

                elif current_step == 'LEFT':
                    # With mirror flip, turning left moves bounding box leftwards
                    if face_center_x < (frame_center_x - 45):
                        step_passed = True

                elif current_step == 'RIGHT':
                    if face_center_x > (frame_center_x + 45):
                        step_passed = True

                elif current_step == 'UP':
                    if face_center_y < (frame_center_y - 35):
                        step_passed = True

                elif current_step == 'DOWN':
                    if face_center_y > (frame_center_y + 35):
                        step_passed = True

                elif current_step == 'BLINK':
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=4)
                    if len(eyes) >= 1:
                        eyes_seen_previously = True
                    elif eyes_seen_previously and len(eyes) == 0:
                        # Eyes disappeared after being visible -> Blink
                        blink_counter += 1
                        eyes_seen_previously = False
                        if blink_counter >= 1:
                            step_passed = True

                if step_passed:
                    face_crop = cv2.resize(roi_gray, (200, 200))
                    reg_captured_samples.append(face_crop)
                    reg_step_index += 1
                    time.sleep(0.4)  # brief pause before next step

            # Overlay Status banner
            cv2.rectangle(frame, (0, 0), (w_frame, 50), (15, 23, 42), cv2.FILLED)
            cv2.putText(frame, instruction, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (56, 189, 248), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    release_camera()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, last_marked_person, app_mode
    app_mode = 'ATTENDANCE'
    camera_active = True
    last_marked_person = None
    return jsonify({'status': 'camera_started', 'mode': app_mode})


@app.route('/start_registration', methods=['POST'])
def start_registration():
    global camera_active, app_mode, reg_step_index, reg_captured_samples, blink_counter, eyes_seen_previously
    app_mode = 'REGISTRATION'
    camera_active = True
    reg_step_index = 0
    reg_captured_samples = []
    blink_counter = 0
    eyes_seen_previously = False
    return jsonify({'status': 'registration_started'})


@app.route('/registration_status')
def registration_status():
    return jsonify({
        'step_index': reg_step_index,
        'total_steps': len(REG_STEPS),
        'message': reg_status_msg,
        'completed': reg_step_index >= len(REG_STEPS)
    })


@app.route('/complete_registration', methods=['POST'])
def complete_registration():
    global app_mode, camera_active, face_recognizer, known_names, reg_captured_samples
    data = request.get_json() or {}
    name = data.get('name', '').strip()

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    if not reg_captured_samples:
        return jsonify({'error': 'No face samples captured'}), 400

    clean_filename = name.replace(' ', '_').lower()
    for idx, sample in enumerate(reg_captured_samples):
        file_path = os.path.join(DATASET_DIR, f"{clean_filename}_{idx + 1}.jpg")
        cv2.imwrite(file_path, sample)

    # Retrain recognizer in memory with the new images
    face_recognizer, known_names = load_known_faces()

    # Reset back to attendance mode
    app_mode = 'ATTENDANCE'
    reg_captured_samples = []
    camera_active = False

    return jsonify({'status': 'success', 'name': name})


@app.route('/camera_status')
def camera_status():
    return jsonify({
        'active': camera_active,
        'mode': app_mode,
        'last_person': last_marked_person
    })


@app.route('/records')
def records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM attendance ORDER BY id DESC')
    rows = cursor.fetchall()
    conn.close()

    return jsonify([
        {'id': row[0], 'name': row[1], 'date': row[2], 'time': row[3]}
        for row in rows
    ])


if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)


