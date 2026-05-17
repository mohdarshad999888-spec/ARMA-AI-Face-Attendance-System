from flask import Flask, render_template, request, redirect, session, jsonify
import firebase_config
from firebase_admin import db

import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import base64
from datetime import datetime
import re
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "secret123"

# dataset folder (NEW ADD)
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# 🔥 FACE CASCADE LOAD
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ==============================
# 🔐 Admin Credentials
# ==============================
ADMIN_EMAIL = "admin@gmail.com"
ADMIN_PASS = "1234"

STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def normalize_student_id(student_id):
    return str(student_id or "").strip()


def is_valid_student_id(student_id):
    return bool(STUDENT_ID_PATTERN.fullmatch(student_id))


def student_dataset_has_images(student_id):
    folder = os.path.join("dataset", student_id)

    if not os.path.isdir(folder):
        return False

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return True

    return False


def student_registered_in_firebase(student_id):
    return db.reference(f"registered_students/{student_id}").get() is not None


def student_id_exists(student_id):
    return student_registered_in_firebase(student_id) or student_dataset_has_images(student_id)


def duplicate_response():
    return jsonify({"success": False, "message": "ID already registered"}), 409


def validation_error_response(message, status=400):
    return jsonify({"success": False, "message": message}), status


def firebase_error_response():
    return jsonify({
        "success": False,
        "message": "Unable to validate ID right now. Please check Firebase connection."
    }), 503

# ==============================
# 🧠 ML MODEL TRAINING
# ==============================
def train_model():
    X = []
    y = []

    if not os.path.exists("dataset"):
        return None

    for person in os.listdir("dataset"):
        person_path = os.path.join("dataset", person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (100, 100))
            X.append(img.flatten())
            y.append(person)

    if len(X) == 0:
        return None

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    print("Model Trained Successfully")
    return model


# 🔥 Load model
model = train_model()

# ==============================
# 🌐 ROUTES
# ==============================

@app.route('/')
def home():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    if email == ADMIN_EMAIL and password == ADMIN_PASS:
        session['admin'] = True
        return redirect('/dashboard')
    return "Invalid Login"


@app.route('/dashboard')
def dashboard():
    if 'admin' in session:
        return render_template('dashboard.html')
    return redirect('/')


@app.route('/register')
def register():
    if 'admin' in session:
        return render_template('register.html')
    return redirect('/')


@app.route('/attendance')
def attendance():
    if 'admin' in session:
        return render_template('attendance.html')
    return redirect('/')


@app.route('/report')
def report():
    if 'admin' in session:
        return render_template('report.html')
    return redirect('/')


@app.route('/logout')
def logout():
    session.pop('admin', None)
    return redirect('/')


# ==============================
# 🔥 Firebase Test
# ==============================
@app.route('/test_firebase')
def test_firebase():
    ref = db.reference('test')
    ref.set({"msg": "Connected"})
    return "Firebase Connected"


# ==============================
# 👤 REGISTER (UPLOAD)
# ==============================
@app.route('/check_student_id')
def check_student_id():
    if 'admin' not in session:
        return validation_error_response("Unauthorized", 401)

    student_id = normalize_student_id(request.args.get("id"))

    if not student_id:
        return validation_error_response("Student ID is required")

    if not is_valid_student_id(student_id):
        return validation_error_response("Invalid Student ID")

    try:
        exists = student_id_exists(student_id)
    except Exception as exc:
        print(f"Firebase ID check failed: {exc}")
        return firebase_error_response()

    if exists:
        return duplicate_response()

    return jsonify({"success": True, "exists": False, "message": "ID available"})


@app.route('/save_student', methods=['POST'])
def save_student():
    if 'admin' not in session:
        return validation_error_response("Unauthorized", 401)

    student_id = normalize_student_id(request.form.get('student_id'))
    file = request.files.get('image')

    if not student_id:
        return validation_error_response("Student ID is required")

    if not is_valid_student_id(student_id):
        return validation_error_response("Invalid Student ID")

    if not file or not file.filename:
        return validation_error_response("Image is required")

    try:
        if student_id_exists(student_id):
            return duplicate_response()

        db.reference(f"registered_students/{student_id}").set({
            "id": student_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "upload"
        })
    except Exception as exc:
        print(f"Firebase registration validation failed: {exc}")
        return firebase_error_response()

    folder = os.path.join("dataset", student_id)
    os.makedirs(folder, exist_ok=True)

    filename = secure_filename(file.filename)
    file_path = os.path.join(folder, filename)
    file.save(file_path)

    global model
    model = train_model()

    return f"Student Registered ID: {student_id}"


# ==============================
# 📸 REGISTER (CAMERA)
# ==============================
@app.route('/save_student_camera', methods=['POST'])
def save_student_camera():
    if 'admin' not in session:
        return validation_error_response("Unauthorized", 401)

    data = request.get_json(silent=True) or {}

    student_id = normalize_student_id(data.get('id'))
    image_data = data.get('image')

    try:
        count = int(data.get('count', 0))
    except (TypeError, ValueError):
        return validation_error_response("Invalid sample count")

    if not student_id:
        return validation_error_response("Student ID is required")

    if not is_valid_student_id(student_id):
        return validation_error_response("Invalid Student ID")

    if not image_data:
        return validation_error_response("Image is required")

    active_registration_id = session.get("registration_id")

    try:
        if count == 0:
            if student_id_exists(student_id):
                return duplicate_response()
        elif active_registration_id != student_id:
            if student_id_exists(student_id):
                return duplicate_response()
            return validation_error_response("Please start registration again.", 409)
    except Exception as exc:
        print(f"Firebase registration validation failed: {exc}")
        return firebase_error_response()

    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(image_data)
    except Exception:
        return validation_error_response("Invalid image data")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return validation_error_response("Invalid image data")

    folder = os.path.join("dataset", student_id)
    os.makedirs(folder, exist_ok=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    if count == 0:
        try:
            if student_id_exists(student_id):
                return duplicate_response()

            db.reference(f"registered_students/{student_id}").set({
                "id": student_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "camera",
                "status": "pending"
            })
            session["registration_id"] = student_id
        except Exception as exc:
            print(f"Firebase registration reservation failed: {exc}")
            return firebase_error_response()

    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]

    face_img = cv2.resize(face_img, (100, 100))

    file_path = os.path.join(folder, f"img_{count}.jpg")
    if not cv2.imwrite(file_path, face_img):
        if count == 0:
            try:
                db.reference(f"registered_students/{student_id}").delete()
                session.pop("registration_id", None)
            except Exception as exc:
                print(f"Firebase cleanup after image save failure failed: {exc}")

        return validation_error_response("Unable to save face image.", 500)

    return "Saved"


# ==============================
# 🔥 FINALIZE
# ==============================
@app.route('/finalize_registration', methods=['POST'])
def finalize_registration():
    if 'admin' not in session:
        return validation_error_response("Unauthorized", 401)

    data = request.get_json(silent=True) or {}
    student_id = normalize_student_id(data.get("id") or session.get("registration_id"))

    if not student_id:
        return validation_error_response("Student ID is required")

    if not is_valid_student_id(student_id):
        return validation_error_response("Invalid Student ID")

    if session.get("registration_id") != student_id:
        return validation_error_response("Please start registration again.", 409)

    folder = os.path.join("dataset", student_id)
    saved_images = []

    if os.path.isdir(folder):
        saved_images = [
            filename for filename in os.listdir(folder)
            if filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if len(saved_images) < 5:
        return validation_error_response("Please save 5 face samples before finalizing.", 400)

    try:
        registration_ref = db.reference(f"registered_students/{student_id}")
        existing_registration = registration_ref.get()

        if (
            existing_registration
            and (
                not isinstance(existing_registration, dict)
                or existing_registration.get("status") != "pending"
            )
        ):
            session.pop("registration_id", None)
            return duplicate_response()

        registration_ref.set({
            "id": student_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_count": len(saved_images),
            "source": "camera",
            "status": "registered"
        })
    except Exception as exc:
        print(f"Firebase finalize registration failed: {exc}")
        return firebase_error_response()

    global model
    model = train_model()

    session.pop("registration_id", None)

    print("Model Updated after registration")
    return "Registration Complete"


# ==============================
# 📸 FACE ATTENDANCE
# ==============================
@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    if 'admin' not in session:
        return "Unauthorized"

    if model is None:
        return "Model not trained"

    data = request.get_json()
    image_data = data['image'].split(",")[1]

    img_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    (x, y, w, h) = faces[0]
    face_img = img[y:y+h, x:x+w]

    face_img = cv2.resize(face_img, (100, 100))
    face_img = face_img.flatten().reshape(1, -1)

    pred = model.predict(face_img)
    student_id = pred[0]

    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    ref = db.reference('attendance')
    records = ref.get()

    already_marked = False

    if records:
        for key, val in records.items():
            if val["id"] == student_id and val["date"] == today:
                already_marked = True
                break

    if not already_marked:
        ref.push({
            "id": student_id,
            "date": today,
            "time": now.strftime("%H:%M:%S")
        })
        return f"Attendance Saved ID: {student_id}"
    else:
        return f"Already Marked ID: {student_id}"


# ==============================
# 📊 REPORT API (NEW ADDED)
# ==============================
@app.route('/get_report')
def get_report():

    if 'admin' not in session:
        return "Unauthorized"

    ref = db.reference('attendance')
    data = ref.get()

    result = []

    if data:
        for key, val in data.items():
            result.append({
                "id": val.get("id"),
                "date": val.get("date"),
                "time": val.get("time")
            })

    return result


# ==============================
# ▶️ RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
