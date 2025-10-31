import cv2
import numpy as np
from deepface import DeepFace
from pymongo import MongoClient
import time

# -----------------------------------
# ⚙️ Configuration
# -----------------------------------
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
SIMILARITY_THRESHOLD = 0.65

# -----------------------------------
# ☁️ MongoDB Atlas Connection
# -----------------------------------
MONGO_URI = "mongodb+srv://lingojikarthikchary_db_user:GBsqnZv0Uy370M4V@cluster0.ygyedqr.mongodb.net/?appName=Cluster0"

try:
    client = MongoClient(MONGO_URI)
    db = client["security"]
    faces_collection = db["faces"]
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"❌ Failed to connect MongoDB Atlas: {e}")
    exit()

# -----------------------------------
# 🧠 Load Stored Embeddings
# -----------------------------------
print("📡 Loading stored face embeddings from MongoDB Atlas...")
stored_faces = list(faces_collection.find({}, {"_id": 1, "name": 1, "embedding": 1}))
if not stored_faces:
    print("⚠️ No face data found in the database. Please run your training script first.")
    exit()

print(f"✅ Loaded {len(stored_faces)} profiles from MongoDB Atlas.\n")

# -----------------------------------
# 🔍 Recognition Function
# -----------------------------------
def recognize_face(face_img):
    try:
        emb_obj = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
        if not emb_obj:
            return None, 0.0

        new_embedding = np.array(emb_obj[0]["embedding"])

        best_match = None
        best_similarity = 0

        for face in stored_faces:
            db_embedding = np.array(face["embedding"])
            sim = np.dot(new_embedding, db_embedding) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(db_embedding)
            )
            if sim > best_similarity:
                best_similarity = sim
                best_match = face

        if best_similarity > SIMILARITY_THRESHOLD:
            return best_match["name"], best_similarity
        else:
            return None, best_similarity

    except Exception as e:
        print(f"⚠️ Error during recognition: {e}")
        return None, 0.0

# -----------------------------------
# 🎥 Real-time Recognition
# -----------------------------------
cap = cv2.VideoCapture(0)  # Change to video path or RTSP link for CCTV
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

print("🚀 Starting real-time face recognition... Press 'q' to quit.")
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = DeepFace.extract_faces(
        img_path=frame,
        detector_backend=DETECTOR,
        enforce_detection=False
    )

    for det in detections:
        facial_area = det["facial_area"]
        x, y, w, h = (
            facial_area["x"],
            facial_area["y"],
            facial_area["w"],
            facial_area["h"]
        )

        face_crop = frame[y:y+h, x:x+w]

        name, similarity = recognize_face(face_crop)

        if name:
            color = (0, 0, 255)  # Red = match found
            label = f"{name} ({similarity:.2f})"
        else:
            color = (0, 255, 0)  # Green = unknown
            label = f"Unknown ({similarity:.2f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("🚨 Real-time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Recognition stopped.")
