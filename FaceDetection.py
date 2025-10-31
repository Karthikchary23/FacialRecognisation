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
# 🗄️ MongoDB Setup
# -----------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["security"]
faces_collection = db["faces"]

# Load stored embeddings once into RAM for speed
print("📡 Loading criminal embeddings from MongoDB...")
stored_faces = list(faces_collection.find({}, {"_id": 1, "name": 1, "embedding": 1}))
if not stored_faces:
    print("⚠️ No stored faces found. Please run the training script first.")
    exit()

print(f"✅ Loaded {len(stored_faces)} criminal profiles.\n")

# -----------------------------------
# 🧠 Helper Function - Face Recognition
# -----------------------------------
def recognize_face(face_img):
    try:
        embedding_obj = DeepFace.represent(
            img_path=face_img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=False
        )
        if not embedding_obj:
            return None, 0.0

        new_embedding = np.array(embedding_obj[0]["embedding"])

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
        print(f"⚠️ Error: {e}")
        return None, 0.0

# -----------------------------------
# 🎥 Real-time Face Detection + Recognition
# -----------------------------------
cap = cv2.VideoCapture(0)  # 0 for webcam, or path for CCTV stream/video file
if not cap.isOpened():
    print("❌ Camera not found or cannot be opened.")
    exit()

print("🚀 Starting real-time face recognition...")
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect all faces in frame using DeepFace backend
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

        # Recognize face
        name, similarity = recognize_face(face_crop)

        if name:
            color = (0, 0, 255)  # Red for criminal
            label = f"{name} ({similarity:.2f})"
        else:
            color = (0, 255, 0)  # Green for unknown
            label = f"Unknown ({similarity:.2f})"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    cv2.imshow("🚨 Real-time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Stopped.")
