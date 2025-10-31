from deepface import DeepFace
from deepface.modules import verification
import cv2, os, json
import numpy as np

# --- config ---
db_path = "faces_db"
threshold = 0.40   # adjust: lower = stricter (fewer false positives)
metric_name = "cosine"  # "cosine" works well with ArcFace

# load label map
with open("labels.json", "r") as f:
    labels = json.load(f)

# load known faces embeddings
known_faces = []
print("🔍 Loading known faces...")
for file in os.listdir(db_path):
    if file.lower().endswith((".jpg", ".png", ".jpeg",".webp")):
        img_path = os.path.join(db_path, file)
        try:
            rep = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)
            if rep and len(rep) > 0:
                known_faces.append({
                    "name": labels.get(file, "Unknown"),
                    "embedding": np.array(rep[0]["embedding"], dtype=float)
                })
                print(f"✅ Loaded {file} as {labels.get(file, 'Unknown')}")
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")

print(f"\n📸 Starting camera... ({len(known_faces)} faces loaded). Press 'q' to quit.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        rep_frame = DeepFace.represent(img_path=frame, model_name="ArcFace", enforce_detection=False)
        if rep_frame and len(rep_frame) > 0:
            embedding = np.array(rep_frame[0]["embedding"], dtype=float)

            best_match = None
            min_distance = float("inf")

            for known in known_faces:
                known_emb = np.array(known["embedding"], dtype=float)

                # robust call: try both signatures (some versions require metric arg)
                try:
                    # preferred explicit call (works if distance_metric required)
                    dist = verification.find_distance(embedding, known_emb, metric_name)
                except TypeError:
                    # fallback: call without metric (older versions)
                    dist = verification.find_distance(embedding, known_emb)

                # ensure dist is a float
                try:
                    dist = float(dist)
                except:
                    # if dist returned as dict or array, attempt safe extraction
                    if isinstance(dist, (list, tuple, np.ndarray)) and len(dist) > 0:
                        dist = float(dist[0])
                    else:
                        dist = float(dist)

                if dist < min_distance:
                    min_distance = dist
                    best_match = known["name"]

            # decide label based on threshold
            if min_distance < threshold:
                text = f"{best_match} ({min_distance:.2f})"
                color = (0, 255, 0)
                if best_match.lower() == "thief":
                    text = f"🚨 THIEF ALERT: {best_match.upper()} 🚨"
                    color = (0, 0, 255)
            else:
                text = f"Unknown ({min_distance:.2f})"
                color = (255, 255, 255)

            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    except Exception as e:
        # keep printing errors but don't crash
        print("⚠️", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
