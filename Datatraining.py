# from deepface import DeepFace
# from pymongo import MongoClient
# import numpy as np
# import os

# # -----------------------------------
# # ⚙️ MongoDB Setup
# # -----------------------------------
# client = MongoClient("mongodb://localhost:27017/")
# db = client["security"]
# faces_collection = db["faces"]

# # -----------------------------------
# # ⚙️ DeepFace Configuration
# # -----------------------------------
# MODEL_NAME = "ArcFace"       # Best accuracy
# DETECTOR = "retinaface"      # Robust detection

# # -----------------------------------
# # 🔁 Function to Process One Image
# # -----------------------------------
# def process_image(img_path, person_id=None):
#     """Extract embedding and store it in MongoDB"""
#     try:
#         # Step 1: Get embedding
#         embedding_obj = DeepFace.represent(
#             img_path=img_path,
#             model_name=MODEL_NAME,
#             detector_backend=DETECTOR,
#             enforce_detection=True
#         )

#         embedding = embedding_obj[0]["embedding"]

#         # Step 2: Prepare MongoDB document
#         if not person_id:
#             person_id = os.path.splitext(os.path.basename(img_path))[0]

#         doc = {
#             "_id": person_id,
#             "name": person_id,
#             "image_path": img_path,
#             "embedding": embedding
#         }

#         # Step 3: Insert or update
#         faces_collection.update_one({"_id": person_id}, {"$set": doc}, upsert=True)
#         print(f"✅ Stored embedding for {person_id}")

#     except Exception as e:
#         print(f"⚠️ Error processing {img_path}: {e}")

# # -----------------------------------
# # 🧠 User Interaction
# # -----------------------------------
# print("Select mode:")
# print("1️⃣  Single Image")
# print("2️⃣  Multiple Images (manually choose)")
# print("3️⃣  Whole Folder")

# choice = input("👉 Enter your choice (1 / 2 / 3): ").strip()

# if choice == "1":
#     img_path = input("📸 Enter the image path: ").strip()
#     if os.path.exists(img_path):
#         process_image(img_path)
#     else:
#         print("❌ Invalid path. File not found.")

# elif choice == "2":
#     n = int(input("🔢 How many images do you want to process? "))
#     for i in range(n):
#         img_path = input(f"➡️  Enter path for image {i+1}: ").strip()
#         if os.path.exists(img_path):
#             process_image(img_path)
#         else:
#             print(f"❌ Invalid path for image {i+1}")

# elif choice == "3":
#     folder_path = input("📁 Enter folder path: ").strip()
#     if not os.path.isdir(folder_path):
#         print("❌ Invalid folder path.")
#     else:
#         files = [
#             f for f in os.listdir(folder_path)
#             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
#         ]
#         print(f"🗂️ Found {len(files)} image(s). Processing...")
#         for f in files:
#             img_path = os.path.join(folder_path, f)
#             process_image(img_path)
# else:
#     print("❌ Invalid choice. Please restart and choose 1, 2, or 3.")

# print("🎯 All embeddings processed successfully.")
from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
import os, datetime

# -----------------------------------
# MongoDB setup
# -----------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["security"]
faces_collection = db["faces"]

# -----------------------------------
# Config
# -----------------------------------
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"

def compute_mean_embedding(image_paths):
    embeddings = []
    for img in image_paths:
        try:
            emb = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=True
            )[0]["embedding"]
            embeddings.append(emb)
        except Exception as e:
            print(f"⚠️ Error processing {img}: {e}")
    if embeddings:
        return np.mean(embeddings, axis=0).tolist()
    else:
        return None

# -----------------------------------
# Folder structure: criminal_dataset/person_name/*.jpg
# -----------------------------------
DATASET_PATH = r"C:\Users\DELL\Desktop\Facail\Facial\criminal_dataset"

for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)
    if os.path.isdir(person_dir):
        imgs = [
            os.path.join(person_dir, f)
            for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if not imgs:
            continue

        mean_emb = compute_mean_embedding(imgs)
        if mean_emb is not None:
            doc = {
                "_id": person_name.lower().replace(" ", "_"),
                "name": person_name,
                "images": imgs,
                "embedding": mean_emb,
                "num_images": len(imgs),
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            faces_collection.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
            print(f"✅ Stored averaged embedding for {person_name} ({len(imgs)} images)")
        else:
            print(f"⚠️ Skipped {person_name}: no valid embeddings.")

print("🎯 All embeddings stored successfully.")
