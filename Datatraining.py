from deepface import DeepFace
from pymongo import MongoClient
import numpy as np
import os, datetime


MONGO_URI = "mongodb+srv://lingojikarthikchary_db_user:GBsqnZv0Uy370M4V@cluster0.ygyedqr.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)

# Test connection
try:
    print("✅ Connected to MongoDB Atlas:", client.list_database_names())
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    exit()

db = client["security"]
faces_collection = db["faces"]


MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"  # try "opencv" if RetinaFace fails

# -----------------------------------
# 🧠 Compute Mean Embedding
# -----------------------------------
def compute_mean_embedding(image_paths):
    embeddings = []
    for img in image_paths:
        try:
            print(f"🔍 Processing image: {img}")
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
        print(f"✅ Generated {len(embeddings)} embeddings")
        return np.mean(embeddings, axis=0).tolist()
    else:
        print("❌ No valid embeddings generated")
        return None


def store_to_db(person_name, imgs, embedding):
    doc = {
        "_id": person_name.lower().replace(" ", "_"),
        "name": person_name,
        "images": imgs,
        "embedding": embedding,
        "num_images": len(imgs),
        "created_at": datetime.datetime.utcnow(),
        "updated_at": datetime.datetime.utcnow()
    }

    faces_collection.update_one({"_id": doc["_id"]}, {"$set": doc}, upsert=True)
    print(f"✅ Stored embedding for '{person_name}' ({len(imgs)} images)")

# -----------------------------------
# 🧭 Training Modes
# -----------------------------------
print("\nChoose training mode:")
print("1️⃣  Single Image")
print("2️⃣  Multiple Images (manual input)")
print("3️⃣  Whole Folder Dataset")

choice = input("\n👉 Enter your choice (1 / 2 / 3): ").strip()

if choice == "1":
    img_path = input("📸 Enter image path: ").strip()
    if os.path.exists(img_path):
        name = os.path.splitext(os.path.basename(img_path))[0]
        emb = compute_mean_embedding([img_path])
        if emb:
            store_to_db(name, [img_path], emb)
    else:
        print("❌ Invalid image path")

elif choice == "2":
    n = int(input("🔢 How many images do you want to process? "))
    for i in range(n):
        img_path = input(f"➡️  Enter path for image {i+1}: ").strip()
        if os.path.exists(img_path):
            name = os.path.splitext(os.path.basename(img_path))[0]
            emb = compute_mean_embedding([img_path])
            if emb:
                store_to_db(name, [img_path], emb)
        else:
            print(f"❌ Invalid path for image {i+1}")

elif choice == "3":
    dataset = input("📁 Enter dataset folder path: ").strip()
    if not os.path.isdir(dataset):
        print("❌ Invalid folder path")
    else:
        persons = os.listdir(dataset)
        print(f"🧩 Found {len(persons)} subfolders in dataset")

        for person in persons:
            person_dir = os.path.join(dataset, person)
            if not os.path.isdir(person_dir):
                continue

            imgs = [
                os.path.join(person_dir, f)
                for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]

            print(f"\n🧠 Found {len(imgs)} images for '{person}'")

            if not imgs:
                print(f"⚠️ No valid images for {person}")
                continue

            emb = compute_mean_embedding(imgs)
            if emb:
                store_to_db(person, imgs, emb)
            else:
                print(f"❌ No embedding generated for '{person}'")

else:
    print("❌ Invalid choice. Please restart and choose 1, 2, or 3.")

print("\n🎯 All embeddings processed and stored successfully.")
