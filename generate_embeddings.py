# Door Genie - combined face embedding generation
# Processes the local dataset and saves all encodings/names in encodings/encodings.pkl.

import os
import pickle
import face_recognition
from tqdm import tqdm

DATASET_DIR = "dataset"
ENCODING_FILE = "encodings/encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Processing dataset images...")

for person_name in tqdm(os.listdir(DATASET_DIR)):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(person_dir, img_name)
        image = face_recognition.load_image_file(image_path)
        boxes = face_recognition.face_locations(image, model='hog')
        encodings = face_recognition.face_encodings(image, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print("[INFO] Saving encodings...")
os.makedirs(os.path.dirname(ENCODING_FILE), exist_ok=True)

with open(ENCODING_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(" Embeddings saved to encodings/encodings.pkl")
