# Door Genie - face enrollment image capture
# Captures up to 50 face crops per person using MTCNN and stores them locally.

import cv2
import os
from mtcnn import MTCNN
from datetime import datetime

# Settings
MAX_IMAGES = 50
IMG_SIZE = (160, 160)
SAVE_DIR = "dataset"
LOG_FILE = "logs/capture_log.csv"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

detector = MTCNN()
cap = cv2.VideoCapture(0)

name = input("Enter person's name: ").strip()
person_dir = os.path.join(SAVE_DIR, name)
os.makedirs(person_dir, exist_ok=True)

print(f"[INFO] Capturing images for: {name}")
count = 0

while count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(rgb)

    if result:
        x, y, w, h = result[0]['box']
        x, y = max(0, x), max(0, y)
        face = rgb[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, IMG_SIZE)
            img_name = f"{name}_{count+1}.jpg"
            img_path = os.path.join(person_dir, img_name)
            cv2.imwrite(img_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            count += 1
            print(f"[{count}/{MAX_IMAGES}] Saved {img_name}")
        except Exception as e:
            print(f"[WARNING] Skipped due to error: {e}")

    cv2.imshow("Capturing...", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Log
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open(LOG_FILE, "a") as f:
    f.write(f"{name},Captured,{timestamp}\n")

print(" Image capture completed.")
