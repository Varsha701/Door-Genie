# Door Genie - main face recognition and visitor interaction application
# Mistral 7B is accessed through OpenRouter; provide the API key via environment variable.

import os
# Suppress TensorFlow INFO and step logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#  Disable ALL TensorFlow/Keras progress bars
import logging
logging.getLogger('tensorflow').disabled = True


import cv2
import face_recognition
import pickle
from mtcnn import MTCNN
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import subprocess
import re

#  Include all 5 known names from your dataset folder
# Registered names are loaded from the combined face-encoding file.
VALID_TAGS = ["tcs", "infosys", "family", "friend", "delivery", "guest", "vendor"]
VALID_PURPOSES = [
    "class", "meeting", "delivery", "guest lecture", "interview", "event", "visit",
    "eee", "cse", "ece", "civil", "mechanical"
]

ENCODING_FILE = "encodings/encodings.pkl"
UNKNOWN_DIR = "unknown_faces"
LOG_FILE = "logs/person_log.txt"
VISIT_LOG_FILE = "visit_log.txt"
FRAME_WIDTH = 640

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "mistralai/mistral-7b-instruct"

os.makedirs(UNKNOWN_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def is_valid_tag(answer, vnr_status):
    if vnr_status == "yes":
        return bool(re.match(r"^[0-9a-zA-Z]{5,}$", answer))
    else:
        return answer in VALID_TAGS

class FaceRecognizer:
    def __init__(self, encodings_path=ENCODING_FILE, unknown_dir=UNKNOWN_DIR):
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        self.known_encodings = data["encodings"]
        self.known_names = data["names"]
        self.unknown_dir = unknown_dir
        self.detector = MTCNN()

    def recognize_face(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(rgb)

        recognized_faces = []
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            face_img = rgb[y:y + h, x:x + w]

            try:
                face_img = cv2.resize(face_img, (160, 160))
                encoding = face_recognition.face_encodings(face_img)

                if not encoding:
                    continue

                encoding = encoding[0].reshape(1, -1)
                sims = cosine_similarity(encoding, self.known_encodings)[0]
                best_match_index = np.argmax(sims)
                confidence = sims[best_match_index]

                if confidence > 0.55:
                    name = self.known_names[best_match_index]
                else:
                    name = "Unknown"
                    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(self.unknown_dir, f"unknown_{time_tag}.jpg")
                    cv2.imwrite(path, frame)

                recognized_faces.append({"name": name, "location": (x, y, w, h)})
            except Exception as e:
                print(f"[ERROR] Face skipped: {e}")
        return recognized_faces

def llm_interact(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": messages
    }
    try:
        if not OPENROUTER_API_KEY:
            print("[ERROR] Set the OPENROUTER_API_KEY environment variable to enable Mistral responses.")
            return "The AI assistant is not configured right now."
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] LLM API request failed: {e}")
        return "Sorry, I'm having trouble connecting right now."

def capture_visitor_details():
    try:
        subprocess.run(["python", "mtcnn_face_capture.py"], check=True)
        print("[INFO] Visitor details captured.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to capture visitor details: {e}")
        return False

def log_visit(name, details=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(VISIT_LOG_FILE, "a") as log:
        log.write(f"[{now}] {name}: {details}\n")

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    recognizer = FaceRecognizer()
    face_recognized = False
    visitor_name = "Unknown"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, int(FRAME_WIDTH * 3 / 4)))
        recognized_faces = recognizer.recognize_face(frame)

        if recognized_faces:
            visitor_name = recognized_faces[0]["name"]
            log_visit(visitor_name, "Face recognized.")
            print(f"\n[ FACE RECOGNIZED]: {visitor_name}")
            face_recognized = True
            break

    cap.release()

    # ↓↓↓ LLM conversation starts ↓↓↓
    user_data = {}
    llm_messages = [
        {"role": "system", "content": "You are a concise and polite AI assistant at a smart door. Ask one question at a time: 1) Name, 2) Are you a staff/student of VNRVJIET?, 3) If no, ask for a tag like delivery/family/etc, 4) Ask purpose of visit. Keep responses short and efficient."}
    ]

    questions = [
        "What is your name?",
        "Are you a staff or student of VNRVJIET?",
        "Please enter your identification tag (organization/family/etc):",
        "What is the purpose of your visit?"
    ]

    print("\n[AI ASSISTANT] Starting interaction...\n")

    for i, q in enumerate(questions):
        llm_messages.append({"role": "user", "content": q})
        question = llm_interact(llm_messages)
        print(f"LLM: {question}")

        while True:
            answer = input("You: ").strip().lower()
            if answer in ["q", "quit", "exit"]:
                print("Exiting interaction.")
                return

            if len(answer) <= 1:
                print("Input too short. Try again.")
                continue

            if "name" in q.lower():
                if answer in [name.lower() for name in recognizer.known_names]:
                    user_data["name"] = answer
                    break
                else:
                    print("Name not recognized. Please enter a valid name from the system.")

            elif "staff or student" in q.lower():
                if answer in ["yes", "no"]:
                    user_data["vnr_status"] = answer
                    break
                else:
                    print("Please answer only with 'yes' or 'no'.")

            elif "identification tag" in q.lower():
                if is_valid_tag(answer, user_data.get("vnr_status", "no")):
                    user_data["tag"] = answer
                    break
                else:
                    if user_data.get("vnr_status") == "yes":
                        print("Invalid student ID. Please enter your correct alphanumeric roll number.")
                    else:
                        print("Unrecognized tag. Please enter a known category (e.g., tcs, delivery, family).")

            elif "purpose" in q.lower():
                if answer in VALID_PURPOSES:
                    user_data["purpose"] = answer
                    break
                else:
                    print("Unrecognized purpose. Please try again with a valid one (e.g., class, meeting).")
        
        llm_messages.append({"role": "assistant", "content": question})
        llm_messages.append({"role": "user", "content": answer})
    print("\n Access Granted. Gate Unlocked. Entry logged successfully.")

if __name__ == "__main__":
    main()
