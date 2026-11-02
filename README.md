# Door Genie – AI-Based Smart Door Access System

Door Genie is an AI-powered smart door access system built using Python and NVIDIA Jetson Nano. It combines real-time face detection, face recognition, and an LLM-based interaction system to identify visitors and handle entry requests.

The system recognizes registered users through a live camera feed and uses Mistral 7B to interact with visitors, collect information, and simulate access decisions.

## Features

- Real-time face detection using MTCNN.
- Face enrollment by capturing multiple images of registered users.
- Face embedding generation using the `face_recognition` library.
- Face recognition by comparing live camera input with stored embeddings using cosine similarity.
- AI-powered visitor interaction using Mistral 7B Instruct.
- Visitor information collection through an interactive terminal interface.
- Local logging of visitor details and access activities.

## Tech Stack

- **Programming:** Python
- **Hardware:** NVIDIA Jetson Nano
- **Computer Vision:** OpenCV, MTCNN, face_recognition
- **Libraries:** NumPy, scikit-learn
- **LLM:** Mistral 7B Instruct
- **API Integration:** OpenRouter

## Project Structure

```text
Door-Genie/
├── main.py
├── mtcnn_face_capture.py
├── generate_embeddings.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## How It Works

**1. Face Enrollment**

The camera captures up to 50 face images for each registered user. MTCNN detects the face, which is cropped, resized, and stored for further processing.

**2. Face Embedding Generation**

The `face_recognition` library processes the captured images and generates numerical face embeddings. All registered users' embeddings are stored together in a single pickle file (`encodings.pkl`).

**3. Real-Time Face Recognition**

The live camera feed is processed to detect visitors. Their face embeddings are compared with the stored embeddings using cosine similarity to identify registered users.

**4. AI-Powered Visitor Interaction**

Once a visitor is identified, Mistral 7B generates conversational prompts through the OpenRouter API. Visitors can respond through the terminal, provide their details, state their purpose, and request access.

**5. Access Simulation and Logging**

Based on the interaction, the system displays access decisions in the terminal and maintains local visitor logs.

## Getting Started

### Prerequisites

- Python environment
- NVIDIA Jetson Nano
- Camera connected to the device
- OpenRouter API key

### Installation

Clone the repository:

```bash
git clone https://github.com/Varsha701/Door-Genie.git
cd Door-Genie
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### API Configuration

Set your OpenRouter API key as an environment variable:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

### Run the Project

**Step 1:** Capture face samples.

```bash
python mtcnn_face_capture.py
```

**Step 2:** Generate face embeddings.

```bash
python generate_embeddings.py
```

**Step 3:** Start Door Genie.

```bash
python main.py
```

## Future Enhancements

- Integration with physical door-lock mechanisms.
- Voice-based visitor interaction.
- Improved visitor verification and access management.
- Enhanced security and notification features.

---

*Built with Python, Computer Vision, and AI on NVIDIA Jetson Nano.*
