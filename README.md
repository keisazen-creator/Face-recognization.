# Face-recognization.
📌 Face Recognition System using OpenCV (Python)

This project implements a real-time face recognition system using Python and OpenCV. It uses the LBPH (Local Binary Pattern Histogram) algorithm for face recognition and Haar Cascade for face detection. The system can train on custom datasets, save trained models, and recognize faces live using a webcam 🚀



🔹 Features

📷 Real-time face detection via webcam

🧠 Face recognition using LBPH algorithm

💾 Trains once, loads saved model automatically

🏷️ Displays person name with confidence score

❓ Labels unknown faces when confidence is low

🖥️ Fullscreen live camera view




🔹 Technologies Used

Python

OpenCV (cv2)

NumPy

Haar Cascade Classifier

LBPH Face Recognizer





🔹 Dataset Structure

dataset/
│
├── Person1/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── Person2/
│   ├── img1.jpg
│   ├── img2.jpg

Each folder name represents the person’s name used during recognition.




🔹 How It Works

1. Reads face images from dataset folders


2. Trains an LBPH face recognition model


3. Saves the trained model and label mappings


4. Detects faces using Haar Cascade


5. Recognizes faces in real-time via webcam






🔹 How to Run

1. Install dependencies:

pip install opencv-python opencv-contrib-python numpy


2. Update dataset path in the code


3. Run the script:

python face_reco.py



Press ESC to exit the application.



🔹 Use Cases

Attendance systems

Security and surveillance

Personal authentication projects

Computer vision learning projects




⭐ Notes

Requires opencv-contrib-python for LBPH support

Works best with multiple clear images per person

Proper lighting improves accuracy



