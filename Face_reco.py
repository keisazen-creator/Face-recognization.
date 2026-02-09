# Import necessary libraries
import cv2
import os
import numpy as np
import pickle  # To save and load label-name mapping

# Load Haar Cascade for face detection
face_cap = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Create LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Confidence threshold
CONFIDENCE_THRESHOLD = 80

# Function to train the face recognizer
def train_face_recognizer(image_folder):
    faces = []
    labels = []
    label_dict = {}

    for label, person_name in enumerate(os.listdir(image_folder)):
        person_folder = os.path.join(image_folder, person_name)

        if os.path.isdir(person_folder):
            label_dict[label] = person_name

            for image_name in os.listdir(person_folder):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(person_folder, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if img is not None:
                    faces.append(img)
                    labels.append(label)

    if faces and labels:
        recognizer.train(faces, np.array(labels))
        recognizer.save('trained_model.yml')

        with open('label_dict.pkl', 'wb') as f:
            pickle.dump(label_dict, f)

        print("Model trained and saved successfully.")
    else:
        print("No valid images found in dataset.")

    return label_dict


# Load existing model or train new one
try:
    recognizer.read('trained_model.yml')
    with open('label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    print("Pre-trained model loaded successfully.")
except:
    print("No pre-trained model found. Training a new model.")
    label_dict = train_face_recognizer("PUT_YOUR_DATASET_FOLDER_PATH_HERE")


# Start webcam
video_cap = cv2.VideoCapture(0)

cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Face Recognition",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    video_data = cv2.flip(video_data, 1)
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        label, confidence = recognizer.predict(roi)

        if confidence < CONFIDENCE_THRESHOLD:
            name = label_dict.get(label, "Unknown")
        else:
            name = "Unknown"

        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(
            video_data,
            f"{name} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2
        )

    cv2.imshow("Face Recognition", video_data)

    if cv2.waitKey(10) == 27:  # ESC key
        break

video_cap.release()
cv2.destroyAllWindows()
