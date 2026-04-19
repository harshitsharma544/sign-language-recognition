import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import pyttsx3
import time
import threading

# 🔊 Speech function (thread-safe)
def speak(text):
    try:
        engine = pyttsx3.init()   # create engine inside thread
        engine.setProperty('rate', 150)
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# Load model
model = load_model("models/sign_model.h5")

# Labels
actions = os.listdir("data_processed")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

last_action = ""
last_time = 0
cooldown = 1.5  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    action = "..."
    confidence = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            # normalization
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y

            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x - base_x,
                    lm.y - base_y,
                    lm.z
                ])

            # prediction
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data, verbose=0)[0]

            confidence = np.max(prediction)
            predicted_word = actions[np.argmax(prediction)]

            # sentence mapping
            sentence_map = {
                "food": "I want food",
                "water": "I need water",
                "help": "I need help",
                "sorry": "I am sorry",
                "thanks": "Thank you",
                "yes": "Yes, please do it",
                "no": "No, I don't want that",
                "stop": "Please stop this",
                "hello": "Hello, how are you?",
                "please": "Please help me"
            }

            action = sentence_map.get(predicted_word, predicted_word)

            # 🔊 AUDIO (non-blocking + controlled)
            current_time = time.time()

            if action != last_action and (current_time - last_time) > cooldown:
                threading.Thread(target=speak, args=(action,), daemon=True).start()
                last_action = action
                last_time = current_time

    # display
    cv2.putText(frame, f"{action} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()