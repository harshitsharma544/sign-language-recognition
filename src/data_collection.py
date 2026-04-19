import cv2
import os
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

DATA_PATH = "data"
SAVE_PATH = "data_processed"

os.makedirs(SAVE_PATH, exist_ok=True)

actions = os.listdir(DATA_PATH)

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    save_action_path = os.path.join(SAVE_PATH, action)
    os.makedirs(save_action_path, exist_ok=True)

    for video in os.listdir(action_path):
        video_path = os.path.join(action_path, video)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0

        #print(f"Processing: {video_path}")  # ✅ DEBUG

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []

                    # wrist-based normalization
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y

                    for lm in hand_landmarks.landmark:
                        landmarks.extend([
                            lm.x - base_x,
                            lm.y - base_y,
                            lm.z
                        ])

                    if len(landmarks) == 63:  # ✅ safety check
                        np.save(
                            os.path.join(save_action_path, f"{video}_{frame_count}.npy"),
                            landmarks
                        )
                        frame_count += 1

        cap.release()

print("✅ Data processing complete!")