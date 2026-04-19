import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

DATA_PATH = "data_processed"

X = []
y = []

actions = os.listdir(DATA_PATH)

for idx, action in enumerate(actions):
    path = os.path.join(DATA_PATH, action)
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file))

        data = data.flatten()

        if data.shape[0] != 63:
            continue

        X.append(data)
        y.append(idx)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(63,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=20)

model.save("models/sign_model.h5")

print("✅ Frame-based model trained and saved!")