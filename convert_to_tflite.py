import tensorflow as tf
import os

print("Loading model...")

# Load trained model
model = tf.keras.models.load_model("models/sign_model.h5")

print("Model loaded successfully!")

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

print("Converting model to TensorFlow Lite...")

# Convert model
tflite_model = converter.convert()

# Save converted model
tflite_path = "models/sign_model.tflite"

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")

# Compare sizes
original_size = os.path.getsize("models/sign_model.h5") / (1024 * 1024)
tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)

print(f"\nOriginal Model Size: {original_size:.2f} MB")
print(f"TFLite Model Size: {tflite_size:.2f} MB")