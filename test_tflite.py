import tensorflow as tf
import numpy as np

print("Loading TFLite model...")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/sign_model.tflite")

# Allocate tensors
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input Shape:", input_details[0]['shape'])

# Create dummy input
sample_input = np.random.rand(1, 63).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], sample_input)

# Run inference
interpreter.invoke()

# Get output tensor
output = interpreter.get_tensor(output_details[0]['index'])

print("\nPrediction Output:")
print(output)