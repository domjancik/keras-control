import os

import cv2
import numpy as np
from keras.models import load_model
from pythonosc import udp_client


# Load the pre-trained Keras model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Set up the OSC client
OSC_HOST = os.environ.get("OSC_HOST", "localhost")
OSC_PORT = os.environ.get("OSC_PORT", 12345)

osc_client = udp_client.SimpleUDPClient("localhost", 12345)  # Replace with your OSC server's address

# Start capturing frames from the webcam
camera = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to 1 if you have multiple cameras

while True:
    # Capture a frame from the webcam
    ret, frame = camera.read()

    # Resize the frame to match the input size of the model (224x224)
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert the frame to a numpy array
    image_array = np.asarray(frame, dtype=np.float32)

    # Normalize the image array
    normalized_image_array = (image_array / 127.5) - 1

    # Reshape the image array to match the model's input shape
    input_data = normalized_image_array.reshape(1, 224, 224, 3)

    # Predict the class probabilities using the model
    predictions = model.predict(input_data)[0]
    class_index = np.argmax(predictions)
    class_name = class_names[class_index].strip()

    # Send the class name and confidence score via OSC
    osc_client.send_message("/classification", (class_name, float(predictions[class_index])))

    # Display the class name and confidence score on the frame
    text = f"Class: {class_name[2:]}, Confidence: {predictions[class_index]:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with classification info
    cv2.imshow("Webcam Classification", frame)

    # Listen to the keyboard for presses (Press 'q' to quit)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()
