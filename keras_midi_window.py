import os
import cv2
import numpy as np
from keras.models import load_model
import mido

# Load the pre-trained Keras model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Set up the MIDI output
MIDI_OUTPUT_DEVICE = os.environ.get("IAC Driver Bus 1")
output = mido.open_output(MIDI_OUTPUT_DEVICE)  # You may need to specify the MIDI output device here

# Start capturing frames from the webcam
camera = cv2.VideoCapture(0)  # Use 0 for the default camera, or change to 1 if you have multiple cameras

# Define a list of MIDI CC numbers for each class
midi_cc_numbers = list(range(1, len(class_names) + 1))

def draw_bars(frame, predictions):
    bar_height = 10
    for i, score in enumerate(predictions):
        # Scale the confidence score to the range [0, 100] for width of the bar
        bar_width = int(score * 100)
        top_left = (10, i * bar_height + 10)
        bottom_right = (10 + bar_width, i * bar_height + 20)
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), -1)
        cv2.putText(frame, class_names[i].strip(), (10 + bar_width + 5, i * bar_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

while True:
    # Capture a frame from the webcam
    ret, frame = camera.read()
    if not ret:
        break

    # Resize the frame to match the input size of the model (224x224)
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    # Convert the frame to a numpy array
    image_array = np.asarray(resized_frame, dtype=np.float32)

    # Normalize the image array
    normalized_image_array = (image_array / 127.5) - 1

    # Reshape the image array to match the model's input shape
    input_data = normalized_image_array.reshape(1, 224, 224, 3)

    # Predict the class probabilities using the model
    predictions = model.predict(input_data)[0]

    # Draw the prediction bars on the original frame
    draw_bars(frame, predictions)

    # Send class confidences as MIDI CC messages
    for i, confidence_score in enumerate(predictions):
        class_name = class_names[i].strip()
        cc_number = midi_cc_numbers[i]
        cc_value = int(confidence_score * 127)  # Scale the confidence score to the range [0, 127]
        cc_message = mido.Message('control_change', control=cc_number, value=cc_value)
        output.send(cc_message)

    # Display the frame
    cv2.imshow("Camera with Predictions", frame)

    # Listen to the keyboard for presses (Press 'q' to quit)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()
