import cv2
import numpy as np
from tensorflow.keras.models import model_from_json


def preprocess_image(img):
    # Resize the image to match the input size of the model
    resized_img = cv2.resize(img, (224, 224))  # Adjust based on your model's input size

    # Normalize pixel values
    normalized_img = resized_img / 255.0

    # Add batch dimension
    input_image = np.expand_dims(normalized_img, axis=0)

    return input_image


def classify_frame(frame, model):
    # Preprocess the input frame
    input_image = preprocess_image(frame)

    try:
        # Make predictions
        prediction = model.predict(input_image)
    except Exception as e:
        print("Error making predictions:", e)
        return None

    return prediction


# Load the ResNet model architecture and weights
try:
    with open('model_resnet50_architecture.json', 'r') as json_file:
        model_json = json_file.read()

    resnet_model = model_from_json(model_json)
    resnet_model.load_weights('model_resnet50_weights.h5')
except Exception as e:
    print("Error loading model:", e)
    exit()

# Open the video file
video_path = 'hrithik boy.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file", video_path)
    exit()

# Process each frame of the video
while True:
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if there are no more frames

    # Classify the frame
    prediction = classify_frame(frame, resnet_model)

    if prediction is not None:
        # Interpret the results
        threshold = 0.25  # Adjust based on your model and requirements
        if prediction[0] < threshold:
            print("Frame classified as a deepfake.")
        else:
            print("Frame classified as real.")

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
