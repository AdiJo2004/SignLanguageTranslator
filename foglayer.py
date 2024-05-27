import argparse
import io
from PIL import Image
import datetime
import numpy as np
from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for, Response,jsonify
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import glob
import json
from ultralytics import YOLO
from io import BytesIO
import mediapipe as mp


url='http://127.0.0.0:8000'
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
       if 'file' in request.files:
            f = request.files['file']
            file_extension = f.filename.rsplit('.',)[-1].lower()
            if file_extension == 'jpg':
                # Perform the detection
                pil_image = Image.open(BytesIO(f.read()))
                mp_drawing = mp.solutions.drawing_utils
                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
                while True:
  # Read the current frame from the webca

    # Flip the frame horizontally for a more natural selfie-view
                      frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
                      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame using MediaPipe Hands
                      results = hands.process(frame_rgb)

    # Extract hand landmarks
                      landmarks = results.multi_hand_landmarks

                      if landmarks:
        # At least one hand is detected, crop the frame
                         for hand_landmarks in landmarks:
            # Extract the x and y coordinates of the landmarks
                           landmark_list = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

            # Calculate the bounding box manually
                           x_min, y_min = min(landmark_list, key=lambda x: x[0])[0], min(landmark_list, key=lambda x: x[1])[1]
                           x_max, y_max = max(landmark_list, key=lambda x: x[0])[0], max(landmark_list, key=lambda x: x[1])[1]
                           w, h = x_max - x_min, y_max - y_min

            # Crop the frame
                           cropped_frame = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # Resize the cropped frame
                           resized_frame = cv2.resize(cropped_frame, (320, 320),interpolation=cv2.INTER_LANCZOS4)
            # Convert the resized frame to JPEG format
                    
            # Prepare the payload (multipart form data)
                           payload = {'file': (reszied_frame, img_encoded.tobytes(), 'image/jpeg')}

            # Send a POST request to the Flask endpoint
                           response = requests.post(url, files=payload)

            # Check if the request was successful (HTTP status code 200)
                           if response.status_code == 200:
                # Parse the JSON response
                            predictions = response.json()

                # Display the predictions
                           print(predictions)
                         else:
                           print('Error:', response.status_code)
                      else:
        # No hand is detected, skip sending the frame
                        continue
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host='0.0.0.0',port=5000,debug=True)