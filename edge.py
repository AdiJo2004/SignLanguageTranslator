import requests
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound


# URL of the Flask endpoint
url = 'http://192.168.0.24:5000'

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to JPEG format
    _, img_encoded = cv2.imencode('.jpg', frame)

    # Prepare the payload (multipart form data)
    payload = {'file': ('image.jpg', img_encoded.tobytes(), 'image/jpeg')}

    # Send a POST request to the Flask endpoint
    response = requests.post(url, files=payload)

    # Check if the request was successful (HTTP status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        predictions = response.json()

        # Display the predictions
        print(predictions)
        def text_to_speech(text, lang='en'):
            tts = gTTS(text=text, lang=lang)
            tts.save("output.mp3")
            playsound("output.mp3")

# Example usage:
       text = predictions['name']
       text_to_speech(text)
    else:
        print('Error:', response.status_code)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()