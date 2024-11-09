from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from twilio.rest import Client
import json

app = Flask(__name__)
CORS(app)

TWILIO_ACCOUNT_SID = ''
TWILIO_AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

yolo_model = YOLO('/Users/mugunthansaravanan/Desktop/Mini-Projects/count/Final/Server/yolov10b.pt')

snake_model = load_model('/Users/mugunthansaravanan/Desktop/Mini-Projects/count/Final/Server/model.h5')

SNAKE_CLASS_MAP = {0: 'Boomslang', 1: 'Cobra'}

ALL_ANIMAL_CLASSES = [
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
    "bear", "zebra", "giraffe", "lion", "tiger", "monkey", 
    "rabbit", "panda"
]

WILD_ANIMAL_CLASSES = [
    "lion", "tiger", "cheetah", "wolf", "fox", "bear", 'Boomslang', 'Cobra' ,"person"
]

YOLO_CONFIDENCE_THRESHOLD = 0.5
SNAKE_CONFIDENCE_THRESHOLD = 0.9

def preprocess_image(image):
    image = image.resize((256, 256))  # Adjust based on the input size used during training
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def send_sms(to_phone_number, message):
    try:
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone_number
        )
        print(f"Message sent with SID: {message.sid}")
    except Exception as e:
        print(f"Error sending SMS: {e}")

def filter_snake_predictions(predictions):
    relevant_predictions = predictions[:, [0, 1]]
    snake_class_index = np.argmax(relevant_predictions)
    snake_confidence = relevant_predictions[0][snake_class_index]
    return snake_class_index, snake_confidence

@app.route('/submit', methods=['POST'])
def submit_form():
    """Endpoint to accept form data containing mobile number and expected counts."""
    data = request.json
    mobile_number = data.get("mobile_number")
    
    if not mobile_number:
        return jsonify({"error": "Mobile number is required"}), 400
    
    if not mobile_number.startswith("+91"):
        mobile_number = "+91" + mobile_number.lstrip("+")  

    expected_counts = data.get("expected_counts", {})  
    
    return jsonify({"message": "Form data received", "mobile_number": mobile_number, "expected_counts": expected_counts})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        image_file = request.files.get('file')
        if not image_file:
            return jsonify({"error": "No file provided"}), 400
        
        form_data = request.form.get('form_data')  
        if form_data:
            form_data = json.loads(form_data)  
            mobile_number = form_data.get("mobile_number")

            if not mobile_number.startswith("+91"):
                mobile_number = "+91" + mobile_number.lstrip("+")
                
            expected_counts = form_data.get("expected_counts", {})
        else:
            return jsonify({"error": "Form data is required"}), 400
        
        print("Received image file for processing.")

        image = Image.open(io.BytesIO(image_file.read()))
        image = image.resize((640, 480))  

        yolo_results = yolo_model(image)

        detected_animals = {}
        detected_wild_animals = {}

        for result in yolo_results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if conf >= YOLO_CONFIDENCE_THRESHOLD:
                    class_name = yolo_model.names[int(cls)]
                    print(f"Detected {class_name} with confidence {conf:.2f}")
                    detected_animals[class_name] = detected_animals.get(class_name, 0) + 1
                    
                    if class_name in WILD_ANIMAL_CLASSES:
                        detected_wild_animals[class_name] = detected_wild_animals.get(class_name, 0) + 1

        if not detected_wild_animals:
            preprocessed_image = preprocess_image(image)
            snake_predictions = snake_model.predict(preprocessed_image)
            snake_class_index, snake_confidence = filter_snake_predictions(snake_predictions)

            if snake_confidence >= SNAKE_CONFIDENCE_THRESHOLD:
                snake_class = SNAKE_CLASS_MAP[snake_class_index]
                detected_animals[snake_class] = detected_animals.get(snake_class, 0) + 1
                detected_wild_animals[snake_class] = detected_wild_animals.get(snake_class, 0) + 1
                print(f"Detected {snake_class} with confidence {snake_confidence:.2f}")

        for animal, expected_count in expected_counts.items():
            detected_count = detected_animals.get(animal, 0)
            if detected_count != expected_count:
                message = f"Detected count for {animal}: {detected_count}. Expected: {expected_count}."
                print(message)
                send_sms(mobile_number, message)
        
        return jsonify(detected_animals)

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
