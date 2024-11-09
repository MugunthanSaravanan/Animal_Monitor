import numpy as np
from PIL import Image
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load YOLO model for general animal detection
yolo_model = YOLO('/Users/mugunthansaravanan/Desktop/Mini-Projects/count/new/Server/yolov10b.pt')

# Load the trained snake detection model
snake_model = load_model('/Users/mugunthansaravanan/Desktop/Mini-Projects/count/new/Server/model.h5')

# Snake species classes for the snake model (Update based on your specific snake classes)
SNAKE_CLASSES = ['Boomslang', 'Cobra', 'Crotale', 'Taïpan', 'Vipère']

# All animal and wild animal classes
ALL_ANIMAL_CLASSES = [
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
    "bear", "zebra", "giraffe", "lion", "tiger", "monkey", 
    "rabbit", "panda", "person"
]

# Add snake species to wild animals list
WILD_ANIMAL_CLASSES = [
    "lion", "tiger", "cheetah", "wolf", "fox", "bear", "Vipère", 'Boomslang', 'Cobra', 'Crotale', 'Taïpan'
]

# Confidence thresholds
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Set your desired YOLO confidence threshold
SNAKE_CONFIDENCE_THRESHOLD = 0.9

# Preprocess image for snake detection model
def preprocess_image(image):
    image = image.resize((256, 256))  # Adjust based on the input size used during training
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def detect_animals(image_path):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = image.resize((640, 480))  # Resize to the size expected by YOLO model

    # Run YOLO model for general animal detection
    yolo_results = yolo_model(image)

    detected_animals = {}
    detected_wild_animals = {}

    # Process YOLO results
    for result in yolo_results:
        print("YOLO Result Boxes:", result.boxes.xyxy)
        print("YOLO Result Confidences:", result.boxes.conf)
        print("YOLO Result Classes:", result.boxes.cls)
        
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if conf >= YOLO_CONFIDENCE_THRESHOLD:
                class_name = yolo_model.names[int(cls)]
                print(f"Detected {class_name} with confidence {conf:.2f}")
                if class_name in ALL_ANIMAL_CLASSES:
                    detected_animals[class_name] = detected_animals.get(class_name, 0) + 1

                    if class_name in WILD_ANIMAL_CLASSES:
                        detected_wild_animals[class_name] = detected_wild_animals.get(class_name, 0) + 1

    # Run snake detection model
    preprocessed_image = preprocess_image(image)
    snake_prediction = snake_model.predict(preprocessed_image)

    print(f"Snake prediction raw output: {snake_prediction}")

    if snake_prediction.size > 0 and len(snake_prediction[0]) == len(SNAKE_CLASSES):
        snake_class_index = np.argmax(snake_prediction)
        snake_confidence = snake_prediction[0][snake_class_index]
        print(f"Snake detection confidence: {snake_confidence:.2f} for class {SNAKE_CLASSES[snake_class_index]}")

        if snake_confidence > SNAKE_CONFIDENCE_THRESHOLD:
            snake_class = SNAKE_CLASSES[snake_class_index]
            detected_animals[snake_class] = detected_animals.get(snake_class, 0) + 1
            detected_wild_animals[snake_class] = detected_wild_animals.get(snake_class, 0) + 1
    else:
        print("No valid snake prediction returned, or prediction does not match expected classes.")

    print("Detection results:", detected_animals)
    return detected_animals

if __name__ == "__main__":
    image_path ='/Users/mugunthansaravanan/Desktop/Mini-Projects/count/test/image.jpeg'
    image_path1 ='/Users/mugunthansaravanan/Desktop/Mini-Projects/count/test/1.jpg'
    image_path2 ='/Users/mugunthansaravanan/Desktop/Mini-Projects/count/test/2.jpg'
    image_path3 ='/Users/mugunthansaravanan/Desktop/Mini-Projects/count/test/3.jpg'
    image_path4 ='/Users/mugunthansaravanan/Desktop/Mini-Projects/count/test/lion.jpg'


    result = detect_animals(image_path)
    result1 = detect_animals(image_path1)
    result2 = detect_animals(image_path2)
    result3 = detect_animals(image_path3)
    result4 = detect_animals(image_path4)

    print(f"Detected Animals in :", result)

    print(f"Detected Animals in :", result1)

    print(f"Detected Animals in :", result2)

    print(f"Detected Animals in :", result3)

    print(f"Detected Animals in :", result4)
