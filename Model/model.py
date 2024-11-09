
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Dataset paths
train_dir = '/Users/mugunthansaravanan/Desktop/Mini-Projects/count/Final/Model/Snake Images/train'
test_dir = '/Users/mugunthansaravanan/Desktop/Mini-Projects/count/Final/Model/Snake Images/test'

# Constants
img_width, img_height = 150, 150
batch_size = 32
epochs = 40
num_classes = 2

# Load and preprocess the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Create the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(256, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Predict class for an input image
def predict_image_class(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    if predicted_class == 0:
        return 'venomous'
    else:
        return 'non-venomous'

# Example usage
image_path = '/content/drive/MyDrive/Snake Images/test/Venomous/00550438.jpg'
predicted_class = predict_image_class(image_path)
print('Predicted class:', predicted_class)

"""ensemble of models by utilizing transfer learning with the VGG16 model as the base"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Dataset paths
train_dir = '/content/drive/MyDrive/Snake Images/train'
test_dir = '/content/drive/MyDrive/Snake Images/test'

# Constants
img_width, img_height = 150, 150
batch_size = 32
epochs = 10
num_classes = 2
num_models = 3  # Number of models in the ensemble

# Load and preprocess the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Create the ensemble of models
ensemble_models = []
for _ in range(num_models):
    # Use VGG16 as the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the base model
    model = keras.Sequential([
        base_model,
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    ensemble_models.append(model)

# Train the ensemble models
for model in ensemble_models:
    model.fit(train_generator,
              steps_per_epoch=train_generator.samples // batch_size,
              epochs=epochs)

# Evaluate the ensemble models
ensemble_scores = []
for model in ensemble_models:
    _, acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
    ensemble_scores.append(acc)

# Average ensemble accuracy
ensemble_accuracy = np.mean(ensemble_scores)
print('Ensemble accuracy:', ensemble_accuracy)

# Predict class for an input image using the ensemble
def predict_image_class(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = np.zeros((num_classes,))
    for model in ensemble_models:
        predictions += model.predict(img_array)[0]

    predicted_class = np.argmax(predictions)

    if predicted_class == 0:
        return 'venomous'
    else:
        return 'non-venomous'

# Example usage
image_path = '/content/drive/MyDrive/Snake Images/test/Non Venomous/112282-850x565-Yellow_Rat_Snake.jpg'
predicted_class = predict_image_class(image_path)
print('Predicted class:', predicted_class)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Dataset paths
train_dir = '/content/drive/MyDrive/Snake Images/train'
test_dir = '/content/drive/MyDrive/Snake Images/test'

# Constants
img_width, img_height = 150, 150
batch_size = 32
epochs = 10
num_classes = 2

# Load and preprocess the dataset
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Use VGG16 as the base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the base model
model = keras.Sequential([
    base_model,
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Predict class for an input image
def predict_image_class(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    if predicted_class == 1:
        return 'venomous'
    else:
        return 'non-venomous'

# Example usage
image_path = '/content/drive/MyDrive/Snake Images/test/Non Venomous/10.jpg'
predicted_class = predict_image_class(image_path)
print('Predicted class:', predicted_class)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
img_width, img_height = 224, 224
batch_size = 32
num_classes = 2
epochs = 10

# Function to load and preprocess the dataset
def load_dataset(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

# Function to create the model with transfer learning
def create_model():
    base_model = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Function to plot the accuracy graph
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Function to draw a colored confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Function to predict the class for an input image
def predict_image_class(model, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    if predicted_class == 1:
        return 'venomous'
    else:
        return 'non-venomous'

# Load the dataset
train_dir = '/content/drive/MyDrive/Snake Images/train'
test_dir = '/content/drive/MyDrive/Snake Images/test'
train_generator, test_generator = load_dataset(train_dir, test_dir)

# Create the model
model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // batch_size)

# Calculate the accuracy of the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Predict class for an input image
image_path = '/content/drive/MyDrive/Snake Images/test/Non Venomous/1.jpg'
predicted_class = predict_image_class(model, image_path)
print('Predicted class:', predicted_class)

# Display the image
img = keras.preprocessing.image.load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Plot the accuracy graph
plot_accuracy(history)

# Predict classes for the test dataset
test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Draw a colored confusion matrix
class_names = ['venomous', 'non-venomous']
plot_confusion_matrix(y_true, y_pred, class_names)

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Constants
img_width, img_height = 224, 224
batch_size = 32
num_classes = 2
epochs = 10

# Function to load and preprocess the dataset
def load_dataset(train_dir, test_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator

# Function to create the model with transfer learning
def create_model():
    base_model = MobileNetV2(input_shape=(img_width, img_height, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# Function to plot the accuracy graph
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Function to draw a colored confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# Function to predict the class for an input image
def predict_image_class(model, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    if predicted_class == 1:
        return 'venomous'
    else:
        return 'non-venomous'

# Load the dataset
train_dir = '/content/drive/MyDrive/Snake Images/train'
test_dir = '/content/drive/MyDrive/Snake Images/test'
train_generator, test_generator = load_dataset(train_dir, test_dir)

# Create the model
model = create_model()

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=test_generator.samples // batch_size)

# Calculate the accuracy of the model
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print('Test accuracy:', test_acc)

# Predict class for an input image
image_path = '/content/drive/MyDrive/Snake Images/test/Venomous/00550438.jpg'
predicted_class = predict_image_class(model, image_path)
print('Predicted class:', predicted_class)

# Display the image
img = keras.preprocessing.image.load_img(image_path)
plt.imshow(img)
plt.axis('off')
plt.show()

# Plot the accuracy graph
plot_accuracy(history)

# Predict classes for the test dataset
test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Draw a colored confusion matrix
class_names = ['venomous', 'non-venomous']
plot_confusion_matrix(y_true, y_pred, class_names)
# Predict classes for the test dataset
test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Calculate evaluation metrics
report = classification_report(y_true, y_pred, target_names=class_names)
print('Classification Report:')
print(report)

# Necessary imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Predict classes for the test dataset
test_generator.reset()
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize the confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print classification report
report = classification_report(y_true, y_pred, target_names=class_names)
print('Classification Report:')
print(report)