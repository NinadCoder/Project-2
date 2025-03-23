import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    r"C:\Users\ninad\OneDrive/Documents/Projects/Project 2/dataset/training.set",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    r"C:/Users/ninad/OneDrive/Documents/Projects/Project 2/dataset",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


cnn = keras.Sequential([
    layers.Input(shape=(64, 64, 3)),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2),

    layers.Flatten(),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])


cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


cnn.fit(x=training_set, validation_data=test_set, epochs=25)


cnn.save("cat_or_dog.h5")

def predict_image(image_path):
    test_image = load_img(image_path, target_size=(64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)

    if result[0][0] == 1:
        print('Prediction: Dog')
    else:
        print('Prediction: Cat')
