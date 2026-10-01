from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from keras.utils import plot_model
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV2
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from concurrent.futures import ThreadPoolExecutor
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from utils.model import build_ensemble_model


import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import threading
import numpy as np
import pandas as pd
import cv2
import os
import gc


tf.keras.backend.clear_session()


NUM_CLASSES = 2
INPUT_SHAPE = (128, 128, 3)
DIRECTORY = "PATH TO DATA"


train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

print("[INFO] loading images...")
train_generator = train_datagen.flow_from_directory(
    "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/train",
    batch_size = 32,
    target_size = (128, 128),
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/valid",
    batch_size = 32,
    target_size = (128, 128),
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    "/kaggle/input/140k-real-and-fake-faces/real_vs_fake/real-vs-fake/test",
    batch_size= 32,
    target_size = (128, 128),
    class_mode='binary'
)

ensemble_model = build_ensemble_model(INPUT_SHAPE, NUM_CLASSES)


lr_callbacks = ReduceLROnPlateau(factor=0.5, patience=2)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

hist = ensemble_model.fit(
    train_generator,
    epochs=10,
    callbacks=[lr_callbacks, early_stopping],
    validation_data=val_generator
)

predict = ensemble_model.evaluate(test_generator)

model_json = ensemble_model.to_json()
with open("AAOHybrid_model.json", "w") as json_file:
    json_file.write(model_json)
    
ensemble_model.save_weights("AAOHybrid_final.weights.h5")