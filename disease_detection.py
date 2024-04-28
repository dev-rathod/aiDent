from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPool2D, MaxPooling2D, ReLU, TFSMLayer)
from keras.metrics import AUC
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np


target_size = (224, 224)
batch_size = 128

train_dir = 'aiDENT/Train'
test_dir = 'aiDENT/Test'

# Normalize pixel values
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data in from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='categorical') 


test_data = train_datagen.flow_from_directory(test_dir,
                                            target_size=target_size,
                                            batch_size=batch_size,
                                            class_mode='categorical')

##since model is overfitting in multiclass dataset so using dataaugmentations

train_augmented = ImageDataGenerator(rescale=1/255.,
                                            width_shift_range=0.2,
                                            height_shift_range=0.2,
                                            zoom_range=0.2,
                                            horizontal_flip=True)

train_data_augmented = train_augmented.flow_from_directory(train_dir,
                                                                target_size=target_size,
                                                                batch_size=batch_size,
                                                                class_mode='categorical')

train_data_augmented_shuffled = train_datagen.flow_from_directory(train_dir,
                                                                            target_size=target_size,
                                                                            batch_size=batch_size,
                                                                            class_mode='categorical',
                                                                            shuffle=True)

if(len(os.listdir('models/')) == 0):
    print(len(train_data))
    print(len(train_data_augmented))
    print(len(train_data_augmented_shuffled))
    model = keras.Sequential([
        # Convolutional and Pooling Layers
        Conv2D(filters=32, kernel_size=3, strides=1, padding='valid', input_shape=(224, 224, 3)),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=64, kernel_size=2, strides=1, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=128, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=256, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(filters=512, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten layer
        Flatten(),

        # Fully connected layers
        Dense(1024, activation='relu'),
        Dropout(0.4),

        Dense(512, activation='relu'),
        Dropout(0.4),

        # Output layer with sigmoid activation
        Dense(6, activation='sigmoid')
    ])

    # Compile the model
    model.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])



    # Fit the model
    history = model.fit(train_data,
                            epochs=10,
                            steps_per_epoch=len(train_data_augmented_shuffled),
                            validation_data=test_data,
                            validation_steps=len(test_data))

    model.save('models/model3')
else:
    model = TFSMLayer("models/modelk3", call_endpoint='serve')
    di = {}

    for fp in os.listdir('aiDENT/Test/ROOT CANAL'):
        if(fp[0] != '.'):
            img = tf.io.read_file(f'aiDENT/Test/ROOT CANAL/{fp}')
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, size=[224, 224])
            img=img/255.

            # print(np.array(model(tf.expand_dims(img, axis=0)))[0].max())

            class_names = ['CROWN', 'Cavity', 'FILLING', 'IMPACTED', 'IMPLANT', 'ROOT CANAL']# Created a list of class_names from the subdirectories
            pred = np.array(model(tf.expand_dims(img, axis=0)))[0].argmax()
            # print(pred)

            

            pred_class = class_names[pred]
            di[pred_class] = di.get(pred_class, 0) + 1
            print(pred_class)

    print(di)
    


    # print(class_names[model(tf.expand_dims(img, axis=0)).argmax()])
    







