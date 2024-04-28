from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPool2D, MaxPooling2D, ReLU)
from keras.metrics import AUC
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

target_size = (224, 224)
batch_size = 32

train_dir = 'aiDENT/Train'
test_dir = 'aiDent/Test'

# Normalize pixel values
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

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
                                             rotation_range=20, # note: this is an int not a float
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


model = keras.Sequential([
    # Convolutional and Pooling Layers
    Input((224, 224, 3)),
    Conv2D(filters=32, kernel_size=3, strides=1, padding='valid'),
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
history = model.fit(train_data_augmented,
                          epochs=2,
                          steps_per_epoch=len(train_data_augmented_shuffled),
                          validation_data=test_data,
                          validation_steps=len(test_data))