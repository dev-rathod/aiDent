from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

target_size = (224, 224)
batch_size = 32

# Normalize pixel values
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# Load data in from directories and turn it into batches
train_data = train_datagen.flow_from_directory('aiDENT/Train',
                                               target_size=target_size,
                                               batch_size=batch_size,
                                               class_mode='categorical') 

test_data = train_datagen.flow_from_directory('aiDENT/Test',
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

train_data_augmented = train_augmented.flow_from_directory('aiDENT/train',
                                                                  target_size=target_size,
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')