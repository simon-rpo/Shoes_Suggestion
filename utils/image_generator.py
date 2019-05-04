from glob import glob

from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

WIDTH = 299
HEIGHT = 299
BATCH_SIZE = 4000  # non used
PATH = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\train'
OUTPUT_TRAIN = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\output\\train\\sandals'

# data prep
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    PATH,
    target_size=(HEIGHT, WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    save_to_dir=OUTPUT_TRAIN,
    save_format='jpg')

# Batch generator
next(train_generator)
next(train_generator)
next(train_generator)
print('Images processed successfully!')
