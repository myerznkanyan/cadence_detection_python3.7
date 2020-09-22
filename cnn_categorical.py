# Convolutional Neural Network (Capstone AUA 2020)

# Building the CNN

import os

from keras.layers import Conv2D, Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# Importing the Keras libraries and packages
from keras.models import Sequential

#Training on CPU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Initialising the CNN
model = Sequential()
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
#model.add(Activation('softmax'))
# compile model
#opt = SGD(lr=0.01, momentum=0.9)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255
                                   ,shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)



# simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

training_set = train_datagen.flow_from_directory("train_image_18|09|2020_v2/train",
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("train_image_18|09|2020_v2/test",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 4,
                         validation_data = test_set,
                         verbose = 1)

model.save("Spectogram_18|09|2020_v3.h5")
print("Saved model to disk")
#Save model in .h5