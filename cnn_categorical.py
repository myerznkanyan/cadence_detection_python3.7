# Convolutional Neural Network (Capstone AUA 2020)

# Building the CNN
"""
"Spectogram_28|09|2020.h5" -- testing binary accuracy metrics tf.keras.metrics.BinaryAccuracy(threshold=0.3) 97.35% accuracy
"Spectogram_29|09|2020.h5" -- testing CNN model based on 29|09|2020 dataset which has images from different performance Audicity
47s 469ms/step - loss: 3.1461e-04 - accuracy: 1.0000 - val_loss: 0.6785 - val_accuracy: 0.8696
"Spectogram_30|09|2020.h5" -- based on 29|09|2020 dataset. Confusion matrix calculated
tp: 9323.3496 - fp: 830.0000 - tn: 8614.3496 - fn: 121.0000 - accuracy: 0.9596 - val_loss: 0.3000 - val_tp: 10668.0000 -
val_fp: 833.0000 - val_tn: 9959.0000 - val_fn: 124.0000 - val_accuracy: 0.9646
"""
import os
import tensorflow as tf
import numpy as np
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D, Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pl
# Importing the Keras libraries and packages
from keras.models import Sequential
#Training on CPU
import keras.metrics

METRICS = [
      tf.keras.metrics.TruePositives(name='tp',thresholds=0.3),
      tf.keras.metrics.FalsePositives(name='fp',thresholds=0.3),
      tf.keras.metrics.TrueNegatives(name='tn',thresholds=0.3),
      tf.keras.metrics.FalseNegatives(name='fn',thresholds=0.3),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      #tf.keras.metrics.Precision(name='precision'),
      #tf.keras.metrics.Recall(name='recall'),
      #tf.keras.metrics.AUC(name='auc'),
]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Initialising the CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))
#model.add(Activation('softmax'))
# compile model
#opt = SGD(lr=0.01, momentum=0.9)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[METRICS])
# Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255
                                   ,shear_range = 0.2,
                                   zoom_range = 0.2)

test_datagen = ImageDataGenerator(rescale = 1./255)



# simple early stopping
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

training_set = train_datagen.flow_from_directory("Data_sets/train_image_29|09|2020/train",
                                                 target_size = (64,64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("Data_sets/train_image_29|09|2020/test",
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

history = model.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 4,
                         validation_data = test_set,
                         verbose = 1)

model.save("Spectogram_30|09|2020.h5")
print("Saved model to disk")
from sklearn.metrics import classification_report
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        "Data_sets/train_image_29|09|2020/test",
        target_size=(64, 64),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator, steps = nb_samples)
y_true = test_generator.classes
y_pred = predict > 0.5




#mat = confusion_matrix(y_true,y_pred.round(), normalize=False)
#print(mat)