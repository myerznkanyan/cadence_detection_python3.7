
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

from keras.utils.vis_utils import plot_model

model = load_model('Spectogram.h5')
#model.summary()

def testfile():
    rslt = model.predict(img_pred)
    if rslt[0][0] == 1:
        prediction = "no_cadence"
    else:
        prediction = "cadence"

    print(prediction)

img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\dataImage\\test\\cadence\\cadence_17.PNG',
        target_size=(64, 64))

img_pred = image.img_to_array(img_pred)
img_pred = np.expand_dims(img_pred, axis=0)
testfile()

"""    

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\LowLight\\noperson\\noperson (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("LowLight noperson finnish")


for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\LowLight\\person\\person (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("LowLight person finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\MidLight\\noperson\\noperson (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("MidLight noperson finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\MidLight\\person\\person (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("MidLight person finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\HighLight\\noperson\\noperson (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("HighLight noperson finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\HighLight\\person\\person (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("HighLight person finnish")

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#rslt = model.predict(img_pred)



"""














=======
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import time

from keras.utils.vis_utils import plot_model

model = load_model('Model/Spectogram_11|09|2020.h5')
#model.summary()

def testfile():
    rslt = model.predict(img_pred)
    if rslt[0][0] == 1:
        prediction = "C6Note"
    else:
        prediction = "A5Note"

    print(prediction)


while True:
    img_pred = image.load_img(
        "train_image/imagestest/C6Note/C6Note (13).jpeg",
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)

    testfile()
    time.sleep(0.025)




"""    

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\LowLight\\noperson\\noperson (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("LowLight noperson finnish")


for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\LowLight\\person\\person (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("LowLight person finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\MidLight\\noperson\\noperson (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("MidLight noperson finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\MidLight\\person\\person (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("MidLight person finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\HighLight\\noperson\\noperson (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("HighLight noperson finnish")

for i in range(1,6):
    integer = str(i)
    img_pred = image.load_img(
        'C:\\Users\\m_yer\\PycharmProjects\\untitled\\src\\data\\test\\Testindata\\HighLight\\person\\person (' + integer +').jpg',
        target_size=(64, 64))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    testfile()
    #printing()

print("HighLight person finnish")

#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#rslt = model.predict(img_pred)



"""















