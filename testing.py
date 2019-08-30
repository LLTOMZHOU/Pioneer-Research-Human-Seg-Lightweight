'''
This is for inference only, with the pre-trianed model weights in the same folder
'''

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
from tensorflow import keras
import tensorflow as tf
import cv2

print(tf.__version__)
print(np.__version__)
print(cv2.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
TRAIN_NUM = 1700
TEST_NUM = 300

MODEL_PATH = "./local_weights.hdf5"

'''
Load the model directly from hdf5 file
'''

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def myIOU(y_true, y_pred):
    smooth = 0.001
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, -1) + K.sum(y_pred, -1) - intersection
    return (intersection + smooth) / (union + smooth)

myModel = keras.models.load_model(MODEL_PATH, custom_objects={'bce_dice_loss': bce_dice_loss,
                                                           'dice_loss': dice_loss, 'myIOU':myIOU})
# myModel.summary()

'''
The above code has shown the model's working correctly.
Now use integrate opencv into the production, feeding the model with direct frames
'''

vidCap = cv2.VideoCapture(0)
frame_count = 0
# Create a batch of size 1 to hold every single frame
images = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))

while True:
    frame_count += 1
    ret, img = vidCap.read() #ret ==> boolean flag for successfully capture or not; img ==> a still frame captured
    if not ret:
        print("fails to access your camera")
        break

    # horizontally flip this image
    img = img[:,::-1,:]
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))

    #Pre-process this image
    input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (IMG_HEIGHT, IMG_WIDTH))
    cv2.medianBlur(input_img, 3)
    input_img = input_img/255.
    images[0] = input_img

    #Now feed in this mini-batch with only 1 image inside
    mask = myModel.predict(images)
    mask = mask[0] # reduce output dimensions
    cv2.medianBlur(mask, 5)
    #Threshold this mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask[mask>0.8] = 1
    mask[mask <= 0.8] = 0
    mask = mask.astype(np.uint8)

    # mask = cv2.adaptiveThreshold(mask*255, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 180, 1)
    print(mask.shape)
    print(img.shape)
    img = cv2.bitwise_and(img, img, mask = mask)


    cv2.imshow("VideoStreaming", img)
    stopKey = cv2.waitKey(12)
    keyChar = chr(stopKey & 0xff)

    if keyChar == "q":
        break
    elif keyChar == "s":
        cv2.imwrite("snapshot.jpg", img[:,::-1,:])

cv2.destroyAllWindows()
vidCap.release()

print(frame_count)
