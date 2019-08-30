import numpy as np
# import cv2
import tensorflow as tf
import psutil
import humanize
import os
import GPUtil as GPU
from tensorflow import keras
import matplotlib.pyplot as plt

print(tf.__version__)
print(np.__version__)
# print(cv2.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

print("Showing GPUs information below:")
GPUs = GPU.getGPUs()
print(GPUs)
gpu = GPUs[0]

process = psutil.Process(os.getpid())
print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
      " | Proc size: " + humanize.naturalsize(process.memory_info().rss))

print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree,
                                                                                            gpu.memoryUsed,
                                                                                            gpu.memoryUtil * 100,
                                                                                            gpu.memoryTotal))

# tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Shape of train batches: ", train_images.shape)
print("Number of train labels available", len(train_labels))

train_images = train_images / 255.0
test_images = test_images / 255.0

print("Now verify the dataset: ")

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

'''
Set up a very basic model without batch normalization
'''
def simple_model():
    my_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    return my_model

def batch_norm_drop_out():
    my_model= keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    return my_model


firstModel = simple_model()

firstModel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting to train the first model:   ")
firstModel.fit(train_images, train_labels, epochs = 40)



secondModel = batch_norm_drop_out()

secondModel.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Starting to train the second model:   ")
secondModel.fit(train_images, train_labels, epochs = 40)

'''
Now evaluate the two models' accuracy respectively
'''
test_loss, test_acc = firstModel.evaluate(test_images, test_labels)
print('Test accuracy of first model:', test_acc)

test_loss, test_acc = secondModel.evaluate(test_images, test_labels)
print('Test accuracy of second model:', test_acc)


