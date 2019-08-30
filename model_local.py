'''
This is for inference only, with the pre-trianed model weights in the same folder
'''

import sys
import random
import os
import warnings

import numpy as np

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import models
from tensorflow.keras import callbacks
import tensorflow.keras.backend as K
import glob

from PIL import Image
import glob

import tensorflow as tf
import time
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
sample_dir = "/kaggle/input/human-segmentation-large/newdataset/train/img/"
mask_dir = "/kaggle/input/human-segmentation-large/newdataset/train/mask/"
test_sample_dir = "/kaggle/input/human-img-seg/dataset_kaggle/testing/sample/"
test_mask_dir = "/kaggle/input/human-img-seg/dataset_kaggle/testing/mask/"

