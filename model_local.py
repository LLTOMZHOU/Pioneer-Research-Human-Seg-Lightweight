import numpy as np
import cv2
import tensorflow as tf

print(tf.__version__)
print("Helloworld")

test_sample_ids = None
test_sample_ids = next(os.walk(test_sample_dir))[2]
print(test_sample_ids[:10])


X_test = np.zeros((len(test_sample_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_test = np.zeros((len(test_sample_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)




