'''
import subprocess

# read the requirements.txt file
with open('requirements.txt', 'r') as file:
    requirements = file.read().splitlines()

# install the packages
for requirement in requirements:
    subprocess.check_call(['pip', 'install', requirement])
'''

import os
import glob

import keras.backend
import natsort
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from numpy import random
from numpy import fliplr
from numpy import flipud
import cv2
import matplotlib.pyplot as plt
import random
import time
from PIL import Image
from scipy import misc, ndimage
import skimage
from skimage import transform
from skimage.exposure import equalize_adapthist
from sklearn import preprocessing
from sklearn.utils import shuffle, compute_class_weight
import tensorflow.keras as K
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Activation,
    Flatten,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import cifar10
import pickle
import datetime
from datetime import datetime
import math



# Define constants
batch_size = 32
input_shape = (80, 80, 3)

tf.config.threading.set_inter_op_parallelism_threads(6)

if os.path.exists('file_list_dict.pkl'):
    with open('file_list_dict.pkl', 'rb') as fp:
        file_list_dict = pickle.load(fp)
        print("load from file")
else:

    labelFolder_UV = r'D:\Machine Learning Github\Scores\SHPA_UV*\**\dict3C*[!.mp4]'
    labelFolder_LI = r'D:\Machine Learning Github\Scores\SHPA_LI*\**\dict3C*[!.mp4]'
    labelFolder = r'D:\Machine Learning Github\Scores\**\**\dict3C*[!.mp4]'



    scores = glob.glob(labelFolder)
    scores_UV = glob.glob(labelFolder_UV)
    scores_LI = glob.glob(labelFolder_LI)

    fileList = natsort.natsorted(scores, reverse=False)
    fileList_UV = natsort.natsorted(scores_UV, reverse=False)
    fileList_LI = natsort.natsorted(scores_LI, reverse=False)

    filenames_UV = {}
    for file_path in fileList_UV:
        filenames_UV[os.path.basename(file_path).split('_')[1]] = file_path
    filenames_LI = {}
    for file_path in fileList_LI:
        filenames_LI[os.path.basename(file_path).split('_')[1]] = file_path
    predict_labels_UV = pd.read_csv(r"Z:\PRJ-NeelyLab\Current staff files\Josie\Machine Learning\SHPA_UV\annotated_jumps.csv")
    predict_labels_LI = pd.read_csv(r"Z:\PRJ-NeelyLab\Current staff files\Josie\Machine Learning\SHPA_LI\annotated_jumps.csv")
    predict_labels_UV['video_name'] = predict_labels_UV['video_info'].apply(lambda x: '.'.join(x.split('.')[0:5]))
    predict_labels_UV['frame_number'] = predict_labels_UV['video_info'].apply(lambda x: x.split('.')[5])
    predict_labels_LI['video_name'] = predict_labels_LI['video_info'].apply(lambda x: '.'.join(x.split('.')[0:5]))
    predict_labels_LI['frame_number'] = predict_labels_LI['video_info'].apply(lambda x: x.split('.')[5])


    for i, row in predict_labels_UV.iterrows():
        predict_labels_UV.at[i, 'scores_path'] = filenames_UV[
            predict_labels_UV.at[i, 'video_name'].replace('.mp4', '')
        ]

    for i, row in predict_labels_LI.iterrows():
        predict_labels_LI.at[i, 'scores_path'] = filenames_LI[
            predict_labels_LI.at[i, 'video_name'].replace('.mp4', '')
        ]


    predict = pd.concat([predict_labels_UV, predict_labels_LI])
    predict = predict[predict.jump == 1]
    jump_labels = predict[['video_name', 'frame_number', 'jump', 'scores_path']]

    #fileList = fileList[:500]



    file_list_dict = {}
    all_labels = []


    def get_frame_number(file_name, frame_num):
        start_frame = int(file_name.split('_')[-1])
        abs_frame = start_frame + frame_num
        return abs_frame

    for file in tqdm(fileList, desc='Processing files'):
        # Get the base name of the video file
        video_basename = os.path.basename(file)

        # Get the label file for this video file
        label_file = file.replace('dict3C', 'dictEtho')

        # Load the label data from the label file
        with open(label_file, 'rb') as lf:
            label_data = pickle.load(lf)['behPredictionRec']

        # Loop over the labels to create a dictionary of file names and labels
        for i, label in enumerate(label_data):
            # Get the frame number from the label
            abs_frame_num = get_frame_number(label_file, i)
            frame_num = i

            # Create a key for this file in the dictionary using the video basename and frame number
            file_key = f"{video_basename}|{frame_num}"

            label_set = int(label[0])


            if file in jump_labels['scores_path'].values and label_set != 3:
                if abs_frame_num in jump_labels[jump_labels['scores_path']==file]['frame_number']:
                    label_set = 0

            # Add the file path and label to the dictionary
            file_list_dict[file_key] = {
                'file_path': file,
                'label': label_set,
                'frame_number': frame_num
            }

            all_labels.append(label_set)

    # Create an empty dictionary to store the filtered items
    filtered_dict = {}
    # Loop through each key-value pair in the file_list_dict dictionary
    for key, value in file_list_dict.items():
        # Check if the label is not equal to 'NA'
        if value['label'] != 'NA':
            # If the label is not equal to 'NA', add the key-value pair to the filtered_dict dictionary
            filtered_dict[key] = value
    # Replace the original file_list_dict dictionary with the filtered dictionary
        file_list_dict = filtered_dict

    with open('file_list_dict.pkl', 'wb') as fp:
        pickle.dump(file_list_dict, fp)
def oversample(file_list_dict, percent_positive):
    # Create two lists to store the keys with label=0 and label=1
    label_0_keys = []
    label_other_keys = []
    for key, value in file_list_dict.items():
        if value['label'] == 0:
            label_0_keys.append(key)
        else:
            label_other_keys.append(key)


    # Determine the number of items to keep from label=1 and the number to oversample from label=0
    dataset_size = len(list(file_list_dict))
    num_label_0_oversampled = int((percent_positive*dataset_size)/(1-percent_positive))
    num_0_duplicate = math.ceil(num_label_0_oversampled/len(label_0_keys))

    label_0_keys_duplicated =[]

    i = 0
    while i<=num_0_duplicate:
        label_0_keys_duplicated = label_0_keys_duplicated + label_0_keys
        i = i+1
    label_0_keys = label_0_keys_duplicated
    # Randomly select the items to keep from label=1 and the items to oversample from label=0
    label_0_keys_oversampled = random.choices(label_0_keys, k=num_label_0_oversampled)
    oversample_list = label_other_keys + label_0_keys_oversampled
    # Combine the selected keys into a new dictionary
    oversampled_dict = {}
    all_labels = []
    i = 0
    for key in oversample_list:
        oversampled_dict[i] = file_list_dict[key]
        label = file_list_dict[key]['label']
        all_labels.append(label)
        i = i+1
    return oversampled_dict, all_labels


oversampled_dict, all_labels = oversample(file_list_dict, 0.05)

# Replace the original file_list_dict with the oversampled dictionary
file_list_dict = oversampled_dict

def normalize_contrast_image(image):
    # Normalize the contrast of the image
    normalized_image = equalize_adapthist(image)

    return normalized_image

def random_zoom(image, zoom_range=(0.5, 1.0)):
    # Randomly choose a zoom factor
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])

    # Use the zoom factor to determine the new shape of the image
    h, w = image.shape[:2]
    new_h = int(h * zoom_factor)
    new_w = int(w * zoom_factor)

    # Rescale the image to the new shape using bilinear interpolation
    zoomed_image = transform.rescale(image, (new_h / h, new_w / w), channel_axis=-1, anti_aliasing=True)

    return zoomed_image

def random_rotation(image, max_angle):
    # Generate a random angle between -max_angle and max_angle
    angle = np.random.uniform(-max_angle, max_angle)

    # Rotate the image by the angle using the nearest neighbor method
    rotated = transform.rotate(image, angle, mode='reflect')

    return rotated

def load_image(key,  img_size=(80, 80, 3), apply_filters = 1):
    # Get the file path and frame number from the dictionary
    #key = key.numpy().decode('utf-8')
    file_path = file_list_dict[key]['file_path']
    frame_num = file_list_dict[key]['frame_number']

    # Load the image data for the specified frame
    with open(file_path, 'rb') as f:
        image_data = pickle.load(f)['recIm3C'][frame_num]

    filters = ["rotate", "zoom", "hflip", "vflip", "blur", "noise"]

    filters = random.sample(filters, 3)

    if apply_filters == 1:
        if "rotate" in filters:
            rotation_range = 25
            angle = random.uniform(-rotation_range, rotation_range)
            image_data = random_rotation(image_data, angle)
        if "zoom" in filters:
            # Apply random zoom
            image_data = random_zoom(image_data)
            # Resize the image
            image_data = transform.resize(image_data, img_size)
        if "hflip" in filters:
            image_data = fliplr(image_data)
        if "vflip" in filters:
            image_data = flipud(image_data)
        if "blur" in filters:
            sigma = 3.0
            image_data = skimage.filters.gaussian(image_data,
                                                  sigma=(sigma, sigma),
                                                  truncate=3.5,
                                                  channel_axis=2)
        if "noise" in filters:
            image_data = skimage.util.random_noise(image_data, mode='gaussian', seed=None, clip=True)

    # Normalize the pixel values
    image_min = np.min(image_data)
    image_max = np.max(image_data)
    if image_max - image_min == 0:
        # Set all values to 0 if the range is zero
        image_data = np.zeros_like(image_data)
    else:
        image_data = (image_data - image_min) / (image_max - image_min)

    # Get the label for the specified frame
    label = file_list_dict[key]['label']
    label = tf.one_hot(label, 4)

    return image_data, label


def split_file_list_dict(file_list_dict, train_percentage=0.7):
    file_keys = list(file_list_dict.keys())
    #random.Random(123).shuffle(file_keys)
    file_keys = random.sample(file_keys, 500000)
    num_train = int(len(file_keys) * train_percentage)
    num_val = (len(file_keys)-num_train)
    num_test = int(num_val/2)
    train_keys = file_keys[:num_train]
    val_keys = file_keys[num_train:]
    test_keys = val_keys[num_test:]
    val_keys = val_keys[:(num_val-num_test)]
    return train_keys, val_keys, num_train, num_val, test_keys


def func_train(i):
    i = i.numpy()
    image, label = load_image(i, input_shape, apply_filters=1)
    return image, label

def func_val(i):
    i = i.numpy()
    image, label = load_image(i, input_shape, apply_filters=0)
    return image, label

def _fixup_shape(image, label):
    image.set_shape([None, 80,80,3])
    label.set_shape([None, 4])
    return image, label

# shuffle and create training and val list:
train_list, val_list, num_train, num_val, test_keys = split_file_list_dict(file_list_dict, train_percentage=0.7)

train = tf.data.Dataset.from_generator(lambda: train_list, tf.uint8)
train = train.map(lambda i:tf.py_function(func = func_train,
                                   inp = [i],
                                   Tout = [tf.float64, tf.float32]),
                                   num_parallel_calls= tf.data.AUTOTUNE)


val = tf.data.Dataset.from_generator(lambda: val_list, tf.uint8)
val = val.map(lambda i:tf.py_function(func = func_val,
                                   inp = [i],
                                   Tout = [tf.float64, tf.float32]),
                                   num_parallel_calls= tf.data.AUTOTUNE)

test = tf.data.Dataset.from_generator(lambda: test_keys, tf.uint8)
test = test.map(lambda i:tf.py_function(func = func_val,
                                   inp = [i],
                                   Tout = [tf.float64, tf.float32]),
                                   num_parallel_calls= tf.data.AUTOTUNE)

if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")

try:
   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
except ValueError:
   tpu = None

if tpu:
   policyConfig = 'mixed_bfloat16'
else:
   policyConfig = 'mixed_float16'

policy = tf.keras.mixed_precision.Policy(policyConfig)
tf.keras.mixed_precision.set_global_policy(policy)


buffer_size_train = num_train//5
buffer_size_val = num_val//batch_size//15
train = train.repeat()
val = val.repeat()
#train = train.shuffle(buffer_size_train, reshuffle_each_iteration= False)
train = train.batch(batch_size)
train = train.map(_fixup_shape)
#train = train.repeat().shuffle(buffer_size_train, reshuffle_each_iteration= False).batch(batch_size)
val = val.batch(batch_size)
val = val.map(_fixup_shape)
train = train.prefetch(tf.data.experimental.AUTOTUNE)
val = val.prefetch(tf.data.experimental.AUTOTUNE)
test = test.batch(batch_size)
test = test.map(_fixup_shape)
test = test.prefetch(tf.data.experimental.AUTOTUNE)

class_weights = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(np.ravel(all_labels,order='C')),
    y = np.ravel(all_labels,order='C')
)
class_weights_dict = dict(enumerate(class_weights))



def f1_m(y_true, y_pred, class_weights_dict):
    tp = K.backend.sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.backend.sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.backend.sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.backend.sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.backend.epsilon())
    r = tp / (tp + fn + K.backend.epsilon())

    f1 = 2 * p * r / (p + r + K.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    weights = K.backend.constant(list(class_weights_dict.values()), dtype='float32')
    f1 = f1*weights
    return tf.reduce_mean(f1)

def f1_loss(y_true, y_pred, class_weights_dict):
    tp = K.backend.sum(tf.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.backend.sum(tf.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.backend.sum(tf.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.backend.sum(tf.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.backend.epsilon())
    r = tp / (tp + fn + K.backend.epsilon())

    f1 = 2 * p * r / (p + r + K.backend.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    weights = K.backend.constant(list(class_weights_dict.values()), dtype='float32')
    f1 = f1*weights
    return 1 - tf.reduce_mean(f1)



K.losses.f1_loss = f1_loss
K.losses.f1_loss.__name__ = 'f1_loss'
f1 = lambda y_true, y_pred: f1_m(y_true,y_pred, class_weights_dict)
K.metrics.f1 = f1
K.metrics.f1.__name__ = 'f1'

METRICS = [
    "accuracy",
    K.metrics.Precision(name='precision'),
    K.metrics.Recall(name='recall'),
    K.metrics.AUC(name='auc'),
    K.metrics.f1
]

custom_objects={'f1_loss': f1_loss, 'f1' : f1}

saved_model = load_model(r"Z:\PRJ-NeelyLab\Current staff files\Josie\Machine Learning\Executeable\model_ResNet50_30_11.h5", custom_objects=custom_objects)

model_name = 'model_ResNet50_jump.h5'

model = saved_model


# Assuming that 'model' is the Keras model object
flat1 = Flatten(name='flatten_layer')(model.layers[-1].output)
class1 = Dense(1024, activation='relu', name='dense_layer_1')(flat1)
output = Dense(4, activation='softmax', name='output_layer')(class1)


model = Model(inputs=model.inputs, outputs=output)


model.compile(loss=lambda y_true, y_pred: f1_loss(y_true,y_pred, class_weights_dict),
              optimizer='adam',
              metrics=METRICS
              )


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
mc = ModelCheckpoint(model_name, monitor='val_f1', mode='max', verbose=1, save_best_only=True)

# Create a TensorBoard callback
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

pre_results = model.evaluate(test, use_multiprocessing=True, steps=int(len(test_keys)/batch_size))
print("accuracy, precision, recall, auc, f1:", pre_results)
with open('test_results_pre', 'wb') as file_pi:
    pickle.dump(pre_results, file_pi)

#tf.profiler.experimental.start(logs)

history = model.fit(train, epochs=15, steps_per_epoch=num_train//batch_size,
                    validation_data=val, validation_steps=num_val//batch_size,
                    class_weight=class_weights_dict, callbacks=[es, mc])

#tf.profiler.experimental.stop()

post_results = model.evaluate(test, use_multiprocessing=True, steps=int(len(test_keys)/batch_size))
print("accuracy, precision, recall, auc, f1:", post_results)

model.save('model_ResNet50_jump.h5')

with open('trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

with open('test_results_post', 'wb') as file_pi:
    pickle.dump(post_results, file_pi)

