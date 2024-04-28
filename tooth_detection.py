import pandas as pd
import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          Input, MaxPool2D, MaxPooling2D, ReLU, TFSMLayer)
from keras.metrics import AUC
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from keras.models import load_model
import numpy as np


model = YOLO('weights/best-3.pt')

img = cv2.imread('full2.jpeg')

plt.imshow(img)

result = model.predict(img, save=True, save_txt=True, project='runs', name='exp')

with open('runs/exp/labels/image0.txt', 'r') as f:
    s = f.read()


x = s.split('\n')
x.pop()


height, width, channels = img.shape
print(width, height, channels)

model = TFSMLayer("models/modelk3", call_endpoint='serve')


for line in x:
    words = line.split(' ')
    # print(words)
    xc, yc, w, h = int(float(words[1]) * width), int(float(words[2]) * height), int(float(words[3]) * width), int(float(words[4]) * height)
    x_min = max(xc - (w // 2), 0)
    y_min = max(yc - (h // 2), 0)
    x_max = min(xc + (w // 2), width)
    y_max = min(yc + (h // 2), height)
    print(x_min, y_min, x_max, y_max)
    print(width, height)
    crop_img = img[y_min:y_max, x_min:x_max]
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    cv2.imwrite('TEMP.jpg', crop_img)
    cv2.imshow('ImageWindow', crop_img)
    cv2.waitKey(0)
    
   
    di = {}

    cv2.imwrite('buffer.jpg', crop_img)
    tooth_tensor = tf.io.read_file('buffer.jpg')
    tooth_tensor = tf.image.decode_image(tooth_tensor, channels=3)
    tooth_tensor = tf.image.resize(tooth_tensor, size=[224, 224])
    tooth_tensor = tf.cast(tooth_tensor, tf.float32)
    tooth_tensor = tf.expand_dims(tooth_tensor, axis=0)

    tooth_tensor=tooth_tensor/255.



    class_names = ['CROWN', 'Cavity', 'FILLING', 'IMPACTED', 'IMPLANT', 'ROOT CANAL']
    res = np.array(model(tooth_tensor))[0]
    pred = res.argmax()

    abbr = {'CROWN': 'CR', 'Cavity': 'CA', 'FILLING': 'F', 'IMPACTED': 'IM', 'IMPLANT': 'IL', 'ROOT CANAL': 'R'}
    


    if(res[pred] > 0.9):
        pred_class = class_names[pred]
        print(pred_class)
        
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
        cv2.rectangle(img, (x_min, y_min - 40), (x_max, y_min), (0, 255, 0), -1)
        cv2.putText(img, abbr[pred_class], (xc - 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                     0.6, (255, 0, 0), 1)

cv2.imwrite('TEMP.jpg', img)






    







