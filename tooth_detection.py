import pandas as pd
import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

model = YOLO('weights/best-3.pt')

img = cv2.imread('full.jpeg')

plt.imshow(img)

result = model.predict(img, save=True, save_txt=True, project='runs', name='exp')

with open('runs/exp/labels/image0.txt', 'r') as f:
    s = f.read()


x = s.split('\n')
x.pop()


height, width, channels = img.shape
print(width, height, channels)


for line in x:
    words = line.split(' ')
    # print(words)
    xc, yc, w, h = int(float(words[1]) * width), int(float(words[2]) * height), int(float(words[3]) * width), int(float(words[4]) * height)
    print(xc, yc, w, h)
    x_min = xc - (w // 2)
    y_min = yc - (h // 2)
    x_max = xc + (w // 2)
    y_max = yc + (h // 2)
    crop_img = img[y_min:y_max, x_min:x_max]
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    cv2.imwrite('TEMP.jpg', crop_img)
    cv2.imshow('ImageWindow', crop_img)
    cv2.waitKey(0)







