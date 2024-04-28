import pandas as pd
import cv2
from ultralytics import YOLO
from matplotlib import pyplot as plt

model = YOLO('weights/best-3.pt')

img = cv2.imread('test3.jpeg')

plt.imshow(img)

result = model(img, save=True, save_txt=True, project='runs', name='exp')