import cv2
import numpy as np
def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("uint8")
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tp = cv2.resize(img, (224, 224,), interpolation=cv2.INTER_CUBIC)
    return tp
