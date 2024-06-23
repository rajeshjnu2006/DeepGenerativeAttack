import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def preprocess_img(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("uint8")
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tp = cv2.resize(img, (224, 224,), interpolation=cv2.INTER_CUBIC)
    return tp
def ssim_cal(img_path,ref_img):
    img=cv2.imread(img_path)
    img = preprocess_img(img)
    return ssim(ref_img, img, data_range=img.max() - img.min())
