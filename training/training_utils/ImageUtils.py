import cv2
import numpy as np
import torch


def load_img(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img /= 255.0
    return img


def save_img(img, path):
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
