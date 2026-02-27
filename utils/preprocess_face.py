import cv2
import numpy as np

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def smooth_detail_enhance(img):
    # Bilateral filter preserves edges
    blur = cv2.bilateralFilter(img, 13, 90, 90)
    sharp = cv2.addWeighted(img, 1.3, blur, -0.3, 0)
    return sharp

def adaptive_gamma(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    gamma = 1.0 + (0.5 - mean_val / 255.0)
    gamma = np.clip(gamma, 0.6, 1.4)
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess_face(img):
    img = white_balance(img)
    img = smooth_detail_enhance(img)
    img = adaptive_gamma(img)
    return img
