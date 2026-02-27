import cv2
import numpy as np

def face_align(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    det = faces.detectMultiScale(gray, 1.2, 6)

    if len(det)==0:
        return img

    x,y,w,h = det[0]
    face = img[y:y+h,x:x+w]

    # align using eyes
    return cv2.resize(face, (720, 960))

def upscale_patch(roi):
    return cv2.resize(roi, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

def enhance_skin_texture(img):
    blur = cv2.GaussianBlur(img,(0,0),sigmaX=2)
    high = cv2.addWeighted(img,1.4, blur,-0.4,0)
    return high
