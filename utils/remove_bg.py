# remove_bg.py
import cv2
import numpy as np
import mediapipe as mp


mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def remove_background(img: np.ndarray, bg_color=(255, 255, 255)):
    """
    Menghapus background menggunakan MediaPipe Selfie Segmentation.
    Output: BGR numpy array (OpenCV format)
    """
    h, w = img.shape[:2]

    # Convert BGR → RGB untuk MediaPipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_selfie.process(rgb)

    if res.segmentation_mask is None:
        # Jika gagal segmentasi → return gambar asli
        return img

    mask = res.segmentation_mask
    mask = cv2.resize(mask, (w, h))

    # Normalisasi mask → 0–1
    mask = np.expand_dims(mask, 2)
    mask = np.repeat(mask, 3, axis=2)

    # Background target
    bg = np.full(img.shape, bg_color, dtype=np.uint8)

    # Blend wajah + background solid
    output = img * mask + bg * (1 - mask)
    output = output.astype(np.uint8)

    return output


def remove_background_transparent(img: np.ndarray):
    """
    Menghapus background dan membuat alpha channel (RGBA PNG).
    """
    h, w = img.shape[:2]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = mp_selfie.process(rgb)

    if res.segmentation_mask is None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    mask = res.segmentation_mask
    mask = cv2.resize(mask, (w, h))

    alpha = (mask * 255).astype(np.uint8)

    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha

    return rgba
