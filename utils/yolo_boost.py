import numpy as np

def adaptive_threshold(label, base_thresh, skin_ratio):
    if skin_ratio < 0.5:
        return base_thresh + 0.05  # hindari false positive di pinggir wajah
    if label in ["Dark Circle", "Acne Scar"]:
        return base_thresh - 0.03  # lebih sensitif
    return base_thresh
