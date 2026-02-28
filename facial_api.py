"""
facial_api.py

Backend API untuk analisis kondisi kulit wajah dari 3 sudut (left, center, right)
dengan pipeline deteksi lengkap (boosted ensemble + adaptive fusion).
+ Notifikasi Email & WhatsApp otomatis (dengan attachment gambar).
+ Endpoint Register User Terpisah.
"""

import os
import re
import json
import base64
import smtplib
import requests
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
    BackgroundTasks,
    Form,
    Depends,
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# ============================================================
#  DATABASE SETUP (SQLAlchemy)
# ============================================================
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# Ganti dengan kredensial database kamu
# DATABASE_URL = "mysql+pymysql://root:@localhost/mirrasense" 
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://mirra_user:passwordkuat123@localhost/mirrasense")
# Format: mysql+pymysql://username:password@host/dbname

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255))
    email = Column(String(255))
    tel = Column(String(255)) # Kolom untuk WhatsApp
    created_at = Column(DateTime, default=datetime.now)

# Dependency untuk mendapatkan session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Model untuk Input User
class UserCreate(BaseModel):
    name: str
    email: str
    tel: str

# ============================================================
#  IMPORT UTILS
# ============================================================
try:
    from utils.remove_bg import remove_background
    from utils.yolo_boost import adaptive_threshold
except ImportError:
    print("[WARN] Modul utils tidak ditemukan. Menggunakan fungsi dummy.")
    def remove_background(img, bg_color=None): return img
    def adaptive_threshold(label, base, skin_r): return base

from collections import defaultdict

# ============================================================
#  KONFIGURASI NOTIFIKASI
# ============================================================

# Aktifkan 2FA di Gmail dan buat App Password: https://myaccount.google.com/apppasswords
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SMTP_USERNAME = "nandaarianto58@gmail.com"
# SMTP_PASSWORD = "nbwzecekclklzird"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "nandaarianto58@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "nbwzecekclklzird")

# Konfigurasi WhatsApp Gateway
WA_API_URL = "https://api.fonnte.com/send" 
WA_API_TOKEN = os.getenv("WA_API_TOKEN", "token_api_wa_kalian")

# ============================================================
#  FASTAPI APP INIT
# ============================================================

app = FastAPI(
    title="Skin Analysis Backend",
    version="2.4.0",
    description="Backend analisis kulit dengan YOLO + Notifikasi Email + DB Storage."
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk development. Production: ganti dengan URL frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  KONFIGURASI GLOBAL
# ============================================================

MODEL_PATH = r"runs/mirra_v2_stage1/weights/best.pt"

IMG_SIZE: int = 864
DEVICE: Any = "cpu"  # 0 untuk GPU, "cpu" untuk CPU
BASE_CONF: float = 0.12
IOU_MERGE: float = 0.55
SOFT_NMS_IOU: float = 0.40
SOFT_NMS_DECAY: float = 0.60

OUTPUT_ROOT: str = "outputs_face"

SKIN_LOWER_1 = np.array([0, 30, 60])
SKIN_UPPER_1 = np.array([25, 180, 255])
SKIN_LOWER_2 = np.array([0, 10, 40])
SKIN_UPPER_2 = np.array([30, 200, 255])


# ============================================================
#  LABEL MAPPING
# ============================================================

MODEL_LABEL_TO_APP: Dict[str, str] = {
    "Acne": "Jerawat",
    "Blackheads": "Komedo Hitam",
    "Whiteheads": "Komedo Putih",
    "Dark_Spots": "Flek Hitam",
    "Wrinkles": "Kerutan",
    "Eyebags": "Kantung Mata",
    "Oily_Skin": "Kulit Berminyak",
    "Dry_Skin": "Kulit Kering",
    "Enlarged_Pores": "Pori-pori Besar",
    "Skin_Redness": "Kemerahan Kulit",
    "Acne_Scars": "Bekas Jerawat",
    "Papules": "Papula",
    "Pustules": "Pustula",
    "Nodules": "Nodul Jerawat",
    "Cystic": "Jerawat Kistik",
    "Folliculitis": "Folikulitis",
    "Milium": "Milia",
    "Keloid": "Keloid",
    "Syringoma": "Siringoma",
    "Flat_Wart": "Kutil Datar",
}

PER_CLASS_TH: Dict[str, float] = {
    "Jerawat": 0.16,
    "Komedo Hitam": 0.15,
    "Komedo Putih": 0.15,
    "Flek Hitam": 0.15,
    "Kerutan": 0.17,
    "Kantung Mata": 0.15,
    "Kulit Berminyak": 0.14,
    "Kulit Kering": 0.14,
    "Pori-pori Besar": 0.15,
    "Kemerahan Kulit": 0.15,
    "Bekas Jerawat": 0.18,
    "Papula": 0.16,
    "Pustula": 0.16,
    "Nodul Jerawat": 0.20,
    "Jerawat Kistik": 0.22,
    "Folikulitis": 0.18,
    "Milia": 0.14,
    "Keloid": 0.20,
    "Siringoma": 0.17,
    "Kutil Datar": 0.18,
}

# severity color
SEV_COL: Dict[str, Tuple[int, int, int]] = {
    "sangat ringan": (180, 255, 180),
    "ringan": (0, 255, 0),
    "sedang": (0, 200, 255),
    "berat": (0, 165, 255),
    "sangat berat": (0, 0, 255),
}

# variant weights
VARIANT_WEIGHTS: Dict[str, float] = {
    "original": 1.0,
    "f:hsv": 0.9,
    "f:clahe": 1.2,
    "f:rgb": 1.1,
    "f:gray": 0.8,
    "flipH": 1.0,
    "s:0.75": 0.8,
    "s:1.00": 1.0,
    "s:1.25": 1.2,
}


# ============================================================
#  LOAD REKOMENDASI PRODUK (FROM JSON)
# ============================================================

RECO_FILE = os.path.join("data", "rekomendasi_produk.json")

def load_rekomendasi_produk():
    if not os.path.exists(RECO_FILE):
        print("[WARN] rekomendasi_produk.json tidak ditemukan")
        return {}

    with open(RECO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

REKOMENDASI_PRODUK = load_rekomendasi_produk()


def generate_produk_rekomendasi(main_concerns, severity_map, uv_level="medium"):
    output = {
        "produk_kulit": {},
        "rekomendasi_uv": None,
    }

    uv_block = REKOMENDASI_PRODUK.get("GLOBAL_UV", {})
    output["rekomendasi_uv"] = uv_block.get(uv_level)

    for label in main_concerns:
        if label not in REKOMENDASI_PRODUK:
            output["produk_kulit"][label] = []
            continue

        sev = severity_map.get(label, "ringan")
        produk = (
            REKOMENDASI_PRODUK[label].get(sev)
            or REKOMENDASI_PRODUK[label].get("ringan", [])
        )
        output["produk_kulit"][label] = produk

    return output


# ============================================================
#  UTILITAS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def adaptive_color_balance(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def apply_filters(img: np.ndarray) -> Dict[str, np.ndarray]:
    f = {"original": img}
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 28)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 22)
    f["hsv"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    lab_clahe = cv2.merge((clahe.apply(l), a, b))
    f["clahe"] = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    rgb = img.copy()
    rgb[:, :, 2] = cv2.add(rgb[:, :, 2], 35)
    rgb[:, :, 1] = cv2.subtract(rgb[:, :, 1], 8)
    f["rgb"] = rgb

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    f["gray"] = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
    return f

def draw_label(img, text, x, y, bg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, 0.5, 1)
    y = max(h + 4, y)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, 0.5, (255, 255, 255), 1)

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    ix1, iy1 = max(x1, X1), max(y1, Y1)
    ix2, iy2 = min(x2, X2), min(y2, Y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def skin_mask_ratio(img, box):
    x1, y1, x2, y2 = map(int, box)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, SKIN_LOWER_1, SKIN_UPPER_1)
    mask2 = cv2.inRange(hsv, SKIN_LOWER_2, SKIN_UPPER_2)
    return float(np.bitwise_or(mask1, mask2).mean() / 255.0)

def compute_face_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, SKIN_LOWER_1, SKIN_UPPER_1)
    mask2 = cv2.inRange(hsv, SKIN_LOWER_2, SKIN_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask.astype(np.float32) / 255.0

def encode_image_to_base64(img):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("encode gagal")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ============================================================
#  SEVERITY CALCULATION
# ============================================================

def compute_severity_percentage(conf, skin_r, contrast, w, h, img_w, img_h):
    area = float(w * h)
    img_area = float(img_w * img_h)
    size_norm = min(1.0, area / (img_area * 0.12 + 1e-6))
    contrast_norm = min(1.0, contrast / 35.0)
    skin_r = float(np.clip(skin_r, 0.0, 1.0))

    score = (
        conf * 0.45 +
        skin_r * 0.15 +
        contrast_norm * 0.20 +
        size_norm * 0.20
    )
    return round(float(np.clip(score, 0.0, 1.0)) * 100.0, 2)

def severity_from_percentage(pct):
    if pct < 20:
        return "sangat ringan"
    elif pct < 40:
        return "ringan"
    elif pct < 60:
        return "sedang"
    elif pct < 80:
        return "berat"
    return "sangat berat"


# ============================================================
#  WBF + SOFT-NMS
# ============================================================
def wbf_fusion(
    boxes: List[List[float]],
    labels: List[str],
    confs: List[float],
    iou_thr: float = IOU_MERGE,
) -> List[Tuple[List[float], str, float]]:
    fused: List[Tuple[List[float], str, float]] = []
    used = [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue

        group = [i]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if labels[i] == labels[j] and iou(boxes[i], boxes[j]) > iou_thr:
                used[j] = True
                group.append(j)

        ws = np.array([confs[k] for k in group])
        ws = np.power(ws, 1.2)
        ws = ws / (ws.sum() + 1e-9)

        coords = np.array([boxes[k] for k in group])
        avg = (coords * ws[:, None]).sum(axis=0)

        c = 1.0
        for k in group:
            c *= (1.0 - confs[k])
        c = 1.0 - c

        fused.append((avg.tolist(), labels[i], float(c)))

    return fused


def soft_nms(
    boxes: List[List[float]],
    labels: List[str],
    confs: List[float],
    iou_thr: float = SOFT_NMS_IOU,
    decay: float = SOFT_NMS_DECAY,
) -> List[Tuple[List[float], str, float]]:
    if not boxes:
        return []

    order = np.argsort(confs)[::-1].tolist()
    boxes = [boxes[i] for i in order]
    labels = [labels[i] for i in order]
    confs = [confs[i] for i in order]

    keep: List[Tuple[List[float], str, float]] = []

    while boxes:
        b0, l0, c0 = boxes[0], labels[0], confs[0]
        keep.append((b0, l0, c0))

        newb, newl, newc = [], [], []
        for i in range(1, len(boxes)):
            if labels[i] == l0 and iou(b0, boxes[i]) > iou_thr:
                confs[i] *= decay
            if confs[i] >= 0.05:
                newb.append(boxes[i])
                newl.append(labels[i])
                newc.append(confs[i])

        boxes, labels, confs = newb, newl, newc

    return keep


# ============================================================
#  HEATMAP ORGANIK (GAUSSIAN BLOB + FACE MASK)
# ============================================================
def add_blob_heatmap(
    heat_img: np.ndarray,
    box,
    severity: str,
    weight: float = 1.0,
    skin_mask: Optional[np.ndarray] = None,
) -> None:
    x1, y1, x2, y2 = map(int, box)
    H, W = heat_img.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w < 4 or roi_h < 4:
        return

    xs = np.linspace(-1, 1, roi_w, dtype=np.float32)
    ys = np.linspace(-1, 1, roi_h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    sigma2 = 0.55 ** 2
    gauss = np.exp(-(X**2 + Y**2) / (2.0 * sigma2))
    if gauss.max() > 0:
        gauss /= gauss.max()

    if skin_mask is not None:
        skin_roi = skin_mask[y1:y2, x1:x2]
        if skin_roi.shape[:2] != gauss.shape:
            skin_roi = cv2.resize(skin_roi, (roi_w, roi_h))
        gauss *= skin_roi

    weight = max(0.1, float(weight))
    gauss *= weight

    col = SEV_COL.get(severity, SEV_COL["ringan"])
    r, g, b = float(col[2]) / 255.0, float(col[1]) / 255.0, float(col[0]) / 255.0

    blob = np.stack([
        gauss * b,
        gauss * g,
        gauss * r,
    ], axis=-1)

    heat_img[y1:y2, x1:x2, :] += blob

def finalize_heatmap(base_img: np.ndarray, heat_raw: np.ndarray) -> np.ndarray:
    maxv = float(heat_raw.max())
    if maxv <= 0:
        return base_img.copy()

    heat_norm = heat_raw / (maxv + 1e-6)
    heat_norm = np.clip(heat_norm, 0.0, 1.0)
    heat_norm = np.power(heat_norm, 0.9)
    heat_blur = cv2.GaussianBlur(heat_norm, (0, 0), sigmaX=32, sigmaY=32)
    heat_u8 = np.clip(heat_blur * 255.0, 0, 255).astype(np.uint8)
    blended = cv2.addWeighted(base_img, 0.65, heat_u8, 0.9, 0)
    return blended


# ============================================================
#  YOLO INIT
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

print(f"[INIT] Load model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INIT] Model loaded.")

def run_predict(source):
    return model.predict(
        source=source,
        conf=BASE_CONF,
        imgsz=IMG_SIZE,
        device=DEVICE,
        half=(DEVICE != "cpu"),
        augment=False,
        verbose=False,
    )

# ============================================================
#  FUNGSI UTAMA ANALISA SATU GAMBAR (BOOSTED + ADAPTIVE)
# ============================================================
def analyze_image_bytes(image_bytes: bytes, pose: str, session_id: str) -> Dict[str, Any]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid / gagal di-decode")

    img = remove_background(img, bg_color=(255, 255, 255))
    img = adaptive_color_balance(img)
    H, W = img.shape[:2]

    face_mask = compute_face_mask(img)

    variants: List[Tuple[str, np.ndarray]] = []

    filters = apply_filters(img)
    for k, fimg in filters.items():
        name = f"f:{k}" if k != "original" else "original"
        variants.append((name, fimg))

    scales = [0.75, 1.0, 1.25]
    for s in scales:
        new_w, new_h = int(W * s), int(H * s)
        scaled = cv2.resize(img, (new_w, new_h))
        variants.append((f"s:{s:.2f}", scaled))

    variants.append(("flipH", cv2.flip(img, 1)))

    tw, th = W // 2, H // 2
    ov = 0.2
    dx, dy = int(tw * (1 - ov)), int(th * (1 - ov))
    tiles: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
    for y in [0, max(0, th - dy)]:
        for x in [0, max(0, tw - dx)]:
            x2, y2 = min(W, x + tw), min(H, y + th)
            tiles.append(((x, y, x2, y2), img[y:y2, x:x2]))
    for i, (bx, tile) in enumerate(tiles):
        variants.append((f"tile{i}", tile))

    all_boxes: List[List[float]] = []
    all_labels: List[str] = []
    all_confs: List[float] = []

    for name, vimg in variants:
        res_list = run_predict(vimg)

        for r in res_list:
            for b in r.boxes:
                cls = int(b.cls[0])
                raw_label = model.names[cls]
                label = MODEL_LABEL_TO_APP.get(raw_label, raw_label)
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [int(x) for x in b.xyxy[0]]

                if name.startswith("tile"):
                    idx = int(name.replace("tile", ""))
                    (tx1, ty1, _, _) = tiles[idx][0]
                    x1 += tx1
                    x2 += tx1
                    y1 += ty1
                    y2 += ty1
                elif name == "flipH":
                    x1, x2 = W - x2, W - x1
                elif name.startswith("s:"):
                    s = float(name.split(":")[1])
                    x1, y1, x2, y2 = int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W - 1, x2), min(H - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                skin_r = skin_mask_ratio(img, (x1, y1, x2, y2))
                if skin_r < 0.35:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                contrast = gray.std()
                if contrast < 10:
                    continue

                base_th = PER_CLASS_TH.get(label, 0.15)
                dyn_th = adaptive_threshold(label, base_th, skin_r)
                if conf < dyn_th:
                    continue

                weight = VARIANT_WEIGHTS.get(name, 1.0)
                boosted_conf = conf * weight

                all_boxes.append([x1, y1, x2, y2])
                all_labels.append(label)
                all_confs.append(boosted_conf)

    if all_boxes:
        fused = wbf_fusion(all_boxes, all_labels, all_confs, IOU_MERGE)
    else:
        fused = []

    if fused:
        f_boxes, f_labels, f_confs = list(zip(*fused))
        snms = soft_nms(list(f_boxes), list(f_labels), list(f_confs),
                        SOFT_NMS_IOU, SOFT_NMS_DECAY)
    else:
        snms = []

    ts = datetime.now().isoformat()
    detections: List[Dict[str, Any]] = []

    summary_conf: Dict[str, List[float]] = {}
    summary_pct: Dict[str, List[float]] = {}

    overlay = img.copy()
    heat_raw = np.zeros_like(img, np.float32)

    severity_counts = {
        "sangat ringan": 0, "ringan": 0, "sedang": 0, "berat": 0, "sangat berat": 0,
    }

    for box, lbl, conf in snms:
        x1, y1, x2, y2 = [int(x) for x in box]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue

        roi = img[y1:y2, x1:x2]
        if roi.size == 0: continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        skin_r = skin_mask_ratio(img, (x1, y1, x2, y2))

        base_th = PER_CLASS_TH.get(lbl, 0.15)
        dyn_th = adaptive_threshold(lbl, base_th, skin_r)
        if conf < dyn_th: continue

        robustness = float(round(conf * skin_r * (contrast / 30.0), 3))
        if robustness < 0.05: continue

        severity_pct = compute_severity_percentage(conf, skin_r, float(contrast), w, h, W, H)
        sev = severity_from_percentage(severity_pct)
        col = SEV_COL.get(sev, SEV_COL["ringan"])
        
        scan_dt = parse_session_datetime(session_id)
        scan_iso = scan_dt.isoformat() if scan_dt else None

        detections.append({
            "label": lbl, "confidence": round(float(conf), 4), "severity": sev,
            "severity_percentage": severity_pct, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": w, "height": h, "skin_ratio": float(round(skin_r, 3)),
            "contrast": float(round(float(contrast), 3)), "robustness": robustness,
            "timestamp": ts, "scanned_at": scan_iso,
        })

        summary_conf.setdefault(lbl, []).append(float(conf))
        summary_pct.setdefault(lbl, []).append(float(severity_pct))

        if sev in severity_counts: severity_counts[sev] += 1

        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, 2)
        draw_label(overlay, f"{lbl} {conf:.2f}", x1 + 3, max(y1 - 6, 15), col)

        add_blob_heatmap(heat_raw, (x1, y1, x2, y2), sev, weight=robustness, skin_mask=face_mask)

    heatmap_img = finalize_heatmap(img, heat_raw)

    summary_stats: List[Dict[str, Any]] = []
    for k in sorted(summary_conf.keys()):
        confs = summary_conf.get(k, [])
        pcts = summary_pct.get(k, [])
        if not confs: continue
        avg_conf = float(np.mean(confs))
        avg_pct = float(np.mean(pcts)) if pcts else 0.0
        sev = severity_from_percentage(avg_pct)
        summary_stats.append({
            "label": k, "count": len(confs), "avg_confidence": avg_conf,
            "severity": sev, "avg_severity_percentage": avg_pct,
        })

    if not detections:
        overlay = img.copy()
        heatmap_img = img.copy()

    out_dir = os.path.join(OUTPUT_ROOT, session_id, pose)
    ensure_dir(out_dir)

    det_path = os.path.join(out_dir, "detections.json")
    with open(det_path, "w", encoding="utf-8") as f:
        json.dump({
            "pose": pose, "timestamp": ts, "detections": detections,
            "summary": summary_stats, "severity_counts": severity_counts,
        }, f, ensure_ascii=False, indent=2)

    overlay_path = os.path.join(out_dir, "overlay.jpg")
    heatmap_path = os.path.join(out_dir, "heatmap.jpg")
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(heatmap_path, heatmap_img)

    overlay_b64 = encode_image_to_base64(overlay)
    heatmap_b64 = encode_image_to_base64(heatmap_img)

    return {
        "pose": pose, "detections": detections, "summary": summary_stats,
        "severity_counts": severity_counts,
        "files": {
            "folder": out_dir, "overlay": "overlay.jpg", "heatmap": "heatmap.jpg",
            "detections_json": "detections.json",
        },
        "overlay_image": {"base64": overlay_b64, "mime": "image/jpeg"},
        "heatmap_image": {"base64": heatmap_b64, "mime": "image/jpeg"},
    }

# ============================================================
#  FUNGSI NOTIFIKASI
# ============================================================

def send_email_notification(email: str, name: str, session_id: str, health_data: Dict, processed_poses: List[str]):
    """Mengirim email hasil analisis dengan lampiran gambar."""
    if not email: return
    
    score = health_data.get("health_score", 0)
    status = health_data.get("health_status", "Tidak Diketahui")
    concerns = ", ".join(health_data.get("main_concerns", ["Tidak ada"]))

    subject = f"Hasil Analisis Kulit MirraSense - Skor {score}"
    
    body_text = f"""
    Hai {name},
    
    Terima kasih telah melakukan scan kulit di MirraSense!
    
    Berikut adalah ringkasan hasil analisis kamu:
    --------------------------------------------------
    Session ID : {session_id}
    Skor Kesehatan : {score}/100
    Status : {status}
    Masalah Utama : {concerns}
    --------------------------------------------------
    
    Kami lampirkan hasil foto analisis pada email ini.
    
    Salam Sehat,
    Tim MirraSense
    """

    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body_text, 'plain'))
        
        desired_order = ['right', 'center', 'left']
        
        for pose in desired_order:
            if pose in processed_poses:
                file_path = os.path.join(OUTPUT_ROOT, session_id, pose, "overlay.jpg")
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        img_data = f.read()
                        image_attachment = MIMEImage(img_data, name=f"result_{pose}.jpg")
                        image_attachment.add_header('Content-Disposition', f'attachment; filename="result_{pose}.jpg"')
                        msg.attach(image_attachment)
                        print(f"[EMAIL] Attaching image: {file_path}")

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"[EMAIL] Berhasil terkirim ke {email}")
    except Exception as e:
        print(f"[EMAIL] Gagal kirim: {e}")

def send_wa_notification(phone: str, name: str, session_id: str, health_data: Dict):
    """Mengirim notifikasi WhatsApp via Gateway API."""
    if not phone: return

    clean_phone = phone.replace("+", "").replace("-", "").replace(" ", "")
    if clean_phone.startswith("0"): clean_phone = "62" + clean_phone[1:]

    score = health_data.get("health_score", 0)
    message = f"Hai {name}! ✨\n\nHasil scan kulitmu (ID: {session_id}) sudah siap.\nSkor Kesehatan: {score}/100.\n\nCek email kamu untuk melihat foto hasil analisis."

    payload = {"target": clean_phone, "message": message}
    headers = {"Authorization": WA_API_TOKEN}

    try:
        # requests.post(WA_API_URL, data=payload, headers=headers)
        print(f"[WA] Simulasi terkirim ke {clean_phone}")
    except Exception as e:
        print(f"[WA] Gagal kirim: {e}")


# ============================================================
#  EVALUASI KESEHATAN KULIT
# ============================================================

HIGH_CONCERN_LABELS = {
    "Jerawat", "Papula", "Pustula", "Nodul Jerawat", "Jerawat Kistik",
    "Bekas Jerawat", "Folikulitis", "Keloid", "Siringoma", "Kutil Datar",
    "Flek Hitam", "Kemerahan Kulit",
}

def evaluate_skin_health(all_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not all_detections:
        return {
            "is_healthy": True, "health_status": "sehat", "health_score": 100.0,
            "severity_score": 0.0, "main_concerns": [],
            "stats": {"total_detections": 0, "high_concern_count": 0, "severe_lesions_count": 0, "per_label_counts": {}},
            "message": "Tidak ada temuan signifikan, kulit tampak sehat.",
        }

    severity_score = sum(d.get("severity_percentage", 0.0) for d in all_detections) / len(all_detections)
    label_counts: Dict[str, int] = {}
    for d in all_detections:
        lbl = d.get("label", "Unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    main_concerns = sorted(label_counts.keys(), key=lambda k: label_counts[k], reverse=True)
    high_concern_count = sum(1 for d in all_detections if d.get("label") in HIGH_CONCERN_LABELS)
    severe_lesions_count = sum(1 for d in all_detections if d.get("severity_percentage", 0.0) >= 60.0)
    total_detections = len(all_detections)

    is_healthy = severity_score <= 25.0 and severe_lesions_count == 0 and high_concern_count == 0

    base_health = max(0.0, 100.0 - severity_score)
    penalty = 0.0
    if high_concern_count > 0: penalty += min(20.0, 5.0 * high_concern_count)
    if severe_lesions_count > 0: penalty += min(20.0, 3.0 * severe_lesions_count)

    health_score = max(0.0, base_health - penalty)
    if is_healthy: health_score = min(100.0, health_score + 10.0)

    health_status = "sehat" if is_healthy else "kurang sehat"
    msg = "Kulit dikategorikan sehat." if is_healthy else "Kulit dikategorikan perlu perhatian."

    return {
        "is_healthy": is_healthy, "health_status": health_status,
        "health_score": round(health_score, 2), "severity_score": round(severity_score, 2),
        "main_concerns": main_concerns,
        "stats": {
            "total_detections": total_detections, "high_concern_count": high_concern_count,
            "severe_lesions_count": severe_lesions_count, "per_label_counts": label_counts,
        },
        "message": msg,
    }

def build_severity_map(all_dets):
    severity_map = {}
    rank = ["sangat ringan", "ringan", "sedang", "berat", "sangat berat"]
    for d in all_dets:
        lbl = d.get("label")
        sev = d.get("severity")
        if lbl and sev:
            old = severity_map.get(lbl)
            if (not old) or (rank.index(sev) > rank.index(old)):
                severity_map[lbl] = sev
    return severity_map

def parse_session_datetime(session_id: str) -> Optional[datetime]:
    try:
        date_part, time_part, *_ = session_id.split("_")
        dt = datetime.strptime(date_part + time_part[:6], "%Y%m%d%H%M%S")
        return dt
    except: return None

# ============================================================
#  API ENDPOINTS
# ============================================================

# 1. ENDPOINT REGISTER USER
@app.post("/users/register")
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """
    Endpoint khusus untuk menyimpan data user ke database.
    Bisa dipanggil terpisah sebelum/tidak dengan proses analisis.
    """
    try:
        new_user = User(
            name=user.name,
            email=user.email,
            tel=user.tel,
            created_at=datetime.now()
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        return {
            "success": True,
            "message": "User berhasil disimpan",
            "data": {
                "id": new_user.id,
                "name": new_user.name,
                "email": new_user.email,
                "tel": new_user.tel,
                "created_at": new_user.created_at.isoformat()
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# 2. ENDPOINT ANALISIS WAJAH
@app.post("/analyze-face")
async def analyze_face(
    background_tasks: BackgroundTasks,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    left: Optional[UploadFile] = File(default=None),
    center: Optional[UploadFile] = File(default=None),
    right: Optional[UploadFile] = File(default=None),
):
    if not any([left, center, right]):
        raise HTTPException(status_code=400, detail="Minimal kirim 1 file gambar.")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ensure_dir(os.path.join(OUTPUT_ROOT, session_id))

    poses_input = {"left": left, "center": center, "right": right}
    poses_output = {}
    all_detections = []
    
    processed_poses_keys = []

    for pose_name, file in poses_input.items():
        if file is None: continue
        content = await file.read()
        try:
            result = analyze_image_bytes(content, pose_name, session_id)
            poses_output[pose_name] = result
            all_detections.extend(result.get("detections", []))
            processed_poses_keys.append(pose_name)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Gambar pose {pose_name} tidak valid: {e}")

    eva = evaluate_skin_health(all_detections)

    # Jalankan notifikasi di background
    if email:
        background_tasks.add_task(send_email_notification, email, name, session_id, eva, processed_poses_keys)
    if phone:
        background_tasks.add_task(send_wa_notification, phone, name, session_id, eva)

    response = {
        "success": True,
        "processed_at": datetime.now().isoformat(),
        "session_id": session_id,
        "user": {"name": name, "email": email, "phone": phone},
        "poses": poses_output,
        "health_evaluation": eva
    }
    return JSONResponse(content=response)


# ============================================================
#  HISTORY & HELPER ENDPOINTS
# ============================================================

def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return None

def parse_session_date(session_id: str) -> Optional[date]:
    try: return datetime.strptime(session_id[:10], "%Y-%m-%d").date()
    except: pass
    try: return datetime.strptime(session_id[:10], "%Y_%m_%d").date()
    except: pass
    m = re.match(r"^(\d{4})(\d{2})(\d{2})", session_id)
    if m:
        y, mo, d = m.groups()
        return date(int(y), int(mo), int(d))
    return None

def get_all_session_ids() -> List[str]:
    if not os.path.exists(OUTPUT_ROOT): return []
    sessions = [s for s in os.listdir(OUTPUT_ROOT) if os.path.isdir(os.path.join(OUTPUT_ROOT, s))]
    return sorted(sessions, reverse=True)

@app.get("/history/list")
async def history_list():
    sessions = get_all_session_ids()
    return {"success": True, "count": len(sessions), "sessions": sessions}

@app.get("/history/file/{session_id}/{pose}/{filename}")
async def history_file(session_id: str, pose: str, filename: str):
    filepath = os.path.join(OUTPUT_ROOT, session_id, pose, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

@app.get("/skin-health/{session_id}")
async def skin_health(session_id: str):
    session_path = os.path.join(OUTPUT_ROOT, session_id)
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail="Session not found")

    all_detections = []
    for pose in ["left", "center", "right"]:
        det_path = os.path.join(session_path, pose, "detections.json")
        det = load_json(det_path)
        if det: all_detections.extend(det.get("detections", []))

    eva = evaluate_skin_health(all_detections)
    severity_map = build_severity_map(all_detections)
    
    return {
        "session_id": session_id, "success": True, **eva,
        "rekomendasi_produk": generate_produk_rekomendasi(eva["main_concerns"], severity_map)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("facial_api:app", host="0.0.0.0", port=8000, reload=True)