import json
import os

RECO_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "rekomendasi_produk.json")

def load_rekomendasi():
    if not os.path.exists(RECO_FILE):
        print("WARNING: rekomendasi_produk.json tidak ditemukan!")
        return {}

    with open(RECO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

REKOMENDASI_PRODUK = load_rekomendasi()
