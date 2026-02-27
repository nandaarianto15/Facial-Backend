# Panduan Instalasi & Menjalankan Aplikasi
Ikuti langkah-langkah berikut untuk menjalankan proyek ini di lingkungan lokal Anda:

## 1. Membuat Virtual Environment
Buat environment virtual untuk mengisolasi dependencies proyek.
```bash
python -m venv venv
```

## 2. Mengaktifkan Virtual Environment
Aktifkan environment yang telah dibuat.
```bash
source venv/bin/activate
```

## 3. Install Requirements
Pastikan pip sudah diperbarui, lalu install semua library yang dibutuhkan.
```bash
pip install --upgrade pip
pip install -r requirement.txt
```

## 4. Menjalankan Server
Jalankan server Uvicorn di background menggunakan perintah berikut:
```bash
nohup uvicorn main:app --host 0.0.0.0 --port 8000 &
```
Aplikasi akan berjalan dan dapat diakses pada http://0.0.0.0:8000 atau http://localhost:8000.
