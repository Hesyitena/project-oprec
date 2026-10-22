# Final Project Oprec PENSHIP — Kode Training Dasar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hesyitena/project-oprec/blob/main/train_colab.ipynb)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)

Repository ini berisi kode dasar untuk melatih model object detection **YOLOv8** yang mendeteksi 3 kelas objek: `bottle`, `laptop`, dan `pencil`.

> [!IMPORTANT]
> Kode ini sengaja dibuat **sangat dasar** sebagai titik awal. Kamu **wajib mengembangkannya lagi** (misalnya melalui tuning hyperparameter, augmentasi data, analisis hasil, atau penambahan skrip inferensi). Detail aturan lengkap dapat dibaca pada **Guidebook Final Project Oprec PENSHIP**.

---

## Daftar Isi
- [Pilihan Lingkungan Training](#pilihan-lingkungan-training)
- [Struktur Proyek](#struktur-proyek)
- [Download / Clone Repository](#download--clone-repository)
- [Persiapan Konfigurasi Roboflow](#persiapan-konfigurasi-roboflow)
- [Cara Menjalankan](#cara-menjalankan)
  - [Opsi A: Google Colab (Direkomendasikan)](#opsi-a-google-colab-direkomendasikan)
  - [Opsi B: Komputer Lokal (Windows / Linux / macOS)](#opsi-b-komputer-lokal-windows--linux--macos)
- [Struktur Output Training](#struktur-output-training)
- [Panduan Pengembangan Kode](#panduan-pengembangan-kode)
- [Template Laporan Pengumpulan](#template-laporan-pengumpulan)

---

## Pilihan Lingkungan Training

| File | Lingkungan | Rekomendasi Penggunaan |
| --- | --- | --- |
| [`train_colab.ipynb`](train_colab.ipynb) | Google Colab | **Sangat disarankan** jika laptop tidak memiliki GPU NVIDIA atau spesifikasi terbatas. Tersedia free GPU T4. |
| [`train.py`](train.py) + [`requirements.txt`](requirements.txt) | Laptop / PC Lokal | Cocok jika memiliki GPU NVIDIA mandiri dengan driver CUDA aktif. |

---

## Struktur Proyek

```text
project-oprec/
├── .gitattributes      # Konfigurasi normalisasi Git (line endings & text diff)
├── .gitignore          # Daftar file/folder yang diabaikan Git (dataset, checkpoint, runs, dll.)
├── requirements.txt    # Daftar dependensi Python utama (ultralytics, roboflow)
├── train.py            # Skrip training dasar untuk eksekusi di komputer lokal
├── train_colab.ipynb   # Jupyter Notebook training untuk Google Colab
├── TEMPLATE_LAPORAN.md # Kerangka laporan; isinya disalin ke README.md repo-mu
└── README.md           # Dokumentasi dan panduan proyek
```

---

## Download / Clone Repository

Jangan kerjakan langsung di repository ini. Buat repository-mu sendiri dulu:

1. Klik **Use this template → Create a new repository** di halaman ini.
2. Beri nama `FP_<NamaLengkap>_<NRP>` dan pilih **Public**.
3. Download repository-mu lewat **Code → Download ZIP**, atau clone:

```bash
git clone https://github.com/<username>/FP_<NamaLengkap>_<NRP>.git
cd FP_<NamaLengkap>_<NRP>
```

---

## Persiapan Konfigurasi Roboflow

Sebelum menjalankan training, siapkan konfigurasi dataset dari Roboflow:

1. Buka dataset kamu di [Roboflow](https://app.roboflow.com).
2. Perhatikan URL dataset: `app.roboflow.com/<workspace>/<project>/<version>` atau buka menu **Download Dataset → YOLOv8 → Show download code**.
3. Dapatkan **Roboflow API Key** dari menu: [https://app.roboflow.com/settings/api](https://app.roboflow.com/settings/api).

Sesuaikan nilai variabel konfigurasi di [`train.py`](train.py) atau cell **4. Konfigurasi** di notebook:

```python
ROBOFLOW_WORKSPACE = "nama-workspace"  # Ganti dengan workspace-mu
ROBOFLOW_PROJECT   = "nama-project"    # Ganti dengan project-mu
ROBOFLOW_VERSION   = 1                 # Nomor versi dataset
MODEL              = "yolov8n.pt"      # Pilihan: yolov8n.pt atau yolov8s.pt
```

> [!WARNING]
> **Keamanan Kredensial**: JANGAN PERNAH menuliskan API key langsung di dalam file kode (`train.py`) atau notebook (`train_colab.ipynb`), terutama jika kamu mengunggah atau membagikan repository ini secara publik. Gunakan Environment Variable atau Colab Secrets.

---

## Cara Menjalankan

### Opsi A: Google Colab (Direkomendasikan)

1. Buka [https://colab.research.google.com/](https://colab.research.google.com/), pilih tab **Upload**, lalu upload `train_colab.ipynb` dari repository-mu. Badge **Open in Colab** di atas hanya membuka notebook asli milik panitia, bukan versi yang sudah kamu kembangkan.
2. Pastikan GPU aktif: menu **Runtime → Change runtime type → T4 GPU → Save**.
3. Masukkan `ROBOFLOW_API_KEY` saat diminta, atau simpan di fitur **Secrets (ikon kunci)** Colab dengan nama `ROBOFLOW_API_KEY`.
4. Jalankan cell secara berurutan (**Shift + Enter**) dari langkah 1 sampai langkah 10.
5. Di akhir sesi, cell nomor 10 akan otomatis mengompres folder output menjadi `runs.zip` dan mengunduh `best.pt` serta grafik hasil ke komputermu.

---

### Opsi B: Komputer Lokal (Windows / Linux / macOS)

#### 1. Persiapan Virtual Environment

**Windows (PowerShell):**
```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Jika laptop memiliki GPU NVIDIA, instal PyTorch versi CUDA terlebih dahulu.
#    GTX 9xx/10xx: cu126. GTX 16xx, RTX 20xx-50xx: cu132 (lihat tabel di guidebook).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. Instal dependensi proyek
pip install -r requirements.txt
```

Kalau aktivasi venv ditolak karena *execution policy*, jalankan dulu `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` di jendela PowerShell yang sama.

**Linux / macOS / WSL (Bash):**
```bash
# 1. Buat dan aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Instal dependensi proyek
pip install -r requirements.txt
```

#### 2. Jalankan Training

**Windows (PowerShell):**
```powershell
$env:ROBOFLOW_API_KEY="API_KEY_KAMU"
python train.py
```

**Linux / macOS / WSL (Bash):**
```bash
export ROBOFLOW_API_KEY="API_KEY_KAMU"
python train.py
```

---

## Struktur Output Training

Hasil proses training akan tersimpan di dalam folder `runs/detect/train/` (atau `train2/`, `train3/`, dst.):

- `weights/best.pt`: Bobot model dengan performa validation mAP tertinggi. File ini yang digunakan untuk pengujian dan inferensi final.
- `weights/last.pt`: Bobot model pada epoch terakhir training.
- `results.png`: Grafik metrik performa (loss training/validation, precision, recall, mAP50, mAP50-95).
- `confusion_matrix.png`: Matriks evaluasi prediksi per kelas (`bottle`, `laptop`, `pencil`).
- `results.csv`: Catatan numerik metrik performa per epoch.

---

## Panduan Pengembangan Kode

Kode starter ini sengaja dibuat sederhana. Batasan yang tetap berlaku sesuai aturan oprec:
- Arsitektur model: **YOLOv8n** atau **YOLOv8s** (pre-trained).
- Resolusi input gambar: **`imgsz=640`**.

### Contoh Arah Pengembangan:
1. **Hyperparameter Tuning**: Eksplorasi nilai `epochs`, `batch` (16, 8, 4), `lr0` (learning rate awal), `optimizer` (`AdamW`, `SGD`), dan `patience` (early stopping).
2. **Augmentasi Data**: Uji kombinasi parameter augmentasi Ultralytics seperti `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `fliplr`, `mosaic`, atau `mixup`.
3. **Eksperimen Terukur**: Catat dan bandingkan beberapa skenario percobaan dalam satu tabel perbandingan mAP.
4. **Skrip Inferensi Demo**: Buat skrip inferensi tambahan (`predict.py`) untuk menguji deteksi pada foto baru, file video, atau webcam langsung (dapat dipakai untuk membuat video rekaman `demo.mp4`).
5. **Analisis Hasil**: Tampilkan contoh gambar deteksi, evaluasi bagian yang mengalami *false positive* atau *false negative*, dan jelaskan bagaimana kamu mengatasinya.
6. **Kerapian Kode & Ekspor**: Implementasikan `argparse` / config YAML untuk konfigurasi modular, atau ekspor model ke format ONNX (`model.export(format="onnx")`).

> Referensi lengkap parameter training Ultralytics: [Ultralytics Train Documentation](https://docs.ultralytics.com/modes/train/)

---

## Cara Pengumpulan

1. Kerjakan semuanya di repository-mu sendiri hasil **Use this template** (Public).
2. Ganti seluruh isi `README.md` di repository-mu dengan isi [`TEMPLATE_LAPORAN.md`](TEMPLATE_LAPORAN.md), lalu isi tiap bagiannya. File `TEMPLATE_LAPORAN.md` boleh dihapus setelahnya.
3. Upload ke repository: kode yang sudah dikembangkan, `best.pt`, dan folder `grafik/` (`results.png`, `results.csv`, `confusion_matrix.png`, `BoxPR_curve.png` dari `runs/detect/train/`). Lewat web: **Add file → Upload files** (maks. 25 MB per file).
4. Upload video demo (30–60 detik) ke Google Drive dengan akses **Anyone with the link → Viewer**. Video tidak di-upload ke repository.
5. Isi Google Form pengumpulan dengan link repository GitHub, link video demo, dan Project ID Roboflow.

Detail lengkap dan checklist ada di **Guidebook Final Project Oprec PENSHIP**.
