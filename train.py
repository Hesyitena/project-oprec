"""
Kode training dasar - Final Project Oprec PENSHIP

Alur:
  1. Download dataset dari project Roboflow peserta (format YOLOv8).
     Apabila folder dataset/ sudah berisi data.yaml, download dilewati.
  2. Training YOLOv8.
  3. Evaluasi model terbaik (best.pt) di split validation.

Persiapan:
  1. Isi ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, dan ROBOFLOW_VERSION di bagian
     KONFIGURASI. Ketiganya tertera pada URL dataset:
         app.roboflow.com/<workspace>/<project>/<version>
     atau pada potongan kode di menu Download Dataset -> format YOLOv8 -> Show download code.
  2. Set API key Roboflow sebagai environment variable. API key DILARANG
     dituliskan di file ini, karena file ini ikut dikumpulkan.

Menjalankan (ganti API_KEY_ROBOFLOW dengan API key Roboflow peserta):
  Windows (PowerShell):
      pip install -r requirements.txt
      $env:ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
      python train.py

  Linux / WSL / macOS:
      pip install -r requirements.txt
      export ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
      python train.py

Untuk training di Google Colab, gunakan train_colab.ipynb.

Kode ini sengaja dibuat sangat dasar dan WAJIB dikembangkan oleh peserta.
Contoh arah pengembangan ada di README dan guidebook (bagian Pengembangan kode).
"""

import os
from pathlib import Path

from ultralytics import YOLO

# ===================== KONFIGURASI =====================
# Dataset (Roboflow)
ROBOFLOW_WORKSPACE = "nama-workspace"  # sesuaikan dengan URL dataset
ROBOFLOW_PROJECT = "nama-project"      # sesuaikan dengan URL dataset
ROBOFLOW_VERSION = 1                   # nomor versi dataset di Roboflow
DATASET_DIR = "dataset"                # folder tujuan download

# Training
MODEL = "yolov8n.pt"  # pilihan: yolov8n.pt atau yolov8s.pt
EPOCHS = 50
IMGSZ = 640           # wajib 640, jangan diubah
BATCH = 16            # turunkan ke 8 atau 4 apabila muncul error out of memory
SEED = 0
# =======================================================


def get_dataset() -> Path:
    """Kembalikan path data.yaml. Download dari Roboflow apabila belum ada."""
    data_yaml = Path(DATASET_DIR) / "data.yaml"
    if data_yaml.exists():
        print(f"Dataset sudah ada, download dilewati: {data_yaml}")
        return data_yaml

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY belum di-set. Lihat bagian 'Menjalankan' di atas file ini."
        )

    from roboflow import Roboflow  # di-import di sini: tidak wajib apabila dataset sudah ada

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT).version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8", location=DATASET_DIR)

    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml tidak ditemukan setelah download: {data_yaml}")
    return data_yaml


def main():
    data_path = get_dataset()

    # Model pretrained (bobot di-download otomatis saat pertama kali dipakai)
    model = YOLO(MODEL)

    # Training. GPU digunakan otomatis apabila tersedia; jika tidak, CPU yang digunakan.
    model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
    )

    # Lokasi hasil training (runs/detect/train, train2, ...)
    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    print(f"\nModel terbaik tersimpan di: {best_path}")

    # Evaluasi model terbaik di split validation
    best_model = YOLO(str(best_path))
    metrics = best_model.val(data=str(data_path), imgsz=IMGSZ)
    print(f"mAP50-95: {metrics.box.map:.3f} | mAP50: {metrics.box.map50:.3f}")


# Guard ini wajib, terutama di Windows (dataloader memakai multiprocessing)
if __name__ == "__main__":
    main()
