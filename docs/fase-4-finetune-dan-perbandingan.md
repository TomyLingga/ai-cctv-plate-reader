# Fase 4: fine-tune, perbandingan dengan baseline, dan kecepatan

Dikerjakan 15 November 2026 di laptop pengembangan (training) dan PC ROG (pengukuran kecepatan). Dataset dan cara membuatnya ada di [dokumen Fase 3](fase-3-dataset-dan-training.md); dokumen ini memuat apa yang dilatih, hasil ukurnya, dan keputusan yang diambil.

## Ringkasan

| Bagian | Keputusan | Dasar |
|---|---|---|
| OCR | Model baru `models/ocr/finetuned/timbangan-20261115.onnx` dipasang di PC ROG pada 15 November 2026 pukul 09.47. Model awal tetap ada di sana untuk kembali. | Pada 280 crop berlabel manusia yang platnya belum pernah dilihat model: 244 terbaca benar, model awal 228. Salah baca dengan confidence 0,9 ke atas turun dari 25 menjadi 14. Per kendaraan 139 benar dari 154 (model awal 134), tidak ada kendaraan yang menjadi salah. |
| Detektor | Tetap `models/detector/plat.onnx` (YOLOv8n) | YOLO26n hasil training menemukan 76 dari 81 plat pada frame kamera, detektor sekarang 80. YOLO26s terlalu lambat untuk GTX 1060. |
| Kecepatan | Tidak ada yang diubah | Empat kamera sudah diproses 7,5 sampai 8 fps dari target 8. Model OCR baru sama cepatnya dengan yang lama. |

Semua angka akurasi di dokumen ini dihitung pada crop berlabel manusia dari kamera timbangan. Jumlah platnya masih sedikit (54 plat, 154 kendaraan), jadi selisih di bawah kira-kira 5 crop tidak dianggap perbedaan.

## OCR

### Cara mengukur

Evaluasi silang per plat seperti di Fase 3: plat dibagi ke 5 kelompok, model dilatih pada crop plat lain, lalu membaca crop berlabel manusia dari kelompok yang ditahan. Tiap crop dibaca satu kali oleh model yang belum pernah melihat platnya.

Laporannya sekarang dibuat `training/ocr/cv_report.py`. Skrip itu membaca ulang crop yang ditahan dengan model tiap kelompok dan menghitung hasil per kendaraan memakai `PlateVoter`, kelas voting yang dipakai service, tanpa koreksi format. Di Fase 3 hasil per kendaraan masih dihitung dengan pendekatan kasar (bacaan terbanyak berbobot confidence), jadi angkanya berbeda satu kendaraan dari tabel di sana.

Pembanding di semua tabel adalah model dari hub, `models/ocr/cct_s_v2_global.onnx`, yaitu model yang dipakai service. `crossval.py` dan `cv_report.py` memakainya lewat `--baseline`. Model akhir tidak boleh menjadi pembanding di evaluasi silang, karena ia dilatih pada semua plat.

### Yang dicoba

| Nama | Data latih tiap kelompok | Pengaturan |
|---|---|---|
| A | Crop kamera timbangan saja | 25 epoch, lr 0,0005, label smoothing 0, augmentasi bawaan pelatih |
| K | A ditambah 400 crop Kaggle | sama dengan A |
| G2 | Crop kamera ditambah 1.879 crop Kaggle | 12 epoch, lr 0,0005, label smoothing 0,01, augmentasi `training/ocr/augmentation-cctv.yaml` |

Crop Kaggle adalah foto ponsel yang tajam. `augmentation-cctv.yaml` (dibuat `training/ocr/make_augmentation.py`) menurunkan mutunya ke arah crop kamera: gambar diperkecil lalu diperbesar lagi, diburamkan, diberi derau dan kompresi JPEG. Label smoothing 0,1 sempat dipertimbangkan dan tidak dipakai, sebab confidence bacaan akan tertahan di sekitar 0,9 sedangkan ambang event di service 0,85 dan 0,8.

```powershell
uv run --frozen --extra cpu python training/dataset/kaggle_recognition_csv.py `
    --dataset "..\dataset\Indonesian License Plate Recognition Dataset" --out data/dataset/ocr/extra/kaggle_recognition_clean.csv
uv run --frozen --extra cpu python training/ocr/crossval.py --out data/training/ocr/cv-20261115-g2 --epochs 12 --lr 0.0005 `
    --label-smoothing 0.01 --extra-csv data/dataset/ocr/extra/kaggle_recognition_clean.csv --augmentation training/ocr/augmentation-cctv.yaml
```

Satu kelompok G2 memakan 14 sampai 15 menit di CPU laptop.

### Mencampur bobot dengan model awal

Fine-tune pada data sesedikit ini memperbaiki sebagian bacaan dan merusak sebagian lain. Dua cara dicoba untuk menahan kerusakan itu:

- Bobot dicampur: bobot model hasil fine-tune dirata-rata dengan bobot model awal, lalu diekspor sebagai satu berkas (`training/ocr/export_onnx.py --blend-with ... --blend-share 0.5`). Kecepatan dan cara pakainya sama dengan model biasa.
- Ensemble saat runtime: probabilitas dari dua atau tiga model dirata-rata untuk tiap crop. Ini menuntut perubahan kode service, dan tiap model tambahan menambah satu kali waktu OCR.

Hasil pada 280 crop, 154 kendaraan (85 di antaranya punya dua crop atau lebih):

| Model | Crop benar | Diperbaiki / dirusak | Salah dengan confidence ≥ 0,9 | Kendaraan benar | Kendaraan ≥ 2 crop benar | Kendaraan diperbaiki / dirusak |
|---|---|---|---|---|---|---|
| Model awal (hub) | 228 | | 25 | 134 | 81 | |
| A | 241 | 21 / 8 | 27 | 137 | 81 | 7 / 4 |
| K | 242 | 22 / 8 | 30 | 138 | 82 | 7 / 3 |
| G2 | 243 | 24 / 9 | 18 | 138 | 83 | 7 / 3 |
| G2, bobot 30% fine-tune | 241 | 15 / 2 | 13 | 138 | 83 | 4 / 0 |
| **G2, bobot 50% fine-tune** | **244** | **19 / 3** | **14** | **139** | **83** | **5 / 0** |
| G2, bobot 70% fine-tune | 243 | 21 / 6 | 15 | 137 | 83 | 5 / 2 |
| A, bobot 50% fine-tune | 243 | 21 / 6 | 23 | 139 | 82 | 7 / 2 |
| Ensemble model awal + G2 | 245 | 18 / 1 | 10 | 138 | 83 | 4 / 0 |
| Ensemble model awal + A + G2 | 249 | 23 / 2 | 8 | 141 | 83 | 7 / 0 |

Tabel itu dibuat ulang dengan:

```powershell
uv run --frozen --extra cpu python training/ocr/cv_report.py --run A=data/training/ocr/cv-20261115 `
    --run G2=data/training/ocr/cv-20261115-g2 --run G2w50=data/training/ocr/cv-20261115-g2@blend-G2w50.onnx --detail G2w50
uv run --frozen --extra cpu python training/ocr/cv_report.py --run A=data/training/ocr/cv-20261115 --run G2=data/training/ocr/cv-20261115-g2 --ensembles 3
```

Berkas `blend-*.onnx` di tiap folder kelompok adalah `last.keras` kelompok itu yang bobotnya dicampur dengan cara yang sama. Keluarannya tersimpan di `data/training/ocr/cv-report-20261115.txt` dan `cv-report-20261115-ensembles.txt`.

Yang terbaca dari tabel:

- G2 dengan bobot 50% dipilih. Ia memperbaiki 19 crop dan merusak 3, dan ketiga crop yang rusak itu confidence-nya di bawah 0,9 (`BK8981AC` menjadi `BK8981AG`, `BK8021MN` menjadi `BK8021HN`, `BK8092FD` menjadi `BK8092FU`). Per kendaraan 5 diperbaiki dan tidak ada yang rusak.
- Kesalahan pada `BK8486WR` (huruf `W` terbaca `H`, `M`, atau `A`), yang muncul di A, K, dan G2 tanpa campuran, hilang setelah bobot dicampur. `M` terbaca `H` tersisa satu crop; kendaraannya (`BK8021MN`, 9 crop) justru menjadi benar setelah voting.
- Salah baca yang diyakini model turun dari 25 menjadi 14. Fine-tune A menaikkannya menjadi 27; itu salah satu alasan model Fase 3 tidak dipasang.
- Ensemble tiga model sedikit lebih baik (249 crop, 141 kendaraan). Selisih 5 crop terhadap pilihan di atas masih di dalam derau, sedangkan biayanya perubahan kode service dan tiga kali waktu OCR. Tidak dipakai; bisa dipertimbangkan lagi bila data uji sudah lebih banyak.
- Keuntungan terbesar tetap dari satu truk: `BL8409LO`, yang oleh model awal terbaca `BE8409LO` atau `BL8409LQ`.

### Model akhir

Pengaturan G2 dipakai sekali lagi untuk melatih satu model pada semua crop bersih (448 crop dari 64 plat) ditambah 1.879 crop Kaggle, lalu bobotnya dicampur 50% dengan model awal:

```powershell
uv run --frozen --extra cpu python training/ocr/crossval.py --final --out data/training/ocr/final-20261115 --epochs 12 --lr 0.0005 `
    --label-smoothing 0.01 --blend-share 0.5 --extra-csv data/dataset/ocr/extra/kaggle_recognition_clean.csv `
    --augmentation training/ocr/augmentation-cctv.yaml
Copy-Item data\training\ocr\final-20261115\model.onnx models\ocr\finetuned\timbangan-20261115.onnx
```

Training 15 menit di CPU laptop. Berkasnya 5,1 MB dengan masukan dan keluaran yang sama dengan model hub (`plate` 10×37, `region` 66), jadi `plate_config` tidak berubah.

Model ini sudah melihat semua plat berlabel, jadi tidak ada angka akurasi yang bisa dihitung untuknya dari data yang ada. Perkiraan akurasinya pada plat baru adalah baris "G2, bobot 50%" di tabel. Truk yang sama datang berulang ke timbangan, sehingga pada plat yang sudah pernah dilabeli hasilnya bisa lebih baik dari perkiraan itu; ini belum diukur.

### Pemeriksaan kedua: split tetap

Pengaturan yang sama dilatih sekali lagi hanya pada split `train` (`--final --final-split train`, 319 crop kita ditambah crop Kaggle), lalu membaca `val.csv` dan `test.csv` yang platnya tidak ikut dilatih. Crop-nya bagian dari 280 crop evaluasi silang, jadi ini bukan data baru, tetapi training-nya terpisah.

| Model | val (61 crop, 12 plat) | test (68 crop, 6 plat) | Gabungan: salah dengan confidence ≥ 0,9 |
|---|---|---|---|
| Model awal (hub) | 54 benar, karakter 96,1% | 61 benar, karakter 96,7% | 5 |
| G2 tanpa campuran bobot | 55 benar, karakter 97,8% | 62 benar, karakter 97,2% | 5 |
| G2, bobot 50% fine-tune | 54 benar, karakter 97,3% | 64 benar, karakter 97,8% | 2 |

Model dengan bobot campuran memperbaiki 3 crop (`BX8460XD` menjadi `BK8460XD`, `B9402PEU` menjadi `B9462PEU`, `BK6788VB` menjadi `BK8788VB`) dan tidak merusak satu pun. Arahnya sama dengan evaluasi silang; besarnya lebih kecil.

### Pemeriksaan ketiga: lewat kode service

Dua pemeriksaan di atas membaca crop yang sudah dipotong. Untuk memastikan model baru berjalan di jalur service yang utuh (detektor, tracking, pemilihan crop, OCR, voting, koreksi), rekaman 13 November (`data\samples\rec-20261113-b`, empat kamera, 30 menit) diputar ulang dua kali di GPU laptop, dengan konfigurasi yang hanya berbeda di `ocr.model_path`:

```powershell
uv run --frozen --extra gpu alpr-id benchmark data\samples\rec-20261113-b --save-crops --out data\benchmarks\replay-20261115\current
uv run --frozen --extra gpu alpr-id benchmark data\samples\rec-20261113-b --save-crops --out data\benchmarks\replay-20261115\finetuned --config <salinan app.yaml dengan model baru>
```

Kedua putaran menghasilkan jumlah yang sama: 4 event (2 di cam-2, 2 di cam-3) dari 10 panggilan OCR. Tiga event sama persis (`BK8460XD`, `BK8985HF`, `BK8354CP`, confidence di atas 0,99 pada kedua model). Satu berbeda: model awal menghasilkan `BK8701ZE` (confidence 0,94), model baru `BK8701LE` (0,98). Pada crop-nya tertulis BK 8701 LE, dan truk itu berlabel `BK8701LE` pada crop 14 November. Model akhir dilatih dengan crop truk itu dari 14 November, jadi ini contoh truk yang datang lagi, bukan plat baru.

Empat kendaraan tidak cukup untuk angka akurasi. Yang ditunjukkan: model baru bekerja di jalur service tanpa mengubah jumlah event atau jumlah bacaan. Perintah `alpr-id recognize` pada lima frame event juga memberi plat yang sama dari kedua model.

## Detektor

### Dataset

`training/detector/build_dataset.py` menggabungkan dua sumber menjadi `data/dataset/detector/`:

- Foto jalan dari Kaggle "Indonesian License Plate Dataset" (kotak plat digambar orang), diperkecil ke 1280 piksel.
- Frame kamera timbangan: frame yang disimpan service bersama tiap event dan key frame dari rekaman 13 November. Frame ini tidak punya kotak buatan orang. Labelnya dari detektor dan OCR yang dipakai service: sebuah kotak dihitung plat hanya bila OCR membaca plat berpola sah dengan kode wilayah yang dikenal dan confidence minimal 0,8. Frame yang memuat kotak yang tidak bisa dipastikan dibuang (51 frame), supaya lampu belakang dan tulisan rompi tidak menjadi label. Frame tanpa kotak sama sekali menjadi contoh latar.

| Split | Gambar | Kotak plat | Isi |
|---|---|---|---|
| train | 1.191 | 1.681 | 800 foto Kaggle, 151 frame kamera berplat, 240 frame kamera kosong |
| val | 152 | 204 | 100 foto Kaggle, 25 frame kamera berplat, 27 frame kosong |
| test_cctv | 81 | 81 | frame event kamera sejak 14 November pukul 12.00 |
| test_public | 100 | 197 | split test Kaggle |

Batas cara pelabelan ini: plat yang terlewat detektor sekarang tidak ada di label mana pun. Pada `test_cctv` sebuah model hanya bisa menunjukkan bahwa ia menemukan yang ditemukan detektor sekarang, bukan bahwa ia menemukan lebih banyak.

### Training

```powershell
uv run --project training python training/detector/train.py train --model models/detector/src/yolo26n.pt `
    --data data/dataset/detector/data.yaml --name yolo26n-20261115
```

120 epoch, gambar 640, batch 16, tanpa pencerminan kiri-kanan, 97 menit di RTX 5050 laptop. Bobot terbaik menurut validasi ada di epoch 93. Dua percobaan pertama gagal karena memori: laptop 16 GB kehabisan ruang commit saat proses pemuat data ikut memuat pustaka CUDA. Karena itu `train.py` memakai `workers=0` dan menyimpan gambar di RAM.

### Hasil

Validasi Ultralytics pada berkas `.pt`:

| Model | test_cctv: presisi / recall / mAP50 / mAP50-95 | test_public: presisi / recall / mAP50 / mAP50-95 |
|---|---|---|
| YOLOv8n yang dipakai service | 0,999 / 0,975 / 0,994 / 0,860 | 0,971 / 0,822 / 0,895 / 0,557 |
| YOLO26n hasil training | 0,987 / 0,962 / 0,984 / 0,778 | 0,951 / 0,981 / 0,989 / 0,730 |

Yang menentukan adalah berkas ONNX lewat kode service (`training/detector/compare_runtime.py`, memakai `PlateDetector` dan ambang confidence service):

| Model | test_cctv (81 plat): ditemukan / terlewat / kotak lebih | val (204 plat): ditemukan / terlewat / kotak lebih |
|---|---|---|
| YOLOv8n yang dipakai service | 80 / 1 / 0 | 174 / 30 / 4 |
| YOLO26n hasil training | 76 / 5 / 2 | 200 / 4 / 17 |

Lima plat yang terlewat YOLO26n di frame kamera dilihat satu per satu: empat di antaranya plat putih `BL8409LO` yang jelas terbaca, satu plat belakang truk yang gelap. Dua kotak lebihnya lampu kuning di bumper (confidence 0,81) dan sebidang bumper tanpa tulisan (0,45). Di foto Kaggle YOLO26n memang jauh lebih baik, tetapi itu foto mobil pribadi dari dekat, bukan pemandangan kamera timbangan.

### Keputusan

Detektor tidak diganti. Di frame kamera YOLO26n lebih buruk, padahal labelnya pun condong ke detektor sekarang, dan ia 2 ms lebih lambat per frame di GTX 1060. Berkasnya disimpan di `models/detector/candidates/yolo26n-20261115.onnx` sebagai bahan pembanding.

YOLO26s tidak dilatih sampai selesai. Model bawaannya diekspor dan diukur dulu di PC ROG: 35,8 ms per frame. Empat kamera pada 8 fps berarti 32 frame per detik, atau 1,15 detik kerja detektor untuk tiap detik, jadi service harus melewati frame.

Untuk menilai detektor dengan adil dibutuhkan frame kamera yang kotaknya digambar orang, termasuk plat yang terlewat detektor sekarang. Contoh yang paling berguna untuk training berikutnya sudah terlihat: plat putih dan lampu kuning.

## Kecepatan di PC ROG

Diukur dengan `scripts/bench_onnx.py` di GTX 1060 (`CUDAExecutionProvider`), 300 kali jalan per model, selagi service tetap berjalan:

| Model | Median | p95 |
|---|---|---|
| Detektor `plat.onnx` (YOLOv8n) | 15,5 ms | 22,2 ms |
| Detektor YOLO26n hasil training | 17,7 ms | 25,2 ms |
| Detektor YOLO26s (bobot bawaan) | 35,8 ms | 48,7 ms |
| OCR model awal `cct_s_v2_global.onnx` | 7,1 ms | 14,0 ms |
| OCR `timbangan-20261115.onnx` | 7,4 ms | 13,3 ms |

Service di sana (versi 0.4.2, empat kamera) memproses 7,5 sampai 8,0 fps per kamera dari target 8, dengan total per frame median 25 sampai 28 ms dan p95 sekitar 55 ms. VRAM terpakai 1.290 dari 3.072 MiB. Target kecepatan sudah tercapai dengan model yang ada, dan model OCR baru tidak mengubahnya, jadi tidak ada optimasi yang dikerjakan. Detektor memakai kira-kira setengah waktu GPU (15,5 ms × 32 frame per detik); itu ruang yang tersisa bila kamera bertambah.

## Pemasangan di PC ROG

Model baru dipasang 15 November 2026 pukul 09.47 atas permintaan pengguna, saat tidak ada plat di depan kamera mana pun. Sesudah restart: empat kamera tersambung, kedua model memakai `CUDAExecutionProvider`, 7,5 sampai 8 fps per kamera, dan sebelas menit kemudian total per frame median 25 sampai 27 ms dengan p95 55 sampai 57 ms, sama dengan sebelum diganti. `config\app.yaml` di laptop diubah sama, supaya deploy berikutnya tidak mengembalikan model awal.

Dua event pertama dari model baru: `BK8606GX` (cam-4, 09.53) dan `BL8409LO` (cam-1, 09.58). Truk kedua itu pagi yang sama dua kali terbaca `BE8409LO` oleh model awal (cam-1 pukul 09.33, cam-4 pukul 09.41).

Yang ada di PC ROG:

- `C:\alpr-id\models\ocr\finetuned\timbangan-20261115.onnx`, sudah diukur kecepatannya di GTX 1060 (tabel di atas);
- `C:\alpr-id\deploy\rog\set-ocr-model.ps1`, yang mengganti baris `model_path` di bagian `ocr` pada `config\app.yaml` lalu me-restart task `ALPR-ID`. Berkas konfigurasi sebelumnya disimpan sebagai `app.yaml.before-ocr-switch`.

Perintah yang dipakai, dari laptop lewat Tailscale:

```powershell
ssh -i $HOME\.ssh\alpr_rog_ed25519 TOMIYASU@100.119.227.41 "powershell -ExecutionPolicy Bypass -File C:\alpr-id\deploy\rog\set-ocr-model.ps1 -Model models/ocr/finetuned/timbangan-20261115.onnx"
```

Kembali ke model awal: perintah yang sama dengan `-Model models/ocr/cct_s_v2_global.onnx`.

Tiga hal yang perlu diperhatikan saat mengganti:

1. Restart menghapus ingatan service tentang kendaraan yang sedang terlihat. Truk yang masih berdiri di timbangan akan terbaca lagi sebagai event baru, jadi ganti saat `plates_in_view` di `GET /cameras` bernilai 0 untuk semua kamera.
2. `deploy.ps1` mengirim `config\app.yaml` dari laptop. Ubah `ocr.model_path` di laptop juga, atau deploy berikutnya mengembalikan model awal.
3. Sesudahnya periksa `GET /health` (empat kamera tersambung, provider `CUDAExecutionProvider`) dan beberapa event pertama di `/ui`.

Karena `config\app.yaml` sekarang menunjuk model baru, `evaluate.py` tanpa `--model` mengukur model baru itu. Untuk membandingkan dengan model awal, sebut berkasnya: `--model models/ocr/cct_s_v2_global.onnx`.

## Batasan dan langkah berikutnya

- Data ujinya sempit: 54 plat dari dua hari, hampir semuanya berkode BK, dan dua truk menyumbang hampir seperlima crop. Angka di dokumen ini menunjukkan arah, belum besarnya keuntungan di lapangan.
- Model akhir belum diuji pada crop yang dikumpulkan setelah ia dilatih. Service menyimpan semua crop yang dibaca di `reads/`, jadi beberapa hari crop baru yang diperiksa lewat halaman periksa label (`training/dataset/make_review_page.py`) akan memberi angka akurasi pertama yang benar-benar bebas, sekaligus menambah ragam plat untuk training berikutnya.
- Kesalahan yang tersisa kebanyakan di angka (`9` terbaca `0`, `8` terbaca `9` atau `3`) dan `C` terbaca `G` pada crop yang kecil dan buram. Model tidak bisa membaca yang tidak tertangkap kamera; memperbesar plat di gambar (zoom atau posisi kamera) akan lebih terasa daripada training tambahan.
- Detektor belum punya data uji yang adil. Yang dibutuhkan: frame kamera dengan kotak buatan orang, terutama plat putih, plat yang terlewat, dan lampu kuning yang keliru ditandai.
- Dua penjaga untuk plat tertutup dari Fase 3 (lewati crop dengan lebar:tinggi di bawah 2,3; tahan plat tanpa huruf seri atau lebih pendek dari 6 karakter) tidak dipasang. Keputusan pengguna 15 November 2026: yang lebih diperlukan adalah hasil bacaan langsung dari stream, bukan penyaringan event.
