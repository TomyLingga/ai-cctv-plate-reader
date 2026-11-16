# Fase 3: dataset dan pipeline training

Dimulai 15 November 2026 atas permintaan pengguna. Dokumen ini mencatat keadaan sebenarnya; angka di sini berasal dari perintah yang dijalankan, dan diberi keterangan bila bukan angka akurasi.

## Sumber data

Service di PC ROG menyimpan setiap crop yang masuk OCR di `data\media\<tanggal>\<kamera>\reads\`, dengan bacaan modelnya di nama berkas. Sampai 15 November 2026 pukul 00.50:

| Tanggal | Crop bacaan |
|---|---|
| 13 November | 68 |
| 14 November | 562 |

Salinannya di laptop ada di `data\dataset\raw\rog-reads\` (crop bacaan dan `events.json`). Folder `data\` tidak masuk git.

## Alur

```powershell
# 1. Bangun dataset dari crop dan daftar event
uv run --frozen --extra cpu python training/dataset/build_ocr_dataset.py --raw data/dataset/raw/rog-reads --out data/dataset/ocr

# 2. Buat halaman periksa label (satu berkas HTML, tanpa server)
uv run --frozen --extra cpu python training/dataset/make_review_page.py --dataset data/dataset/ocr --raw data/dataset/raw/rog-reads --splits test,val,none

# 3. Setelah diperiksa: simpan labels.csv ke data\dataset\ocr\review\labels.csv, lalu ulangi langkah 1

# 4. Ukur sebuah model pada sebuah CSV
uv run --frozen --extra cpu python training/ocr/evaluate.py --csv data/dataset/ocr/test.csv
```

### Dari mana label berasal

- **Label manusia** (`labels.csv` dari halaman periksa) selalu menang. Halaman itu juga menandai plat yang tertutup atau terpotong, yang tidak terbaca, dan gambar yang bukan plat. Crop yang dinyatakan "terlihat utuh" oleh pemeriksa tidak lagi disaring dengan tebakan bentuk dan panjang bacaan di bawah. Tulisan `NOTPLATE` di kotak plat diperlakukan sama dengan pilihan "Bukan plat". Satu pemeriksaan tetap berlaku: bila label sebuah crop berbeda dari label crop lain kendaraan yang sama, crop itu disisihkan sebagai kemungkinan salah ketik.
- **Label otomatis** dipakai bila event kendaraan itu layak dipercaya: berpola plat umum, kode wilayahnya ada di daftar dan tidak hasil koreksi, confidence minimal 0,9, dan sedikitnya dua crop kendaraan itu terbaca persis sama dengan plat event. Crop lain dari kendaraan yang sama ikut mendapat label itu walau bacaannya berbeda; itulah contoh sulit yang berguna untuk training.

Label otomatis adalah model yang sepakat dengan dirinya sendiri. Ia boleh dipakai untuk training, tetapi bukan untuk mengukur akurasi: `test.csv` hanya berisi crop berlabel manusia.

### Pembagian

Pembagian dilakukan per plat, bukan per crop, dengan hash teks plat: 70% train, 15% val, 15% test. Truk yang sama tidak pernah berada di train dan test sekaligus, dan pembagiannya tidak berubah saat data bertambah. Semua data berasal dari CCTV timbangan.

### Crop yang tidak ikut training

Pengguna meminta plat yang tertutup tidak dipakai, kecuali terbukti tidak merusak model. Crop berikut diberi keterangan di kolom `suspect` dan tidak masuk `train.csv`, `val.csv`, maupun `test.csv`:

- ditandai tertutup, tidak terbaca, atau bukan plat oleh pemeriksa;
- berlabel manusia tetapi berbeda dari label crop lain kendaraan yang sama;
- berlabel otomatis dan bentuknya di luar 2,2 sampai 5,5 (lebar dibanding tinggi). Pada 241 crop plat utuh rasionya 2,72 sampai 3,59 (persentil 5 sampai 95), dan hanya 2 yang di bawah 2,3;
- berlabel otomatis dan bacaannya dua karakter atau lebih lebih pendek dari labelnya;
- berlabel otomatis tetapi bacaannya berbeda lebih dari tiga karakter dari label.

Untuk menjawab apakah crop tertutup merusak model, skrip juga menulis `train_with_covered.csv` (crop train ditambah crop tertutup dari plat yang sama-sama masuk train) dan `test_covered.csv`.

## Keadaan dataset setelah label manusia masuk (15 November 2026, 02.20)

Pengguna memeriksa 456 crop di halaman periksa: 281 terlihat utuh, 31 tertutup atau terpotong, 144 tidak terbaca (23 di antaranya ditulis `NOTPLATE`). Dari 281 yang utuh, 52 labelnya dibetulkan dari isian awal.

| | Jumlah |
|---|---|
| Crop terbaca dari nama berkas | 625 dari 630 |
| Berlabel manusia | 312 (termasuk 31 yang tertutup) |
| Berlabel otomatis (belum diperiksa) | 169 |
| Tanpa label: tidak terbaca 121, bukan plat 23 | 144 |
| Berlabel dan bersih | 448, dari 64 plat berbeda |
| Bersih tetapi bacaan model berbeda dari label (contoh sulit) | 64 |
| train / val / test (bersih) | 319 / 61 / 68 crop, dari 46 / 12 / 6 plat |
| Berlabel manusia di train / val / test | 151 / 61 / 68 |

Satu label disisihkan: crop `093056_t00027_C18880G_046_900066.jpg` diberi `BK8550GI`, sedangkan crop lain dari kendaraan itu `BK8510GI`.

Test set tetap hanya memuat 6 plat, dan 33 dari 68 crop-nya milik satu truk (`BK8486WR`). Itu terlalu sempit untuk memutuskan model mana yang lebih baik, jadi keputusan memakai evaluasi silang per plat (`training/ocr/crossval.py`, lihat di bawah).

## Keadaan dataset sebelum label manusia (15 November 2026, 01.00)

| | Jumlah |
|---|---|
| Crop terbaca dari nama berkas | 625 dari 630 |
| Berlabel otomatis | 257 |
| Berlabel manusia | 0 |
| Belum berlabel | 368 |
| Berlabel dan bersih | 255, dari 57 plat berbeda |
| Bersih tetapi bacaannya berbeda dari label (contoh sulit) | 15 |
| train / val / test (bersih) | 168 / 46 / 41 crop, dari 39 / 12 / 6 plat |

Model yang sekarang dipakai, diukur pada 168 crop train berlabel otomatis: 156 crop (92,9%) sama dengan labelnya, karakter 98,8%. Ini kesepakatan model dengan hasil voting-nya sendiri, bukan akurasi.

Dua hal membatasi dataset ini:

1. Dari 255 crop bersih, model sudah membaca 240 dengan benar. Melatih ulang pada data yang sebagian besar sudah benar memberi sedikit pelajaran baru. Yang berharga adalah 368 crop yang belum berlabel, yaitu crop yang membuat model ragu atau salah, dan itu memerlukan label manusia.
2. Test set berlabel manusia belum ada, jadi akurasi belum bisa dihitung.

Halaman periksa label ada di `data\dataset\ocr\review\review.html`: 456 crop (test, val, dan yang belum berlabel).

## Lingkungan dan perintah training OCR

Training memakai lingkungan sendiri di `training\ocr\` (TensorFlow hanya dibutuhkan di sini). Versinya mengikuti lockfile fast-plate-ocr 1.1.0: keras 3.12.1, tensorflow 2.19.1, tf2onnx 1.16.1. `coremltools` dan `ai-edge-litert` tidak dipasang; keduanya hanya untuk export CoreML dan TFLite, dan `coremltools` tidak punya wheel Windows.

Training berjalan di CPU. TensorFlow tidak mendukung GPU di Windows sejak versi 2.11, dan modelnya kecil (1,26 juta parameter), jadi beberapa ratus sampai beberapa ribu crop masih selesai dalam hitungan menit. Jalur GPU lewat backend PyTorch belum dicoba.

```powershell
$env:UV_CACHE_DIR = "D:\uv-cache"; $env:UV_LINK_MODE = "copy"
uv sync --project training/ocr

# Bobot awal dan konfigurasi model (sekali saja), dari rilis "arg-plates" fast-plate-ocr:
#   models\ocr\pretrained\cct_s_v2_global.keras
#   models\ocr\pretrained\cct_s_v2_global_model_config.yaml
#   models\ocr\pretrained\cct_s_v2_global_plate_config.yaml

# Fine-tune
$env:KERAS_BACKEND = "tensorflow"
uv run --project training/ocr fast-plate-ocr train `
  --model-config-file models/ocr/pretrained/cct_s_v2_global_model_config.yaml `
  --plate-config-file models/ocr/pretrained/cct_s_v2_global_plate_config.yaml `
  --annotations data/dataset/ocr/train.csv --val-annotations data/dataset/ocr/val.csv `
  --weights-path models/ocr/pretrained/cct_s_v2_global.keras `
  --epochs 15 --batch-size 32 --lr 0.0001 --weight-decay 0.0005 --label-smoothing 0.0 `
  --output-dir data/training/ocr

# Export ke ONNX, lalu ukur pada CSV yang sama dengan model lama
uv run --project training/ocr python training/ocr/export_onnx.py --model data/training/ocr/<run>/best.keras `
  --plate-config data/training/ocr/<run>/plate_config.yaml --out models/ocr/finetuned/<nama>.onnx
uv run --frozen --extra cpu python training/ocr/evaluate.py --csv data/dataset/ocr/val.csv `
  --model models/ocr/finetuned/<nama>.onnx
```

Catatan yang menentukan bentuk skripnya:

- `fast-plate-ocr export --format onnx` di versi 1.1.0 gagal di Windows (issue #103: berkas sementara tidak bisa dibuka ulang). `training/ocr/export_onnx.py` mengulang langkah yang sama dengan berkas sementara biasa, lalu memeriksa bahwa keluaran ONNX sama dengan Keras.
- Pelatih menggabungkan folder CSV dan `image_path` sebagai teks, jadi path di `train.csv` dan `val.csv` ditulis relatif terhadap folder CSV.
- CSV membawa kolom `plate_region` berisi `Indonesia`. Tanpa kolom itu pelatih membangun model tanpa keluaran `region`, berbeda dari bobot awalnya.
- Model yang dipakai service baru diganti setelah model baru menang pada test set berlabel manusia. Sampai itu ada, hasil training hanya disimpan di `models\ocr\finetuned\`.

## Uji alur, 15 November 2026

Tujuannya memastikan train, export, dan evaluasi berjalan di laptop ini sebelum label manusia ada. Datanya 168 crop train dan 46 crop val, semuanya berlabel otomatis. Perintahnya seperti di atas: 15 epoch, batch 32, lr 0,0001.

- Training selesai dalam 134 detik di CPU.
- Export lewat `export_onnx.py` berhasil; masukan dan keluaran ONNX sama bentuknya dengan model hub (`input` uint8 N×64×128×3, keluaran `plate` N×10×37 dan `region` N×66).

| Model | val (46 crop) | train (168 crop) |
|---|---|---|
| Model yang dipakai service | 45 sama dengan label, karakter 99,5% | 156, karakter 98,8% |
| `best.keras` hasil uji | 45, karakter 99,5% | 156, karakter 98,8% |
| `last.keras` hasil uji | 45, karakter 99,7% | 157, karakter 99,0% |

Angka ini kesepakatan dengan label otomatis, bukan akurasi, dan tidak menunjukkan perbaikan yang berarti. Itu sesuai dugaan: datanya sedikit dan sebagian besar sudah dibaca benar. Model hasil uji tidak dipasang di PC ROG.

Dua hal yang dipelajari untuk training sungguhan:

1. `best.keras` hampir sama dengan bobot awal (selisih probabilitas terbesar 0,013; tidak ada crop yang bacaannya berubah), sedangkan `last.keras` berubah (5 dari 168 crop berbeda bacaan). Sebabnya dua: validasi memakai bobot rata-rata bergerak (EMA) yang nyaris belum bergeser dalam 90 langkah, dan `val_plate_acc` tidak berubah sepanjang 15 epoch sehingga yang tersimpan sebagai terbaik adalah epoch pertama. Dengan data sedikit, pakai `--no-use-ema` atau epoch lebih banyak, dan pilih checkpoint dengan `--early-stopping-metric val_plate_char_acc` atau `val_loss`.
2. `cct_s_v2_global.keras` tidak bisa dimuat utuh oleh keras 3.12.1 (`quantization_config` tidak dikenal); berkas itu disimpan dengan Keras yang lebih baru. Untuk fine-tune itu tidak mengganggu, karena pelatih hanya memuat bobotnya (`--weights-path`).

## Fine-tune dengan label manusia, 15 November 2026

Semua angka di bagian ini dihitung pada crop berlabel manusia, satu crop sekali baca, tanpa voting dan tanpa koreksi. Jadi ini akurasi model sendirian, bukan akurasi event yang keluar dari service.

### Split tetap

`train.csv` 319 crop, checkpoint dipilih dengan `val.csv`. Perintahnya seperti di atas dengan `--epochs 50 --lr 0.0005 --no-use-ema --early-stopping-metric val_plate_char_acc`. Run dengan lr 0,0001 kalah jelas di `val_loss` (0,17 dibanding 0,05) dan dihentikan. Run lr 0,0005 terputus di epoch 41 karena sesi kerja tertutup; `best.keras` (epoch 19) dan `last.keras` (epoch 41) utuh dan diekspor ke `models\ocr\finetuned\label-20261115-best.onnx` dan `-last.onnx`.

| Model | val (61 crop, 12 plat) | test (68 crop, 6 plat) |
|---|---|---|
| Model yang dipakai service | 54 benar (88,5%), karakter 96,1% | 61 benar (89,7%), karakter 96,7% |
| `label-20261115-best` | 56 benar (91,8%), karakter 98,8% | 62 benar (91,2%), karakter 97,8% |
| `label-20261115-last` | 55 benar (90,2%), karakter 98,2% | 62 benar (91,2%), karakter 97,4% |

Di test, fine-tune memperbaiki 3 crop dan merusak 2 (`BK8486WR` terbaca `BK8286WR` dan `BK8486MR`). Selisih satu crop pada 6 plat tidak cukup untuk memutuskan.

### Evaluasi silang per plat

`training/ocr/crossval.py` membagi plat ke 5 kelompok dengan hash teksnya. Untuk tiap kelompok, model dilatih pada crop semua plat lain (label manusia dan otomatis), lalu membaca crop berlabel manusia dari kelompok itu. Hasilnya 280 crop dari 54 plat, masing-masing dibaca model yang belum pernah melihat platnya. Tidak ada yang dipilih dengan data uji: pengaturannya sama di semua kelompok (25 epoch, lr 0,0005, batch 32, tanpa EMA) dan yang dipakai bobot epoch terakhir.

```powershell
uv run --frozen --extra cpu python training/ocr/crossval.py --out data/training/ocr/cv-20261115 --epochs 25 --lr 0.0005
uv run --frozen --extra cpu python training/ocr/crossval.py --out data/training/ocr/cv-20261115-covered --epochs 25 --lr 0.0005 --with-covered
```

Satu kelompok memakan 5 sampai 7 menit di CPU laptop. Progress per epoch ditulis ke `<out>\progress.log`; kelompok yang sudah selesai tidak dilatih ulang bila skrip dijalankan lagi.

| | Model service | Fine-tune tanpa plat tertutup |
|---|---|---|
| Crop terbaca benar (dari 280) | 228 (81,4%) | 241 (86,1%) |
| Akurasi karakter | 95,7% | 97,2% |
| Crop yang diperbaiki / dirusak fine-tune | | 21 / 8 |
| Per kelompok (crop benar) | 77, 44, 29, 56, 22 | 89, 46, 30, 54, 22 |
| Salah baca dengan confidence 0,9 ke atas | 25 dari 52 | 27 dari 39 |
| Rata-rata confidence salah baca | 0,834 | 0,910 |
| Plat berkode 1 huruf (38 crop, 5 plat) | 32 | 32 |
| Plat berkode 2 huruf (242 crop, 49 plat) | 196 | 209 |

Dihitung per kendaraan (kamera dan track yang sama, bacaan terbanyak berbobot confidence, sebagai pendekatan kasar voting di service):

| | Model service | Fine-tune |
|---|---|---|
| Semua kendaraan (154) | 135 benar | 138 benar |
| Kendaraan dengan 2 crop atau lebih (85) | 82 benar | 82 benar |

Yang terbaca dari angka itu:

- Kenaikan per crop nyata tetapi terkumpul di satu truk. Kelompok pertama naik 12 crop (77 ke 89), sebagian besar `BL8409LO` yang oleh model lama terbaca `BE8409LO` atau `BL8409LQ`; empat kelompok lain bersama-sama hanya naik 1 crop.
- Per kendaraan hampir tidak ada bedanya. Voting sudah menutup sebagian besar salah baca satu crop, dan untuk kendaraan dengan dua crop atau lebih hasilnya sama (82 dari 85).
- Fine-tune membawa kesalahan baru. `M` terbaca `H` lima kali (`BK8021MN` menjadi `BK8021HN`), `W` terbaca `H` sekali, dan `B9098FEW` mendapat huruf `I` tambahan setelah `B` di tiga crop. M, W, dan H memang kelompok huruf yang rawan di plat Indonesia (keputusan arsitektur 9).
- Model hasil fine-tune lebih yakin saat salah: rata-rata confidence salah baca naik dari 0,834 ke 0,910. Ambang voting di service (0,85) dan ambang bacaan saat lewat (0,8) disetel untuk model lama, jadi dengan model baru lebih banyak salah baca yang lolos. Training ini memakai `--label-smoothing 0.0`; itu perlu dinaikkan pada training berikutnya.
- Datanya sempit: 64 plat, dan dua truk (`BL8409LO`, `BK8486WR`) menyumbang hampir seperlima crop bersih.

Kesimpulan sementara: model hasil fine-tune ini belum dipasang di PC ROG. Keuntungannya per kendaraan terlalu kecil untuk menutup risiko kesalahan M/H dan confidence yang kurang bisa dipercaya. Langkah yang lebih menjanjikan adalah menambah ragam plat (crop beberapa hari lagi, diperiksa lewat halaman yang sama) lalu melatih ulang dengan label smoothing dan batas jumlah crop per plat.

Lanjutannya ada di [dokumen Fase 4](fase-4-finetune-dan-perbandingan.md): dengan crop Kaggle, augmentasi buram, dan bobot yang dicampur dengan model awal, kesalahan baru itu hampir hilang dan salah baca ber-confidence tinggi turun di bawah model awal.

### Dengan dan tanpa plat tertutup

Evaluasi silang yang sama, dengan 31 crop bertanda tertutup ikut dilatih (`--with-covered`):

| | Tanpa plat tertutup | Dengan plat tertutup |
|---|---|---|
| Crop benar (dari 280) | 241 | 241 |
| Diperbaiki / dirusak | 21 / 8 | 26 / 13 |
| Kendaraan benar (dari 154) | 138 | 140 |
| Kendaraan dengan 2 crop atau lebih (dari 85) | 82 | 81 |

Jumlah crop benar sama, tetapi crop yang berubah bacaannya lebih banyak ke dua arah. Plat tertutup tidak menambah apa-apa pada ukuran ini, jadi tetap tidak dipakai. Dua run yang hanya berbeda 31 crop ini juga menunjukkan besarnya derau: daftar crop yang rusak berbeda cukup jauh di antara keduanya, sehingga selisih di bawah kira-kira 5 crop tidak bisa dianggap perbedaan nyata.

## Dataset luar yang diperiksa, 15 November 2026

Pengguna mengunduh empat dataset dari Kaggle (`..\dataset`, 10.063 berkas, 3,3 GB) dan satu dari Roboflow (`..\from roboflow`, 4.512 gambar). Semuanya hanya dibaca di tempatnya; tidak ada yang disalin ke dataset OCR proyek.

| Dataset | Isi | Penilaian |
|---|---|---|
| Kaggle "Indonesian License Plate Dataset" | 1.000 foto jalan 3024×3024 dengan 1.906 kotak plat (seluruh plat termasuk baris tanggal) dan teksnya | Kotaknya konsisten dan plat berukuran relatif mirip dengan kamera kita. Dipakai untuk melatih detektor. |
| Kaggle "Indonesian License Plate Recognition Dataset" | 1.906 crop plat 602×189 dengan kotak per huruf | Crop dari dataset pertama; teks dari kotak huruf cocok dengan teks dataset pertama pada 1.902 crop. 27 teks di luar pola plat. Calon data tambahan OCR. |
| Kaggle `plate_detection_dataset` | 1.383 gambar 640 px, anotasi COCO, lisensi Apache 2.0 | Tidak dipakai. Isinya campuran: foto dataset pertama yang diperkecil, plat motor jarak dekat, dan CCTV jalan raya. Aturan kotaknya tidak konsisten: pada foto motor kotak hanya mencakup baris nomor, tanpa baris tanggal. |
| Kaggle `plate_text_dataset` | 1.863 gambar dan teks plat | Hanya 656 yang berbentuk crop plat; 1.207 sisanya foto utuh atau plat motor dua baris. Banyak crop yang pecah-pecah. |
| Roboflow "Indonesia Number Plate" v4 | 4.512 gambar 640×640 (tiga salinan augmentasi per gambar asal di train), satu kelas, Public Domain | Tidak dipakai. Sebagian besar plat memenuhi seluruh gambar (lebar kotak median 626 dari 640 piksel) dan gambar ditarik menjadi persegi, sedangkan di kamera kita plat hanya beberapa persen lebar gambar. 15% labelnya poligon, bukan kotak. Tanpa teks plat. |

Bedanya dengan crop kamera timbangan: data luar itu plat hitam mobil dan motor pribadi, difoto dekat dan tajam, dengan kode wilayah B, L, N, H, E. Crop kita plat kuning dan putih truk berkode BK, lebar median 118 piksel, dan buram. Model yang dipakai service membaca 1.685 dari 1.906 crop dataset kedua dengan benar (88,4%, karakter 97,3%) dan 437 dari 656 crop berbentuk plat di `plate_text_dataset` (66,6%); label keduanya tidak diperiksa ulang, jadi itu angka kecocokan.

Apakah crop luar membantu atau merusak diukur dengan evaluasi silang yang sama: 400 crop dataset kedua (pilihan tetap dari 1.879 yang berpola plat) ditambahkan ke data latih setiap kelompok (`--extra-csv`, `--extra-limit 400`), lalu model membaca 280 crop kita.

| | Tanpa data luar | Ditambah 400 crop Kaggle |
|---|---|---|
| Crop benar (dari 280) | 241 | 242 |
| Diperbaiki / dirusak | 21 / 8 | 22 / 8 |
| Kendaraan benar (dari 154) | 138 | 138 |
| Kendaraan dengan 2 crop atau lebih (dari 85) | 82 | 82 |
| Salah baca dengan confidence 0,9 ke atas | 27 dari 39 | 30 dari 38 |

Pada takaran itu crop Kaggle tidak membantu dan tidak merusak: selisihnya di dalam derau. Kesalahan `M` menjadi `H` lebih jarang (tidak lagi masuk dua belas kekeliruan terbanyak, dari lima kali) dan `B9098FEW` tidak lagi mendapat huruf tambahan; jumlahnya terlalu sedikit untuk disebut pasti.

INDO-ALPR (Mendeley Data, DOI 10.17632/j3256db3fz.1, CC BY 4.0) juga dilihat. Nama berkasnya hanya memuat angka dan seri (`1067 TJR_5.jpg`), tanpa kode wilayah, jadi tidak bisa langsung dipakai sebagai label plat utuh.

## Yang juga terlihat dari crop

Detektor menandai benda yang bukan plat: deret lampu belakang, bagian bak, dan tulisan "SATPAM" di rompi petugas. Di halaman periksa itu ditandai "Bukan plat", dan nanti menjadi contoh negatif untuk training detektor.
