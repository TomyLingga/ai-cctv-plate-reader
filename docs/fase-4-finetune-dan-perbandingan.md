# Fase 4: fine-tune, perbandingan dengan baseline, dan kecepatan

Dikerjakan 15 November 2026 di laptop pengembangan (training) dan PC ROG (pengukuran kecepatan). Dataset dan cara membuatnya ada di [dokumen Fase 3](fase-3-dataset-dan-training.md); dokumen ini memuat apa yang dilatih, hasil ukurnya, dan keputusan yang diambil.

## Ringkasan

| Bagian | Keputusan | Dasar |
|---|---|---|
| OCR | Model baru `models/ocr/finetuned/timbangan-20261115.onnx` dipasang di PC ROG pada 15 November 2026 pukul 09.47. Model awal tetap ada di sana untuk kembali. | Pada 280 crop berlabel manusia yang platnya belum pernah dilihat model: 244 terbaca benar, model awal 228. Salah baca dengan confidence 0,9 ke atas turun dari 25 menjadi 14. Per kendaraan 139 benar dari 154 (model awal 134), tidak ada kendaraan yang menjadi salah. |
| Detektor | Tetap `models/detector/plat.onnx` (YOLOv8n) | YOLO26n hasil training menemukan 76 dari 81 plat pada frame kamera, detektor sekarang 80. YOLO26s terlalu lambat untuk GTX 1060. |
| Kecepatan | Tidak ada yang diubah | Empat kamera sudah diproses 7,5 sampai 8 fps dari target 8. Model OCR baru sama cepatnya dengan yang lama. |
| Detektor, 18 November 2026 | `models/detector/candidates/yolov8n-20261118b.onnx` (YOLOv8n yang sama, dilatih lanjut) dipasang di PC ROG pada 18 November 2026 pukul 15.59. `plat.onnx` tetap ada di sana untuk kembali. | Pada 443 plat di frame event 18 November: 437 berskor 0,5 ke atas, detektor sekarang 317. Tidak ada plat yang ditemukan detektor sekarang yang hilang. |

Semua angka akurasi di dokumen ini dihitung pada crop berlabel manusia dari kamera timbangan. Jumlah platnya masih sedikit (54 plat, 154 kendaraan), jadi selisih di bawah kira-kira 5 crop tidak dianggap perbedaan. Bagian detektor tidak memuat angka akurasi: kotak platnya diberi detektor dan OCR, bukan orang, jadi di sana yang dihitung hanya berapa plat berlabel yang ditemukan.

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

## Detektor, 18 November 2026: plat malam dan plat yang berdiri jauh

Dikerjakan atas permintaan pengguna ("coba latih plat malam, tapi di laptop ku saja, nanti hasil modelnya di deploy ke RoG"). Latar belakangnya ada di [dokumen Fase 2 bagian 3d](fase-2-baseline-dan-benchmark.md): plat truk yang berdiri di timbangan hanya selebar kira-kira 100 piksel, dan setelah ROI diperkecil ke masukan detektor tinggal 38 piksel. Area fokus menolong, tetapi plat malam di balik bull bar (`BK 8391 LJ`) tetap di bawah skor 0,5 di semua pandangan.

Yang dilatih kali ini YOLOv8n yang sama dengan yang dipakai service (`models/detector/src/best.pt`), bukan arsitektur baru. Kecepatannya di GTX 1060 karena itu tidak berubah.

### Cara memberi label

Frame kamera tetap tidak punya kotak buatan orang. `training/detector/build_dataset.py` memberi label seperti ini:

- Detektor service melihat tiap frame dari semua pandangan: ROI yang diperkecil, area fokus, dan tiap petak 640 piksel dari ROI pada resolusi asli. Plat yang hilang di ROI yang diperkecil ditemukan di petak.
- Detektor kedua, YOLOv9-t dari open-image-models (`models/detector/yolo-v9-t-640-license-plates-end2end.onnx`), melihat area fokus dan petak yang sama. Bila keduanya menemukan plat yang sama, kotak detektor service yang dipakai. Detektor kedua hanya menambah plat yang tidak terlihat detektor service.
- Sebuah kotak dihitung plat bila OCR membaca plat berpola sah, berkode wilayah yang dikenal, minimal 6 karakter, dengan confidence minimal 0,8; atau bila service melaporkan plat yang memenuhi syarat yang sama di tempat itu pada frame itu.
- Kotak yang tidak bisa dipastikan ditutup warna abu-abu di gambar latih: bukan plat, bukan juga latar.
- Tiap frame menjadi dua gambar, yaitu dua pandangan yang diterima detektor service: ROI dan area fokus.

Data uji `test_cctv` adalah semua frame event 18 November 2026 (00.00 sampai 12.07). Frame 14 sampai 17 November dan rekaman 13 November masuk train dan val. Truk yang sama datang tiap hari, jadi truk di data uji kebanyakan bukan truk baru.

| Split | Gambar | Kotak plat | Isi |
|---|---|---|---|
| train | 2.283 | 3.016 | 800 foto Kaggle, 1.368 gambar dari frame event, 115 dari rekaman |
| val | 321 | 400 | 100 foto Kaggle, 199 dari frame event, 22 dari rekaman |
| test_cctv | 443 | 443 | frame event 18 November: 232 gambar ROI, 211 gambar area fokus; 146 di antaranya malam |
| test_public | 100 | 197 | split test Kaggle |

Angka di bagian ini bukan akurasi. Labelnya dari dua detektor dan OCR, jadi yang dihitung adalah berapa plat berlabel yang ditemukan sebuah model pada pandangan yang dipakai service. Plat yang tidak terlihat kedua detektor di pandangan mana pun tidak ada di label.

### Putaran 1 dan tiga kesalahan labelnya

Putaran pertama (`data/dataset/detector-20261118`, model `yolov8n-20261118.onnx`) memakai detektor service saja sebagai pemberi label, menerima semua event service sebagai bukti plat, dan menambahkan 486 gambar "latar kosong": key frame rekaman yang tidak memuat satu usulan pun. Hasilnya di 443 plat uji bagus (tabel di bawah), tetapi plat belakang truk tangki yang berdiri di cam-4 tidak diberi kotak sama sekali, juga pada ambang 0,05:

| Frame cam-4, area fokus | Plat | Detektor sekarang | Putaran 1 | Putaran 2 |
|---|---|---|---|---|
| 08.35.55 | `BK8865XD` | 0,30 | tidak ada kotak | 0,89 |
| 08.36.57 | `BK8865XD` | 0,71 | 0,86 | 0,88 |
| 08.53.11 | `BK8768CA` | 0,57 | 0,79 | 0,88 |
| 08.53.32 | `BK8768CA` | 0,41 | tidak ada kotak | 0,91 |
| 09.34.57 | `BK8621XC` | 0,35 | 0,86 | 0,88 |
| 09.37.22 | `BK8621XC` | 0,49 | 0,87 | 0,86 |

Penyebabnya ditemukan dengan melihat gambarnya:

1. **Gambar latar.** "Detektor tidak melihat apa-apa" bukan bukti frame itu kosong. Dari 60 gambar latar area fokus cam-4, sedikitnya 23 menampilkan truk tangki dari belakang dengan plat yang terlihat (sebagian tertutup pipa atau ban, sebagian kotor). Model diajari bahwa plat di posisi itu adalah latar. Gambar latar sekarang mati secara bawaan (`--backgrounds 0`).
2. **Event sebagai bukti.** Service pernah melaporkan blok trotoar kuning sebagai plat (`E6G` 0,44 dan `E6611` 0,48 di cam-3 pada 18 November dini hari) dan tulisan rompi (`SECUR1TY` 0,87). Kotaknya ikut menjadi label. Sekarang event hanya menjadi bukti bila berpola sah, berkode wilayah yang dikenal, dan confidence minimal 0,8.
3. **Plat separuh tertutup.** Teks seperti `D85HF`, `BM955`, dan `B33` adalah bagian plat yang masih terlihat di balik pipa atau ban. Plat tertutup tidak dipakai sebagai data latih (aturan pengguna 15 November 2026), jadi teks di bawah 6 karakter tidak lagi dihitung plat. Sembilan gambar dibuang dari dataset putaran 2 dengan aturan ini.

Satu hal lain muncul saat membuat ulang dataset: di GPU laptop (RTX 5050, `CUDAExecutionProvider`) model OCR membaca teks yang salah dari 400 di antara 400 crop (`BK8695XT` menjadi `CKWA1CWJC1`), sedangkan kedua detektor memberi kotak yang sama dengan CPU. Guru label karena itu selalu menjalankan OCR di CPU. Di GTX 1060 PC ROG OCR membaca benar.

### Putaran 2

```powershell
# lingkungan GPU: detektor di GPU, OCR tetap di CPU
.venv-gpu\Scripts\python.exe training/detector/build_dataset.py --out data/dataset/detector-20261118b `
    --public kaggle="../dataset/Indonesian License Plate Dataset" --test-from 2026-11-18T00:00 `
    --events data/rog-20261118 --videos data/samples/rec-20261113-a data/samples/rec-20261113-b
uv run --project training python training/detector/train.py train --model models/detector/src/best.pt `
    --data data/dataset/detector-20261118b/data.yaml --name yolov8n-20261118b --epochs 80
```

Dataset selesai dalam 61 menit, training 80 epoch dalam 60 menit di RTX 5050. Hasil lewat kode service (`compare_runtime.py`, ambang confidence service 0,35; skor 0,5 ke atas adalah syarat tracker memulai jejak):

| Model | Ditemukan dari 443 | Skor 0,5 ke atas | Terlewat | Kotak lebih |
|---|---|---|---|---|
| Detektor sekarang (`plat.onnx`) | 351 | 317 | 92 | 4 |
| Putaran 1 | 429 | 427 | 14 | 6 |
| Putaran 2 (`yolov8n-20261118b.onnx`) | 437 | 437 | 6 | 4 |

Plat dengan skor 0,5 ke atas, per pandangan:

| Pandangan | Plat | Sekarang | Putaran 1 | Putaran 2 |
|---|---|---|---|---|
| ROI, siang | 159 | 108 | 151 | 156 |
| ROI, malam | 73 | 37 | 71 | 71 |
| Area fokus, siang | 138 | 112 | 134 | 138 |
| Area fokus, malam | 73 | 60 | 71 | 72 |

Dan per lebar plat di frame kamera (plat yang lebih jauh lebih kecil):

| Pandangan dan lebar plat | Plat | Sekarang | Putaran 1 | Putaran 2 |
|---|---|---|---|---|
| ROI, di bawah 90 piksel | 13 | 4 | 9 | 11 |
| ROI, 90 sampai 105 | 74 | 41 | 72 | 73 |
| ROI, 105 sampai 125 | 114 | 80 | 110 | 112 |
| ROI, 125 ke atas | 31 | 20 | 31 | 31 |
| Area fokus, di bawah 90 piksel | 13 | 8 | 11 | 12 |
| Area fokus, 90 sampai 105 | 74 | 56 | 73 | 74 |
| Area fokus, 105 sampai 125 | 114 | 104 | 112 | 114 |
| Area fokus, 125 ke atas | 10 | 5 | 9 | 10 |

`compare_runtime.py` juga membandingkan plat demi plat dengan model pertama. Putaran 1 kehilangan dua plat yang ditemukan detektor sekarang (satu turun dari 0,74 ke 0,47, satu tidak diberi kotak). Putaran 2 tidak kehilangan satu pun: 120 plat naik ke 0,5 ke atas, 0 turun.

Enam plat yang terlewat putaran 2 juga terlewat detektor sekarang. Semuanya dilihat: dua kali tulisan `BG 7070 N` yang dicat di badan truk, dan empat plat di gambar ROI (satu setengah tertutup, satu rusak, satu gelap dengan orang di depannya, satu plat depan yang jelas). Empat kotak lebihnya: papan nama berlampu di truk (0,76), panel putih kosong di samping plat (0,58), deretan lampu di bull bar (0,52), dan satu plat asli yang di gambar uji ditutup abu-abu karena tidak ada yang memastikannya (0,66).

Di foto Kaggle (197 plat) yang ditemukan naik dari 141 menjadi 191. Kotak lebihnya 15, detektor sekarang 2. Kelima belasnya dilihat: 11 plat asli yang tidak berlabel di dataset itu, 4 salah (lampu depan 0,36, deret baut bumper 0,62, tas bermotif 0,83, papan nama warung 0,50). Di split val 397 dari 400 plat berskor 0,5 ke atas (detektor sekarang 332).

### Lewat jalur service

`scripts/compare_focus.py --detector` menjalankan rekaman lewat pipeline service (tracking, tunggu berhenti, OCR, voting) dengan detektor lain. Dua potongan rekaman 13 November dicoba, keduanya dengan area fokus. Rekaman ini ikut menyumbang 137 gambar ke train dan val, jadi ini pemeriksaan bahwa calon bekerja di jalur service, bukan ukuran pada data baru.

| Rekaman | Detektor | Frame dengan plat terlihat | Event |
|---|---|---|---|
| cam-3, 10 menit (plat depan) | sekarang | 4.519 dari 4.781 | `BK8985HF` 0,999 dan `BK8104GP` 0,999 |
| cam-3, 10 menit (plat depan) | putaran 2 | 4.615 dari 4.781 | `BK8985HF` 0,999 dan `BK8104GP` 0,999 |
| cam-4, 15 menit (truk tangki dari belakang) | sekarang | 56 dari 7.200 | tidak ada |
| cam-4, 15 menit (truk tangki dari belakang) | putaran 2 | 1.140 dari 7.200 | `B985HF` 0,914 (saat lewat) dan `BG351DPP` 0,405 |

Di cam-3 tidak ada yang berubah. Di cam-4 terlihat harga dari detektor yang lebih peka: ketiga truk di potongan itu platnya terhalang (pipa, ban cadangan, kotoran), dan plat yang dulu tidak ditemukan sekarang ditemukan lalu dibaca salah. `B985HF` adalah `BK8985HF` yang dua hurufnya tertutup pipa; truk yang sama terbaca benar dari depan di cam-3. Event bacaan salah dari plat terhalang akan lebih sering muncul di kamera yang melihat truk dari belakang. Service sekarang pun sudah menghasilkannya dari plat belakang (`K8768CA` 0,99 di cam-4 pada 18 November pukul 08.53).

### Yang belum diukur

- Seberapa sering event bacaan salah itu muncul dalam sehari, dan berapa event benar yang bertambah di malam hari. Tidak ada rekaman malam; ukurannya baru ada setelah calon berjalan di kamera sungguhan.
- Apakah OCR membaca benar plat di bawah 90 piksel yang sekarang ditemukan.
- Plat yang tidak terlihat kedua detektor pemberi label.
- Tiga kotak salah di frame kamera berskor di atas 0,5, jadi tracker akan mengikutinya dan OCR akan mencoba membacanya. Detektor sekarang punya kelemahan yang sama terhadap tulisan (event `SATPAM`, `SECUR1TY`).

### Pemasangan dan rekaman

Pengguna meminta calon itu dipasang ("pasang sekarang") dan PC ROG merekam video untuk sementara, supaya bisa dilihat dari jarak berapa plat terbaca dan tidak.

- Berkas disalin ke `C:\alpr-id\models\detector\candidates\yolov8n-20261118b.onnx` (SHA-256 sama dengan di laptop). Di GTX 1060, 200 kali jalan selagi service berjalan: median 17,3 ms dan p95 20,4 ms; `plat.onnx` 17,2 ms dan 20,3 ms.
- `deploy\rog\set-detector-model.ps1` mengganti `detector.model_path` pada 15.59.02, saat tidak ada plat di depan kamera mana pun. Tiga puluh tiga detik kemudian: empat kamera tersambung, 8 fps diproses per kamera, kedua model di `CUDAExecutionProvider`. Tiga menit kemudian, dengan truk di depan tiga kamera: total per frame median 45 sampai 66 ms, 8 fps tetap tercapai, GPU 63%, VRAM 1.480 MiB.
- Event pertama sesudahnya: `BK8286XG` (cam-1, 0,992), `BK8064FC` (cam-4, 0,996), dan truk yang sama dari belakang di cam-3 sebagai `K8064FC` (0,856).
- Kembali ke detektor lama: `set-detector-model.ps1 -Model models/detector/plat.onnx` di PC ROG, dan baris yang sama di `config\app.yaml` laptop.

Rekaman: `scripts/record_long.py` menyalin paket video keempat kamera tanpa encode ulang ke `C:\alpr-id\data\samples\rec-long-20261118\cam-<n>\`, satu berkas tiap 30 menit yang dinamai menurut jam server saat paket pertamanya. Mulai 18 November pukul 16.01, berhenti sendiri 19 November pukul 18.00 atau bila sisa disk C di bawah 150 GB (waktu mulai 385 GB). Uji 75 detik sebelumnya memberi 2 sampai 10 MB per 30 detik per kamera, jadi satu hari kira-kira 40 sampai 90 GB. Prosesnya berdiri sendiri, bukan task terjadwal: bila PC restart, rekaman tidak lanjut. Membuat berkas bernama `STOP` di folder itu menghentikannya lebih awal.

Rekaman itu belum dianalisis. Rencananya: potongan di sekitar tiap event, dan di sekitar truk yang tidak menghasilkan event, dijalankan lewat `scripts/compare_focus.py`, lalu lebar plat saat pertama ditemukan dan saat pertama terbaca benar dicatat per kamera, siang dan malam.

### Zoom digital seperti di ai_cctv

Layanan lama memakai `ZOOM_FACTOR = 2.5`: bagian tengah frame, 1/2,5 lebar dan tingginya (1.024 × 576 piksel dari 2.560 × 1.440), dipotong dan hanya itu yang diberikan ke detektor. Pengguna meminta cara itu dicoba di laptop. `training/detector/compare_views.py` memberi detektor pandangan itu pada frame event 18 November yang sama dan menghitung plat berlabel yang mendapat kotak berskor 0,5 ke atas, untuk tiap gabungan pandangan:

| Pandangan yang dipakai | Detektor terpasang, dari 232 plat | Detektor lama, dari 232 plat |
|---|---|---|
| ROI saja | 227 | 147 |
| Area fokus saja | 210 | 175 |
| Zoom 2,5 saja (cara ai_cctv) | 219 | 163 |
| ROI dan area fokus (service sekarang) | 231 | 201 |
| ROI dan zoom 2,5 | 231 | 184 |
| ROI, area fokus, dan zoom 2,5 | 231 | 205 |

Dengan detektor yang sekarang terpasang, zoom menemukan semua 219 plat yang ada di dalam potongannya, tetapi 13 plat lain berada di luar potongan itu. Sebagai pengganti ia kehilangan 12 plat dibanding cara sekarang, semuanya di siang hari; sebagai pandangan ketiga ia tidak menambah satu pun. Tidak ada kotak kuat yang hanya ditemukan pandangan zoom, jadi ia juga tidak memperlihatkan plat yang belum berlabel. Dengan detektor lama pandangan ketiga itu menambah 4 plat, dengan harga satu panggilan detektor lagi per frame (17 ms di GTX 1060).

Zoom karena itu tidak dipasang. Dari 232 plat berlabel tinggal satu yang belum ditemukan, jadi yang masih bisa didapat adalah plat yang lebih jauh daripada semua yang berlabel, dan itu baru terlihat dari rekaman 18 sampai 19 November. Zoom juga tidak menolong OCR: crop plat selalu diambil dari frame asli lalu diubah ke ukuran masukan model OCR.

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
