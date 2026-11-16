# Fase 1: audit proyek lama dan rencana alpr-id

Disusun 13 November 2026. Dokumen ini mencatat apa yang ditemukan di proyek lama, apa yang sudah dipastikan dari dokumentasi dan pengukuran, rancangan yang diusulkan, dan hal yang masih perlu keputusan.

## 1. Cakupan audit

| Folder | Isi | Cara diperiksa |
|---|---|---|
| `ai_cctv` | Layanan pembaca plat yang sekarang berjalan (Flask, YOLO, PaddleOCR) | Semua file di luar venv dibaca; model diperiksa tanpa dimuat ke PyTorch |
| `timbangan-ocr-service` | OCR dokumen surat masuk dan keluar, bukan plat | Dibaca untuk mencari bagian yang bisa dipakai ulang |
| `inl_register` | Aplikasi registrasi dan penimbangan truk (Laravel) | Dibaca untuk mencari siapa yang memakai layanan plat lama |

Tidak ada file kerja, model, atau konfigurasi di ketiga folder yang diubah. Satu efek samping terjadi: perintah `git diff --stat HEAD` yang dijalankan agen audit menulis ulang `.git/index` di `ai_cctv/project0` dan `ai_cctv/project0/detection-plate` pada pukul 10:41. Yang berubah hanya cache stat; isi staging, branch, dan commit tetap. Sejak itu tidak ada perintah git yang dijalankan di folder lama.

## 2. Temuan di `ai_cctv`

Isinya satu aplikasi Flask dalam tiga salinan yang hampir sama. Yang dijalankan di server adalah `project0/file_lama/app.py`, lewat Task Scheduler bernama "AI CCTV - Plate Monitor".

| Bagian | Temuan | Dipakai di alpr-id |
|---|---|---|
| Kamera | Empat URL RTSP Hikvision, main stream. Empat file memuat urutan kamera yang berbeda; yang berlaku di produksi: 1=.205, 2=.201, 3=.206, 4=.200 | Ya, dipindah ke `.env` |
| `best.pt` | YOLOv8n satu kelas bernama `plat`, dilatih pada ukuran 640 dengan ultralytics 8.3.176 pada 11 September 2025. Dua salinannya identik | Ya, sebagai detektor baseline dan model auto-label |
| `normalize_plate` dan `VALID_PREFIX` | Membersihkan hasil OCR dan memeriksa kode wilayah. Daftar kodenya kurang 11 kode (D, E, F, DP, DW, PA, PB, PG, PS, PT, PY), jadi plat Bandung, Cirebon, dan Bogor ditolak | Perilakunya dijadikan bahan test case; kodenya tidak dipakai |
| Launcher autostart | Restart otomatis, log berputar, proses anak ikut mati saat task dihentikan | Polanya dipakai untuk `install-autostart.ps1` |
| Model PaddleOCR lokal dan `yolov8_n_500e_coco.onnx` | Tidak dirujuk kode mana pun | Tidak |
| Data | Tidak ada gambar, video, dataset, label, crop, atau catatan hasil baca | Dataset dikumpulkan dari nol |

Metadata `best.pt` mencatat mAP50-95 0,751 dan recall 0,908 pada data validasi pelatihnya. Dataset itu tidak ada di sini, jadi angka tersebut bukan ukuran akurasi pada CCTV timbangan.

Penanganan huruf M dan W yang terbaca H tidak ditemukan di ketiga salinan `app.py`, dan tidak ada langkah perbaikan citra selain zoom digital 2,5 kali.

Kelemahan layanan lama yang menentukan rancangan baru:

- Tidak ada tracking, voting, confidence, maupun penyimpanan. Pembacaan terakhir menimpa yang sebelumnya, dan setahun operasi tidak meninggalkan data.
- Satu faktor zoom untuk semua kamera: bagian tengah frame dipotong lalu diperbesar kembali dengan interpolasi, dan crop OCR diambil dari gambar hasil perbesaran itu.
- OCR memakai PaddleOCR bahasa Inggris generik di CPU, dijalankan tiap YOLO kelima pada dua kotak terbesar tanpa menilai ketajaman.
- Baris masa berlaku ikut tergabung: `BK 1234 09.27 ABC` menjadi `BK 12340927 ABC`.
- Karakter depan yang salah baca dibuang diam-diam: `8K 1234 ABC` menjadi `K 1234 ABC`.
- Plat di luar pola dibuang, bukan dikembalikan dengan penanda.
- Kredensial kamera tertulis di kode, tercetak di log, dan ikut ter-commit ke dua repo GitHub (untuk alamat kamera yang lebih lama).

## 3. Temuan di proyek tetangga

### Siapa yang memakai layanan plat lama

Layanan lama beralamat `http://192.168.1.163:5000`. Dua klien bergantung padanya:

- **Aplikasi desktop VB.NET di PC timbangan.** Empat timer memanggil `GET /plate/1` sampai `/plate/4` tiap detik dan hanya membaca field `plate` dan `status`. Nilai `plate` dipakai kalau `status` bernilai `"online"`, lalu dibandingkan dengan `registrasi.no_polisi` memakai kesamaan teks persis setelah `Trim` dan `ToUpper`, jadi spasi ikut menentukan. Alamat layanan ditulis langsung di kode desktop.
- **Web `inl_register`.** Pop-up cek plat menampilkan `GET /video/1..4` (MJPEG) langsung dari browser admin.

Desktop memilih kamera dari jenis registrasi dan tahap timbang: OUTGOING timbang pertama kamera 1, OUTGOING timbang kedua kamera 2, INCOMING timbang pertama kamera 3, INCOMING timbang kedua kamera 4. Sumber yang dibaca adalah `inl_register/docs/WeightbridgeForm.vb`; belum tentu sama dengan program yang terpasang.

Akibatnya untuk alpr-id: supaya bisa menggantikan layanan lama tanpa mengubah desktop, perlu rute `GET /plate/{n}` dan `GET /video/{n}` dengan bentuk jawaban yang sama, di alamat dan port yang sama.

### Yang bisa dipakai ulang dari `timbangan-ocr-service`

- Label Studio 1.23.2 sudah terpasang di `D:\labelstudio`. Konfigurasi labelnya memakai kotak ditambah transkripsi per kotak, cocok untuk memberi label detektor dan OCR sekaligus.
- Script konversi ekspor Label Studio ke format YOLO, pembuat pra-label untuk diimpor ke Label Studio, pembagi dataset dengan test set yang dikunci lewat hash isi file, dan pembanding dua model pada split yang sama.
- `app/core/domain_repair.py`: peta karakter mirip dua arah dan tiga batas pengaman supaya bacaan yang rusak parah tidak diubah menjadi tampak valid.
- Resep training yang terbukti di laptop ini: torch 2.7.1+cu128 pada RTX 5050.

Gambar di proyek itu semuanya dokumen kertas dan memuat data pribadi supir, jadi tidak disalin.

## 4. Hasil ukur kamera

Diukur 13 November 2026 pukul 15:14 dari laptop pengembangan dengan `scripts/probe_rtsp.py`, enam detik per kamera:

| Kamera | IP | Codec | Resolusi | fps | Bitrate | Waktu buka |
|---|---|---|---|---|---|---|
| cam-1 | 192.168.1.205 | H.265 | 2560×1440 | 20 | 2,3 Mbps | 1,3 detik |
| cam-2 | 192.168.1.201 | H.265 | 2560×1440 | 20 | 1,8 Mbps | 5,3 detik |
| cam-3 | 192.168.1.206 | H.264 | 2560×1440 | 20 | 2,2 Mbps | 1,5 detik |
| cam-4 | 192.168.1.200 | H.265 | 2560×1440 | 20 | 1,7 Mbps | 1,2 detik |

Tidak ada frame yang hilang pada pengukuran ini. Pengukuran pertama pagi harinya menunjukkan frame hilang dan satu koneksi terputus di tiga kamera; saat itu laptop tersambung lewat hotspot dan penyebab pastinya tidak diselidiki.

Sub stream tiga kamera hanya 640×360 (cam-3: 1280×720), terlalu kecil untuk membaca plat.

Keempat kamera dipasang rendah di ujung jalur timbangan dengan lensa lebar. cam-1 dan cam-2 melihat timbangan yang sama dari dua ujung, begitu juga cam-3 dan cam-4. Belum ada frame dengan truk di atas timbangan, jadi ukuran plat dalam piksel belum diketahui. Frame contoh ada di `data/samples/`.

## 5. Hasil cek dokumentasi

Riset dilakukan terhadap dokumentasi resmi, kode sumber, dan PyPI. Agen pemeriksa ulang gagal berjalan karena batas pemakaian, jadi hanya baris bertanda "introspeksi" yang sudah dicek dua kali, yaitu dengan memasang paketnya dan memeriksa langsung.

| Paket | Versi | Yang dipastikan | Status |
|---|---|---|---|
| ultralytics | 8.4.211 | `yolo26n.pt` dan `yolo26s.pt` tersedia; argumen `nms` dan `quantize` menggantikan `end2end`, `half`, `int8`; export `nms=False` menghasilkan `(1, 300, 6)`; tracker bawaan bukan lagi ByteTrack | introspeksi |
| fast-plate-ocr | 1.1.0 | `LicensePlateRecognizer`, `run(..., return_confidence=True)`, `PlatePrediction.char_probs`; `cct-s-v2-global-model` ada, input 64×128 RGB, 10 slot | introspeksi |
| open-image-models | 0.6.0 | `create_detector`, enam nama model plat YOLOv9 | introspeksi |
| trackers | 2.7.0 | `ByteTrackTracker` dan `update(detections, frame, timestamp)`; `sv.ByteTrack` sudah dihapus dari supervision | introspeksi |
| fastapi | 0.162.5 | SSE bawaan di `fastapi.sse` | introspeksi |
| av (PyAV) | 19.0.2 | Membaca keempat kamera asli; butuh Python 3.12 | diuji ke kamera |
| fast-plate-ocr (training) | 1.1.0 | Perintah `train`, `valid`, `export`; fine-tune lewat `--weights-path`; export gagal di Windows (issue #103); TensorFlow tanpa GPU di Windows native | riset saja |
| ultralytics (training) | 8.4.211 | Argumen `train`, `val(split="test")`, nama atribut metrik, argumen `augmentations` | riset saja |
| Aturan plat | - | Perpol 7/2021: kode wilayah, 1-4 angka, seri huruf; 57 kode wilayah di lampiran ditambah PG, PS, PT, PY (2024) | riset saja |
| onnxruntime-gpu | 1.26.0 | Rilis PyPI terakhir dengan build CUDA 12; wheel Windows untuk Python 3.12 ada dan ter-resolve bersama cuDNN 9.10.2.21 | riset, resolusi dicek dengan `uv lock` |

Catatan dari riset yang memengaruhi rencana:

- **Indonesia sudah ada di data latih `cct-s-v2-global-model`.** Penulisnya melaporkan akurasi plat 0,979 untuk Indonesia pada data validasi miliknya sendiri. Itu bukan data CCTV timbangan dan tidak boleh dikutip sebagai akurasi proyek ini.
- **Ultralytics tidak menerbitkan bobot khusus plat.** Bobot YOLO26 bawaan dilatih pada COCO yang tidak punya kelas plat, jadi YOLO26 baru bisa dipakai setelah dilatih dengan data sendiri.
- **Lisensi Ultralytics.** Paket dan bobotnya berlisensi AGPL-3.0. Halaman lisensi Ultralytics menyatakan alat internal perusahaan yang tidak dibuka source-nya memerlukan Enterprise License, dan bobot hasil training ikut AGPL-3.0. Ini berlaku juga untuk `best.pt` lama.
- **Data publik.** Tidak ada dataset publik yang bersumber dari CCTV, dan plat putih baru nyaris tidak ada di data publik yang berlabel teks. Yang paling dekat dengan kasus truk: INDO-ALPR (6.174 crop berlabel teks, 1.495 plat kuning, CC BY 4.0) dan ICVLP (468 crop plat kuning kendaraan niaga). Beberapa dataset memakai ulang gambar yang sama, jadi penggabungan harus disertai deduplikasi.

## 6. Rancangan

```
kamera RTSP ──> reader per kamera (PyAV, frame terbaru, reconnect)
                    │ ROI dari frame resolusi asli
                    ▼
              detektor plat (ONNX Runtime) ──> ByteTrack per kamera
                                                    │ crop terbaik per track, paling banyak N kali
                                                    ▼
                                    OCR (fast-plate-ocr, ONNX Runtime)
                                                    │ probabilitas per karakter
                                                    ▼
                          koreksi berbasis posisi ──> voting per track ──> satu event
                                                                              │
                                              SQLite + crop + snapshot <──────┤
                                              REST API, SSE, rute lama <──────┘
```

| Tahap | Keputusan | Alasan |
|---|---|---|
| Capture | PyAV, satu thread per kamera, hanya frame terakhir yang disimpan, timeout dan reconnect bertahap | Sudah terbukti membaca keempat kamera; opsi per stream; backend FFmpeg OpenCV tidak bisa diatur ukuran buffernya |
| ROI | Kotak per kamera di config, dipotong dari frame 2560×1440 | Plat kecil di lensa lebar; zoom digital layanan lama membuang detail |
| Deteksi | ONNX Runtime langsung, tanpa PyTorch di runtime. Baseline: `best.pt` lama diekspor ke ONNX, dibandingkan dengan `yolo-v9-t-640-license-plate-end2end` | Runtime lebih ringan; kedua model sudah punya kelas plat |
| Tracking | `trackers.ByteTrackTracker` per kamera dengan `timestamp` capture | Berlisensi Apache-2.0 dan tidak membawa PyTorch |
| OCR | `cct-s-v2-global-model` dari file lokal; crop dipilih dari skor ketajaman dan ukuran | Model sudah mengenal plat Indonesia; fine-tune di Fase 4 |
| Koreksi | Berdasarkan posisi, dengan batas jumlah penukaran | Mengikuti pola yang sudah teruji di `domain_repair.py` |
| Voting | Menjumlahkan probabilitas penuh tiap slot karakter antar-frame | Lebih tahan terhadap karakter yang selisihnya tipis daripada voting hasil akhir saja |
| Event | Keluar saat hasil stabil atau track berakhir; plat sama di kamera sama dalam jendela waktu dianggap satu kendaraan | Truk berhenti lama dan track bisa terputus oleh orang lewat |
| Penyimpanan | SQLite WAL, satu thread penulis, crop dan snapshot per tanggal | Sekaligus menjadi sumber data untuk Fase 3 |
| API | FastAPI, SSE bawaan, satu worker | Tiap worker akan membuka stream kamera sendiri |
| Deployment | Windows langsung: `start-alpr.bat` dan `install-autostart.ps1`; Dockerfile dan compose disediakan | Layanan lama juga berjalan lewat Task Scheduler; GPU di Docker Windows perlu WSL2 |

### Runtime GPU

GTX 1060 adalah kartu generasi Pascal, dan dukungan perangkat lunak untuk generasi itu sudah berhenti di semua lapisan. Menurut catatan rilis NVIDIA, Microsoft, dan PyTorch:

| Lapisan | Versi terakhir yang mendukung GTX 1060 | Yang tidak bisa dipakai |
|---|---|---|
| Driver NVIDIA | Branch 580 (terbaru 582.78, 30 September 2026) | - |
| CUDA | 12.x (terakhir 12.9) | 13.x |
| cuDNN | 9.10.2 | 9.11 ke atas |
| `onnxruntime-gpu` di PyPI | 1.26.0 (build CUDA 12) | 1.27 ke atas (build CUDA 13) |
| TensorRT | 8.6.1, tanpa paket pip untuk Windows | 10.x dan 11.x |
| PyTorch | 2.14.1+cu126 | 2.15 ke atas |

Keputusan yang mengikuti:

- **Provider utama: CUDA Execution Provider dari `onnxruntime-gpu==1.26.0`** dengan `nvidia-cudnn-cu12==9.10.2.21`. Pin ini sudah ada di extra `gpu` pada `pyproject.toml` dan ter-resolve dengan wheel Windows. DLL CUDA ikut terpasang lewat pip dan dimuat dengan `onnxruntime.preload_dlls()`, jadi tidak perlu memasang CUDA di sistem.
- **TensorRT dicoret untuk GTX 1060.** Arsitektur di permintaan menyebut TensorRT bila ada GPU; untuk kartu ini jalurnya tidak tersedia.
- **Cadangan:** OpenVINO di CPU (`openvino` 2026.4.1 masih mendukung Core generasi 7 dan 8), lalu DirectML (`onnxruntime-directml` 1.24.4, sudah tidak dikembangkan).
- **Decode:** GTX 1060 mendukung decode H.264 dan H.265 lewat NVDEC, dan wheel PyAV untuk Windows sudah memuatnya. Dipakai atau tidaknya ditentukan dari pengukuran beban CPU.
- **Laptop training (RTX 5050)** memerlukan build PyTorch CUDA 12.8 atau lebih baru. Build itu tidak berjalan di GTX 1060, jadi lingkungan training dan lingkungan inferensi memang terpisah.

Semua baris di atas berasal dari dokumentasi dan metadata paket. Belum ada yang diuji di GTX 1060. Langkah pertama Fase 2 di PC target adalah memasang extra `gpu`, menjalankan satu model di `CUDAExecutionProvider`, dan memastikan provider yang aktif bukan CPU. Driver di HP OMEN (581.63) sudah memenuhi syarat, tetapi sebaiknya diperbarui ke 582.78.

Angka latency untuk GTX 1060 persis tidak ditemukan di sumber mana pun. Yang paling dekat adalah catatan proyek Frigate: 14 sampai 16 ms per frame untuk model YOLO kecil ukuran 320 di GTX 1070 dengan ONNX Runtime CUDA. Itu angka pihak lain pada kartu dan ukuran input yang berbeda.

### API

Sesuai permintaan: `GET /health`, `GET /cameras`, `GET /cameras/{id}/current`, `GET /events?camera=&since=&limit=`, `GET /events/stream` (SSE), `POST /recognize`. Bentuk event mengikuti contoh di permintaan, dengan field tambahan yang diusulkan: `region_known` dan `alternatives`.

Rute tambahan untuk klien lama, bisa dimatikan di config:

- `GET /plate/{n}` mengembalikan `{"plate", "raw", "fps", "status"}` dengan HTTP 200 untuk kamera 1 sampai 4, diambil dari memori. Nomor kamera lain dijawab 404, sama seperti layanan lama.
- `GET /video/{n}` mengembalikan MJPEG.

## 7. Penanganan khusus

**Huruf M dan W yang terbaca H.** Pada plat Indonesia, M dan W digambar dengan coretan tengah yang pendek. Kalau coretan itu hilang, bentuknya menjadi H. Ketiganya huruf yang sah di kode wilayah maupun seri belakang, jadi aturan penukaran tidak bisa dipakai. Penanganannya berlapis:

1. Crop diambil dari frame resolusi asli dan masuk ke OCR tanpa sharpening, thresholding, denoise, atau upscaling.
2. Voting memakai probabilitas penuh tiap karakter. Satu frame tajam yang yakin pada M bisa mengalahkan beberapa frame buram yang condong ke H.
3. H, M, W didaftarkan sebagai kelompok huruf mirip. Bila pemenang dan runner-up berasal dari kelompok itu dan selisihnya tipis, event memuat bacaan alternatif. Karakter yang masih seimbang juga menahan event supaya tidak terbit lebih awal.
4. Fine-tune OCR di Fase 4 harus memuat cukup contoh M dan W dari kamera timbangan.
5. Laporan benchmark mencatat jumlah kekeliruan H dengan M dan H dengan W secara terpisah.

Seberapa sering model pretrained keliru pada kasus ini belum diketahui. Angkanya baru ada setelah crop plat ber-M dan ber-W terkumpul di Fase 2.

**Baris masa berlaku.** Label OCR hanya nomor plat. Apakah model pretrained ikut membaca digit masa berlaku belum diketahui dan akan diukur. Dua pengaman disiapkan: memotong bagian bawah crop (`ocr.crop_bottom_ratio`) dan membuang empat digit di belakang seri huruf saat post-processing.

**Plat di luar pola.** TNI dan Polri (angka, strip, kode satuan), CD, CC, CH, dan RI dikembalikan dengan `format_valid: false`. Alfabet model hanya angka dan huruf besar, jadi tanda strip tidak akan muncul di hasil.

## 8. Rencana per fase

### Fase 2: runtime baseline, API, benchmark

1. Skema config (pydantic) dan logging terstruktur.
2. Reader RTSP, ROI, dan status kamera.
3. Detektor ONNX Runtime dengan dua model baseline, tracker, OCR, pemilihan crop.
4. Validasi format, koreksi karakter, voting, dengan pytest. Kasus uji diambil dari perilaku `normalize_plate` lama dan contoh di `domain_repair.py`.
5. SQLite, penyimpanan crop dan snapshot.
6. Semua endpoint API dan rute lama.
7. `start-alpr.bat`, `install-autostart.ps1`, Dockerfile, docker-compose.
8. Benchmark latency per tahap (decode, deteksi, tracking, OCR, post-processing) di laptop dan di PC target.

Fase 2 menghasilkan angka latency. Angka akurasi baru bisa keluar setelah ada test set berlabel dari CCTV; pengumpulan datanya dimulai begitu runtime Fase 2 berjalan dan menyimpan crop.

### Fase 3: pipeline dataset dan training

1. Ekstraksi frame dari RTSP, video, atau snapshot hasil runtime.
2. Auto-label dengan model baseline, keluarannya berupa pra-label yang dikoreksi di Label Studio.
3. Konversi ekspor Label Studio ke format YOLO (detektor) dan CSV `image_path, plate_text` (OCR).
4. Split tetap. Test set hanya dari CCTV, dikunci lewat hash, dibagi per kunjungan kendaraan dan per tanggal. Data publik dideduplikasi dan hanya masuk train dan val.
5. Augmentasi: blur gerak, cahaya rendah, hujan, perspektif miring, plat kotor.
6. Script training dan evaluasi detektor (mAP50-95, recall) dan OCR (akurasi plat utuh, akurasi per karakter).
7. Laporan benchmark: akurasi end-to-end, latency per tahap, rincian per warna plat dan per pasangan karakter yang sering tertukar.

### Fase 4: fine-tune dan optimasi

1. Latih YOLO26n dan YOLO26s, bandingkan dengan baseline pada test set yang sama.
2. Fine-tune `cct-s-v2-global-model`.
3. Optimasi kecepatan: opsi CUDA provider, batch antar-kamera, hardware decode, dan OpenVINO sebagai pembanding. TensorRT tidak tersedia untuk GTX 1060.
4. Pilih konfigurasi yang memenuhi target: akurasi plat end-to-end minimal 95% dan latency paling lama 100 ms per frame per kamera untuk empat kamera.

## 9. Risiko dan hal yang belum pasti

| Hal | Keadaan sekarang | Dampak |
|---|---|---|
| GPU di GTX 1060 | Hanya ONNX Runtime 1.26.0 dengan CUDA 12; TensorRT tidak tersedia; belum diuji di kartunya | Optimasi kecepatan Fase 4 tidak bisa mengandalkan TensorRT. Tumpukan ini juga tidak akan mendapat pembaruan lagi |
| Beban decode empat stream 2560×1440 | Belum diukur di PC target | Mungkin perlu membatasi fps yang diproses atau memakai hardware decode |
| Dataset | Tidak ada sama sekali | Fase 3 bergantung pada pengumpulan dari kamera; akurasi 95% belum bisa dinilai |
| Fine-tune dan export OCR di Windows | TensorFlow tanpa GPU di Windows native, export gagal di Windows, WSL belum terpasang di laptop | Perlu WSL2, mesin Linux, atau script export sendiri |
| Lisensi Ultralytics | AGPL-3.0 atau Enterprise | Keputusan perusahaan sebelum YOLO dipakai di produksi |
| Pencocokan persis di desktop | Format spasi harus sama dengan `registrasi.no_polisi`, yang ditulis dengan dan tanpa spasi | Satu format keluaran tidak akan cocok dengan semua baris |
| Docker | Belum terpasang di laptop | Dockerfile dan compose tidak bisa diuji di sini |
| OpenCV 5 | Ter-resolve ke 5.0.0.93 | Sudah jalan dengan fast-plate-ocr dan trackers saat dicek; belum diuji dalam beban penuh |
| Kredensial lama di GitHub | Password kamera untuk alamat lama ada di commit yang sudah di-push | Perlu dicek apakah repo publik dan password itu masih dipakai |

## 10. Keputusan pengguna dan yang masih terbuka

Diputuskan pengguna pada 13 November 2026:

1. alpr-id berjalan di PC ROG Strix (i7-8700, Windows 11, GTX 1060 3 GB), IP `192.168.1.37`. Aplikasi desktop dikompilasi ulang supaya menunjuk ke alamat baru. alpr-id memakai port 5000, sehingga yang berubah di klien hanya alamat host: `http://192.168.1.37:5000`.
2. Rute lama `/plate/{n}` dan `/video/{n}` disediakan.
3. Pemetaan kamera benar: 1 Outgoing masuk, 2 Outgoing keluar, 3 Incoming masuk, 4 Incoming keluar.

Masih terbuka:

4. Lisensi Ultralytics: AGPL-3.0 atau Enterprise. Ini keputusan perusahaan, tidak bergantung pada PC yang dipakai.
5. Untuk Fase 3 dan 4: boleh memasang WSL2 di laptop, atau ada mesin Linux untuk fine-tune OCR.
6. Untuk Fase 3: boleh tidaknya memakai `registrasi.no_polisi` sebagai petunjuk label, karena tiap penimbangan sudah punya plat terdaftar.
7. Contoh gambar plat ber-M atau ber-W yang terbaca H, bila ada, untuk dijadikan kasus uji.

Catatan untuk kompilasi ulang desktop (`WeightbridgeForm.vb`), di luar cakupan alpr-id:

- Alamat layanan plat sebaiknya dibaca dari file konfigurasi, bukan ditulis di kode, dan IP PC ROG dibuat tetap.
- Perbandingan plat sebaiknya mengabaikan spasi di kedua sisi, karena `registrasi.no_polisi` tersimpan dengan dan tanpa spasi.
- Blok penanganan error kamera 1 menulis ke kotak teks kamera 2 (baris 202).
- `HttpClient` dibuat baru tiap detik tanpa batas waktu; jawaban yang lambat akan menumpuk permintaan.
