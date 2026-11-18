# Fase 2: runtime baseline, API, dan benchmark

Disusun 13 November 2026. Dokumen ini mencatat apa yang dibangun di Fase 2, apa yang benar-benar dijalankan untuk memeriksanya, angka yang terukur, dan apa yang belum diuji.

## 1. Yang dibangun

| Bagian | Isi |
|---|---|
| Capture | Satu thread PyAV per kamera yang hanya menyimpan frame terbaru, dengan timeout, reconnect bertahap, jeda panjang bila login ditolak, dan pengukur ketertinggalan terhadap timestamp stream |
| Deteksi | Detektor ONNX Runtime pada ROI per kamera. Membaca tiga bentuk keluaran: `(N, 300, 6)`, `(N, 4+kelas, anchor)`, dan `(M, 7)` |
| Tracking | `trackers.ByteTrackTracker` per kamera. Status kendaraan diteruskan bila tracker memberi ID baru di posisi yang sama |
| OCR | fast-plate-ocr `cct-s-v2-global-model`, mengambil probabilitas penuh tiap karakter. Paling banyak 5 bacaan per kendaraan, dipilih dari crop terbaik |
| Post-processing | Validasi format, koreksi O/0, I/1, B/8, S/5, Z/2 berdasarkan posisi, pembuangan digit masa berlaku, voting per karakter, kelompok huruf mirip H/M/W |
| Event | Satu event per kendaraan per kamera, disimpan di SQLite bersama crop plat dan snapshot frame |
| API | `/health`, `/cameras`, `/cameras/{id}/current`, `/events`, `/events/stream` (SSE), `/recognize`, `/media`, serta rute lama `/plate/{n}` dan `/video/{n}` |
| Menjalankan | `start-alpr.bat` (klik dua kali) dan `install-autostart.ps1` (Task Scheduler dan aturan firewall) |
| Alat | `alpr-id benchmark`, `alpr-id recognize`, `scripts/probe_rtsp.py`, `scripts/record_rtsp.py`, `scripts/fetch_models.py` |
| Lain-lain | Dockerfile, docker-compose, lingkungan `training/` terpisah untuk export model, 128 test pytest |

Model yang dipakai:

- Detektor: `best.pt` dari layanan lama (YOLOv8n, satu kelas `plat`), diekspor ke ONNX ukuran 640 dengan NMS tertanam.
- OCR: `cct-s-v2-global-model` apa adanya, belum di-fine-tune.
- YOLO26 belum dipakai. Bobot bawaannya tidak punya kelas plat, jadi baru masuk setelah ada data latih (Fase 4).

## 2. Yang benar-benar dijalankan

| Yang diperiksa | Cara | Hasil |
|---|---|---|
| Logika (format, koreksi, voting, pipeline, penyimpanan, API, SSE) | `pytest`, 128 test | lulus semua |
| Lint | `ruff check .` | bersih |
| Service dengan empat kamera asli | `alpr-id serve` selama beberapa menit, lewat hotspot dari PC ber-LAN | empat kamera tersambung, 8 fps diproses per kamera, tanpa reconnect |
| Semua endpoint | dipanggil ke service yang berjalan | menjawab sesuai rancangan, termasuk 404 dan 400 untuk masukan salah |
| SSE | test dengan server HTTP sungguhan: event diterbitkan dari thread lain dan terbaca di klien | lulus |
| `start-alpr.bat` | dijalankan, `/health` dan `/plate/3` menjawab | berjalan |
| `install-autostart.ps1` | `-Action Install` di PC ROG | task `ALPR-ID` terdaftar dan berjalan, aturan firewall terbuat, API terjangkau dari laptop lewat Tailscale. Perilaku setelah PC di-restart **belum diuji** |
| Jalur GPU | extra `gpu` dipasang di laptop (RTX 5050), kedua model dijalankan | `CUDAExecutionProvider` aktif untuk detektor dan OCR |
| Docker | tidak dijalankan | **belum diuji**, Docker tidak terpasang |
| GTX 1060 | dipasang di PC ROG lewat SSH, benchmark 90 detik, lalu service dijalankan sebagai task terjadwal | CUDA aktif, empat kamera 8 fps, lihat bagian 4 |

## 3. Bacaan baseline

Ini contoh hasil, bukan angka akurasi. Akurasi baru bisa dihitung setelah ada test set berlabel dari CCTV (Fase 3). Kolom "plat sebenarnya" dibaca dengan mata dari crop.

| Sumber | Kamera | Plat sebenarnya | Hasil event | Confidence | Bacaan per frame |
|---|---|---|---|---|---|
| Rekaman 10 menit | cam-2 | BM 9719 XX (putih, tulisan merah) | BM 9719 XX | 0,94 | `BM9719XX`, `BU9719XX` |
| Rekaman 10 menit | cam-1, plat belakang, sebagian tertutup selang | BM 9719 XX | BM 9719 XX | 0,90 | `BM9719KX`, `BM9719XX` |
| Langsung | cam-3 | BK 8985 HF (kuning) | BK 8985 HF | 1,00 | `BK8985HF` dua kali |
| Langsung | cam-3 | BK 8354 CP (kuning) | BK 8354 CP | 0,98 | `BK8354CP` dua kali |
| Langsung, kode final | cam-1 | BK 8801 XG (kuning) | BK 8801 XG | 1,00 | `BK8801XG` dua kali |

Yang terlihat dari contoh ini:

- Kelima penampakan (empat truk, satu di antaranya terlihat dari dua kamera) menghasilkan event yang benar. Lima contoh terlalu sedikit untuk disebut angka akurasi.
- Bacaan per frame bisa salah satu karakter (M terbaca U, X terbaca K). Voting antar-frame membetulkannya di kedua kasus.
- Baris masa berlaku di bawah nomor tidak ikut terbaca pada kelima contoh.
- Plat truk yang berhenti di timbangan hanya selebar 85 sampai 115 piksel pada frame 2560×1440. Detektor masih menemukannya, tetapi ini dekat batas bawah.
- Belum ada contoh plat ber-huruf M atau W dengan bentuk yang disebut pengguna, selain `BM` di atas. Kekeliruan H/M/W belum bisa dinilai.

### 3a. Rekaman 30 menit per kamera

Pipeline dengan kode final dijalankan atas rekaman 30 menit dari tiap kamera (13 November 2026, sekitar pukul 15.57 sampai 16.27):

| Kamera | Event | Plat pada crop menurut mata | Confidence | Bacaan |
|---|---|---|---|---|
| cam-3 | BK 8985 HF | BK 8985 HF | 1,00 | 2 |
| cam-3 | BK 8104 GP | BK 8104 GP | 1,00 | 2 |
| cam-3 | BK 8354 CP | BK 8354 CP | 0,96 | 3 |
| cam-2 | BK 8460 XD | BK 8460 XD | 0,83 | 5 |
| cam-2 | BK 8701 ZE | dua huruf terakhir tidak bisa dipastikan dari crop | 0,90 | 3 |
| cam-1 | tidak ada event | - | - | - |
| cam-4 | tidak ada event | - | - | - |

Empat dari lima event cocok dengan crop-nya; satu tidak bisa saya pastikan karena crop-nya buram. Di cam-2 confidence lebih rendah dan bacaan lebih banyak, karena platnya lebih kecil dan beberapa bacaan per frame tidak sepakat.

Rekaman cam-1 dan cam-4 tidak saya tonton, jadi tidak diketahui apakah memang tidak ada plat yang menghadap kedua kamera itu selama 30 menit, atau ada yang terlewat. Truk yang berhenti menghadap cam-2 dan cam-3 memperlihatkan bagian belakangnya ke cam-1 dan cam-4.

Latency dari proses ini tidak dipakai, karena laptop sedang dibebani pekerjaan lain.

### 3b. Baca langsung dibanding baca setelah berhenti (14 November 2026)

Malam pertama di PC ROG menghasilkan dua event berkode wilayah yang tidak ada, `BF 8418 FD` dan `JK 9027 AG`. Crop bacaan yang tersimpan menunjukkan sebabnya: bacaan diambil saat truk masih bergerak. Truk pertama dibaca `BC8418FD`, `BC841PD`, `BF8418FD` dalam dua detik; truk lain dibaca `BE6250GP` dan `BC8250GP` sebelum bacaan ketiganya benar, `BK8250GP`.

Dua perubahan dibuat di versi 0.4.0: plat dibaca setelah kendaraan berhenti, dan kode wilayah di luar daftar diganti kode terdekat. Untuk yang pertama, rekaman 30 menit cam-2 dan cam-3 (bagian 3a) dijalankan lewat dua pipeline sekaligus dengan deteksi yang sama: cara lama (baca langsung) dan cara baru.

| Truk | Cara lama: bacaan per frame | Event lama | Cara baru: bacaan per frame | Event baru | Selisih waktu event |
|---|---|---|---|---|---|
| cam-3 BK 8985 HF (sudah berhenti) | benar, benar | 1,00 | benar, benar | 1,00 | +0,8 detik |
| cam-3 BK 8104 GP (lewat 5,5 detik, tidak berhenti) | benar, benar | 1,00 | benar (satu bacaan saat jejak ditutup) | 1,00 | +5,1 detik |
| cam-3 BK 8354 CP | `BK8354EP`, benar, benar | 0,95 | benar, benar | 1,00 | +2,3 detik |
| cam-2 BK 8701 ZE | benar, `BK6707ZE`, benar | 0,90 | benar, benar | 0,94 | +0,4 detik |
| cam-2 BK 8460 XD | benar, `BK450XD`, benar, benar, benar | 0,83 | benar, benar | 1,00 | −1,3 detik |

"Benar" berarti sama dengan plat event, yang pada bagian 3a dicocokkan dengan crop menurut mata (dua huruf terakhir BK 8701 ZE tidak bisa dipastikan).

Yang terlihat dari lima truk ini:

- Kelima event sama platnya di kedua cara.
- Bacaan per frame yang salah: 3 dari 15 di cara lama, 0 dari 9 di cara baru. Satu jejak lain di cam-2 tidak menghasilkan teks di kedua cara.
- Truk berhenti 0,7 sampai 7,2 detik setelah platnya pertama terlihat. Karena itu batas tunggu dipasang 10 detik; dengan 6 detik truk BK 8460 XD terbaca saat masih merayap.
- Event truk yang lewat tanpa berhenti keluar sekitar 5 detik lebih lambat, karena baru dibaca saat jejaknya ditutup.

Lima truk di siang hari bukan angka akurasi. Malam hari, yang memunculkan masalahnya, belum ada rekamannya; hasil cara baru di malam hari baru terlihat dari event di PC ROG.

Untuk koreksi kode wilayah, crop bacaan kedua event tadi dibaca ulang di laptop. `BF8418FD` menjadi `BK 8418 FD`: di antara kode dua huruf yang ada di daftar, BK paling didukung OCR pada ketiga bacaan (0,06 sampai 0,13, lawan BE 0,04 sampai 0,10). `JK9027AG` menjadi `BK 9027 AG` hanya karena bobot lokasi; tanpa bobot itu OCR memilih DK. Crop truk kedua sangat buram dan angkanya pun tidak pasti, jadi hasil itu tetap tebakan. Bacaan `BE6250GP` tidak berubah, karena BE kode sah.

### 3c. Sehari penuh dengan versi 0.4.0 (14 November 2026)

Versi 0.4.0 berjalan di PC ROG dari 02.37 dan mencatat 211 event sampai tengah malam. Versi itu masih membaca kendaraan yang tidak berhenti: setelah 10 detik bergerak, atau satu kali dari frame terbaik saat jejaknya ditutup.

| Kamera | Event | 1 bacaan | 2 bacaan | 3 sampai 4 | 5 atau lebih |
|---|---|---|---|---|---|
| cam-1 | 73 | 28 | 29 | 3 | 13 |
| cam-2 | 83 | 22 | 50 | 4 | 7 |
| cam-3 | 39 | 21 | 14 | 1 | 3 |
| cam-4 | 16 | 3 | 11 | 2 | 0 |

Foto dari tiga kelompok dilihat satu per satu (9 sampai 12 contoh per kelompok):

- Event 2 bacaan: truk berhenti di timbangan, plat tajam selebar 110 sampai 150 piksel, termasuk yang malam.
- Event 5 bacaan atau lebih: truk juga berhenti, tetapi platnya kecil (104 sampai 122 piksel) dan bacaan per frame tidak sepakat; confidence 0,42 sampai 0,92.
- Event 1 bacaan: truk yang lewat dekat kamera tanpa berhenti, plat 110 sampai 390 piksel. Siang hari banyak yang terbaca benar; malam hari buram (`2C147411`, `L 1114 DC`).

Dari 74 event 1 bacaan, 7 adalah truk yang sudah punya event di kamera yang sama dalam 10 menit sebelumnya, 24 punya event di kamera pasangannya, dan 43 adalah satu-satunya catatan truk itu. Tujuh event berisi teks tanpa angka (`SATPAN`, `SATPANV`): tulisan di rompi petugas yang terdeteksi sebagai plat.

Akibatnya versi 0.4.1 hanya membaca plat yang diam dan menolak teks tanpa angka. Yang hilang dengan keputusan itu: catatan untuk truk yang tidak pernah berhenti di depan kamera mana pun, termasuk yang siang hari sebenarnya terbaca benar saat lewat.

Plat yang sama muncul lebih dari sekali pada 48 plat. Sebagian besar adalah lintasan terpisah yang berjarak menit sampai jam (`BK 8486 WR` 20 kali sepanjang hari di tiga kamera). 26 event mengulang truk yang baru dilaporkan kamera pasangannya di timbangan yang sama, dan 14 mengulang truk di kamera yang sama.

### 3d. Plat truk yang diam tidak terdeteksi, dan area fokus (17 November 2026)

Kejadiannya di cam-4 pukul 16.45: truk `BK 8013 XJ` berdiri di timbangan, halaman pemantauan menulis "Tidak ada kendaraan", dan `/plate/4` kosong. Event-nya sendiri sudah keluar pukul 16.45.07 (dua bacaan, 99,6%) ketika plat masih selebar 127 piksel. Setelah truk berhenti di posisi akhirnya, plat tinggal selebar 106 piksel dan detektor tidak melihatnya lagi.

Detektor menerima area baca (`roi`, 1792×936 piksel) yang diperkecil ke 640 piksel, jadi plat itu tinggal sekitar 38 piksel. Frame asli yang sama diuji dengan potongan persegi di sekitar titik berhenti, ambang detektor 0,05:

| Potongan | Skala ke detektor | Skor plat |
|---|---|---|
| `roi` yang dipakai, 1792×936 | 0,36 | tidak ada kotak |
| 1152×1152 | 0,56 | tidak ada kotak |
| 1024×1024 | 0,62 | 0,30 |
| 896×896 | 0,71 | tidak ada kotak |
| 800×800 | 0,80 | 0,31 |
| 720×720 | 0,89 | 0,56 |
| 640×640 | 1,00 | 0,73 |

Tracker baru memulai jejak dari skor 0,5, jadi potongannya harus sekitar 640 piksel.

Potongan saja bukan pengganti area lebar. Pada 100 frame event kendaraan diam yang tersimpan (25 per kamera), plat yang terdeteksi dengan skor minimal 0,5:

| Cara | Plat terdeteksi |
|---|---|
| Area lebar saja | 93 |
| Potongan 640 saja | 89 |
| Potongan 800 saja | 91 |
| Potongan 960 saja | 87 |
| Area lebar atau potongan 640 | 97 |

Frame itu tersimpan karena area lebar pernah mendeteksi platnya, jadi angka 93 lebih bagus dari kenyataan; kasus seperti `BK 8013 XJ` tidak ada di dalamnya. Detektor ini tidak konsisten terhadap ukuran plat: `BK 8834 BF` di cam-3 mendapat 0,74 pada area lebar dan tidak terdeteksi pada potongan 640.

Karena itu tiap kamera mendapat `focus`: potongan 640×640 piksel di sekitar titik tengah plat kendaraan diam (dari 395 event, potongan itu memuat 386), dicari sebagai tambahan atas `roi`, dan hasil keduanya digabung sebelum tracking. Satu panggilan detektor di GTX 1060 memakan median 15,4 ms saat service berjalan, dan dua panggilan per frame untuk empat kamera pada 8 fps akan memenuhi GPU yang juga dipakai service OCR surat. Maka potongan fokus hanya dicari tiap frame keempat selama tidak ada plat terlihat, dan di setiap frame sejak salah satu pencarian menemukan plat sampai 3 detik setelah deteksi terakhir.

Uji pada 10 menit pertama rekaman cam-3 (`rec-20261113-b`, 4.781 frame diproses, `scripts/compare_focus.py`):

| | Area lebar saja | Dengan fokus |
|---|---|---|
| Frame dengan plat terlihat | 4.262 | 4.519 |
| Frame dengan plat yang layak dikirim `/plate/3` | 4.256 | 4.507 |
| Event | `BK 8985 HF` | `BK 8985 HF`, `BK 8104 GP` |

`BK 8104 GP` berdiri di timbangan dan terbaca 99,9% pada detik ke-569; dengan area lebar saja truk itu belum terbaca sampai menit ke-10. Rekaman 30 menit keempat kamera tidak selesai diuji: sembilan proses dijalankan bersamaan dan berebut CPU.

Dipasang di PC ROG pukul 19.49. Sesudahnya keempat kamera diproses 8 fps, total per frame median 23 sampai 26 ms dan p95 46 sampai 53 ms saat timbangan kosong, GPU 31 sampai 62%. Yang belum diukur: beban dan hasil saat beberapa truk berdiri bersamaan di siang hari.

## 4. Latency

Angka di bawah diukur di laptop pengembangan (i7-14700HX, RTX 5050), kecuali bagian "PC target" yang diukur di PC ROG setelah pemasangan.

### Satu kamera dari rekaman, CPU

Rekaman cam-2, 282 frame diproses, ONNX Runtime 1.31.1 CPU, pukul 16.00:

| Tahap | p50 ms | p95 ms |
|---|---|---|
| decode satu frame (H.265, 2560×1440) | 5,3 | 7,7 |
| konversi ke BGR | 1,4 | 1,8 |
| pra-proses (potong ROI, letterbox ke 640) | 3,6 | 4,5 |
| deteksi | 15,1 | 18,3 |
| tracking | 0,4 | 0,7 |
| OCR, satu crop (hanya 2 kali dipanggil) | 27 | 39 |
| total per frame yang diproses | 21,0 | 27,8 |

### Empat kamera asli bersamaan, CPU

90 detik, 8 fps per kamera, pukul 16.08:

| Kamera | fps diproses | total p50 ms | total p95 ms | reconnect |
|---|---|---|---|---|
| cam-1 | 8,0 | 32,5 | 97,6 | 0 |
| cam-2 | 8,5 | 32,2 | 99,4 | 0 |
| cam-3 | 8,0 | 29,0 | 90,4 | 0 |
| cam-4 | 8,0 | 28,9 | 115,0 | 0 |

`total` mencakup waktu menunggu giliran model. Median jauh di bawah 100 ms; p95 berada di sekitar 100 ms karena empat kamera bergiliran memakai satu model di CPU. Pengukuran ini termasuk detik-detik awal saat model dan koneksi baru dibuka, dan diambil sebelum perbaikan penjadwalan frame.

### Empat kamera asli bersamaan di PC target (GTX 1060)

PC ROG (i7-8700, GTX 1060 3 GB, driver 581.95), `onnxruntime-gpu` 1.26.0 dengan CUDA 12, 90 detik setelah 10 detik pemanasan, 13 November 2026 pukul 20.15. PC itu sedang menjalankan aplikasi lain (browser, SQL Server, Node).

| Kamera | fps diproses | deteksi p50 ms | total p50 ms | total p95 ms | total maks ms |
|---|---|---|---|---|---|
| cam-1 | 7,5 | 25,6 | 51,2 | 64,0 | 84,9 |
| cam-2 | 8,0 | 15,5 | 27,4 | 58,0 | 75,0 |
| cam-3 | 8,0 | 26,6 | 33,4 | 44,7 | 77,7 |
| cam-4 | 7,5 | 15,8 | 43,2 | 63,9 | 88,4 |

Tidak ada reconnect. Umur frame saat diambil: median 21 sampai 28 ms. Dari frame keluar decoder sampai hasilnya siap: median 51 sampai 74 ms, p95 86 sampai 116 ms.

Kedua model memakai `CUDAExecutionProvider`, baik saat dijalankan dari sesi SSH maupun dari task terjadwal yang berjalan tanpa login. Selama pengukuran keluar satu event (`BK 8621 XC` di cam-3). Saat service berjalan, VRAM terpakai 1.238 MiB dari 3.072 MiB (928 MiB di antaranya sudah dipakai aplikasi lain sebelum service menyala), dan beban CPU seluruh PC sekitar 54%.

RTX 5050, `onnxruntime-gpu` 1.26.0 dengan CUDA 12, GPU sedang tidak dipakai, pukul 16.29:

| Tahap | p50 ms | p95 ms |
|---|---|---|
| inferensi detektor | 7,7 sampai 8,1 | 11 sampai 13 |
| OCR satu crop, termasuk resize | 6,0 sampai 6,4 | 9 sampai 10 |

### Dua pengukuran yang tidak dipakai

Benchmark empat kamera diulang pukul 17.02 (CPU) dan 17.05 (GPU) setelah perbaikan kode. Saat itu laptop sedang dipakai menjalankan game yang memakai lebih dari dua core CPU dan GPU pada 100%. Deteksi menjadi 36 ms di CPU dan 140 ms di GPU, dan model tidak sanggup melayani empat kamera pada 8 fps. Angka itu tidak mewakili kemampuan sistem.

Yang tetap bisa dipetik dari dua pengukuran itu: ketika model kewalahan, tiap kamera melewati frame (yang diproses turun ke 6 fps di CPU dan 2 fps di GPU), tidak ada reconnect, dan frame yang diambil tetap frame terbaru (umur median sekitar 25 sampai 30 ms). Delay tidak menumpuk.

### Terhadap target

| Target | Keadaan |
|---|---|
| Latency paling lama 100 ms per frame per kamera | Di PC target dengan empat kamera: median 27 sampai 51 ms, p95 45 sampai 64 ms, maksimum 88 ms dalam 90 detik pengukuran. Terpenuhi pada pengukuran ini |
| Empat kamera tanpa delay menumpuk | Rancangannya membuang frame lama, dan itu terlihat bekerja saat sistem dibebani. `lag_s` di `/cameras` melaporkan ketertinggalan pembaca stream |
| Akurasi plat end-to-end minimal 95% | Belum diukur. Perlu test set berlabel |

Untuk mengukur di PC ROG, setelah service terpasang: `uv run --frozen --extra gpu alpr-id benchmark --live 120`, saat PC tidak dipakai untuk hal lain.

## 5. Temuan tinjauan kode dan perbaikannya

Setelah runtime berjalan, satu agen meninjau kodenya tanpa mengubah apa pun. Temuan yang diperbaiki:

| Temuan | Perbaikan |
|---|---|
| Jatah 5 bacaan OCR habis dalam 1,6 detik pertama. Truk yang terbaca buruk saat mendekat tidak dibaca lagi setelah berhenti | Dua bacaan pertama tetap cepat. Sisanya disimpan untuk crop yang 25% lebih baik dari yang sudah dibaca, atau dipakai tiap 3 detik bila tidak ada yang lebih baik. Kendaraan yang tidak menghasilkan bacaan layak dibaca ulang setelah 8 detik |
| Bacaan lemah (confidence 0,2) tetap muncul di `/plate/{n}` | Plat hanya ditampilkan bila confidence-nya minimal 0,4 |
| Kegagalan menyimpan event (disk penuh, database terkunci) membuat tiap frame mengulang event dengan ID baru | Event dianggap terbit walau penyimpanan gagal; kegagalannya dicatat di log |
| Jendela 120 detik dihitung dari saat crop terbaik, bukan dari saat kendaraan terakhir terlihat | Stempel waktu diperbarui selama kendaraan masih terlihat |
| Event bisa terbit cepat walau satu karakter masih seimbang antara dua pilihan | Tiap karakter harus mencapai 0,6 sebelum event terbit lebih awal |
| `BK12345` (BK 1234 S, S terbaca 5) dipotong menjadi `BK1` karena dianggap masa berlaku | Empat digit terakhir harus berupa bulan dan tahun, dan bacaan yang cukup dibetulkan dengan satu penukaran tidak dipotong |
| `lost_track_buffer: 30` berarti 1 detik di pustaka tracker, bukan 30 frame | Diganti `lost_track_s: 4` |
| Tanpa aturan firewall, task terjadwal hanya bisa diakses dari PC itu sendiri | `install-autostart.ps1` membuat aturan untuk port API |
| Klik di jendela konsol membekukan semua kamera (mode QuickEdit) | QuickEdit dimatikan saat service mulai. Belum diuji dengan mengklik jendelanya |
| Login yang ditolak diulang tiap 10 detik | Jeda 10 menit setelah login ditolak |
| Lainnya | Track ditutup walau kamera tidak mengirim frame; event yang tertunda dikeluarkan saat service berhenti; `/cameras` tidak lagi bisa gagal karena membaca data yang sedang ditulis; Ctrl+C tidak menggantung saat ada penonton video; pembersihan media berjalan harian; `since` menerima `+07:00` yang tidak di-encode |

## 6. Yang berbeda dari permintaan

- **TensorRT tidak dipakai.** GTX 1060 tidak didukung TensorRT 10 ke atas. Provider GPU-nya CUDA lewat ONNX Runtime 1.26.0.
- **YOLO26 belum dipakai** di Fase 2, dengan alasan di bagian 1.
- **Bacaan OCR bisa melebihi N per kendaraan** dalam satu keadaan: bila 5 bacaan pertama tidak menghasilkan apa pun yang layak dan kendaraannya masih ada, 5 bacaan baru diberikan setelah 8 detik. Ini bisa dimatikan dengan `ocr.retry_after_s: 0`.
- **Event memuat field tambahan** di luar contoh: `snapshot_url`, `region_known`, `alternatives`, `reads`, `raw`.
- **Docker** disediakan tetapi bukan jalur utama, dan belum diuji.

## 7. Yang belum diuji

- Service menyala sendiri setelah PC ROG di-restart.
- Akses ke port 5000 dari PC timbangan di LAN. Yang sudah diuji baru dari laptop lewat Tailscale.
- Dockerfile dan docker-compose.
- Service berjalan berhari-hari. Uji terlama sekitar 4 menit.
- Kondisi malam, hujan, dan plat kotor.
- Hardware decode (`capture.hwaccel`).
- Perilaku aplikasi desktop terhadap rute `/plate/{n}` yang baru.

## 8. Memasang di PC ROG

1. Salin folder `alpr-id` tanpa `.venv`, `.venv-gpu`, dan `data\samples`. Sertakan `models\`.
2. Pasang uv. Perbarui driver NVIDIA ke 582.78.
3. Di `.env`, set `ALPR_EXTRA=gpu`.
4. `uv sync --frozen --extra gpu`.
5. Jalankan `start-alpr.bat`, lalu buka `http://127.0.0.1:5000/health`. `providers` harus memuat `CUDAExecutionProvider`. Kalau hanya `CPUExecutionProvider`, GPU tidak terpakai dan log memuat `provider_unavailable`.
6. `uv run --frozen --extra gpu alpr-id benchmark --live 120`.
7. Setelah hasilnya baik, jalankan `install-autostart.ps1` dari PowerShell Administrator.
8. Di `.env` Laravel, set `CCTV_ANPR_URL=http://192.168.1.37:5000`. Kompilasi ulang desktop dengan alamat yang sama.

Di PC ber-GPU, semua contoh perintah di README yang memakai `--extra cpu` harus diganti `--extra gpu`. Menjalankan perintah dengan `--extra cpu` di sana akan mengganti ONNX Runtime GPU di `.venv` dengan versi CPU.

## 9. Bahan untuk Fase 3

- Service menyimpan setiap crop yang di-OCR di `data\media\<tanggal>\<kamera>\reads\`, dengan hasil baca di nama filenya. Membiarkan service berjalan beberapa hari menghasilkan bahan dataset tanpa kerja tambahan.
- Rekaman 10 dan 30 menit dari keempat kamera ada di `data\samples\`.
- Label Studio 1.23.2 sudah terpasang di `D:\labelstudio`.
