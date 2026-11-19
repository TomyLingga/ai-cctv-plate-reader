# Rekaman 18–19 November 2026: dari lebar berapa plat ditemukan dan terbaca

Pengguna meminta PC ROG merekam video untuk sementara, supaya bisa dilihat dari jarak berapa plat terbaca dan tidak. Dokumen ini memuat cara mengukurnya, hasilnya, dan saran yang keluar dari situ.

## Bahan

- Rekaman keempat kamera tanpa encode ulang, 18 November pukul 16.01 sampai 19 November pukul 08.32 (16,5 jam, 135 berkas, 50,4 GB), dibuat `scripts/record_long.py`. Detektor yang berjalan di service selama itu adalah `yolov8n-20261118b.onnx`, dipasang pukul 15.59.
- Event service pada jam yang sama: 360 event (`data/rog-20261119/alpr.sqlite3`, salinan lewat backup SQLite).

"Jarak" di sini diukur sebagai lebar plat dalam piksel pada frame kamera: makin jauh truk, makin kecil platnya. Panjang fokus lensa kamera tidak saya ketahui, jadi angka meter hanya perkiraan: plat selebar 105 piksel berarti kira-kira 6 m dari kamera bila lensanya 4 mm, kira-kira 4,5 m bila 2,8 mm (lebar plat 430 mm, sudut pandang dari lembar spesifikasi Hikvision).

## Cara mengukur

`scripts/scan_recording.py` mengambil satu key frame tiap kira-kira 6 detik (38.790 frame) dan memberikannya ke detektor lewat dua pandangan yang dipakai service (ROI dan area fokus), untuk detektor terpasang dan detektor lama `plat.onnx`. Tiap kotak dengan skor 0,35 ke atas dibaca OCR service. `scripts/scan_report.py` lalu:

- merangkai kotak dari frame ke frame menjadi "kunjungan" satu truk;
- menganggap plat sebuah kunjungan diketahui bila sedikitnya tiga bacaannya sama dan berpola plat sah ("bacaan mantap");
- membandingkan tiap kunjungan dengan event service pada kamera dan menit yang sama.

Bacaan mantap bukan kebenaran. Potongan gambar ke-32 kunjungan yang event-nya berbeda satu atau dua huruf dari bacaan mantap saya lihat satu per satu: pada 10 event-nya yang salah, pada 2 bacaan mantapnya yang salah, pada 8 keduanya salah, dan pada 12 gambarnya tidak cukup jelas untuk memutuskan. Angka di bawah karena itu bukan akurasi; akurasi butuh label manusia.

Pemindaian dijalankan di CPU. Percobaan pertama di lingkungan GPU laptop tidak bisa dipakai: pada 4 dari 8 proses OCR membaca sampah untuk seluruh rekaman, padahal sesi OCR-nya di CPU dan hanya detektor yang di GPU. Berkas yang sama terbaca benar saat dipindai ulang. Skrip sekarang membaca tiap bacaan kesepuluh dua kali dengan dua sesi OCR dan berhenti bila keduanya sering berbeda; pada pemindaian CPU 0 dari 1.275 pemeriksaan berbeda.

## Hasil

### Detektor: lama dan baru pada frame yang sama

Plat yang dibaca OCR sebagai plat sah dengan confidence 0,8 ke atas, dan berapa di antaranya mendapat kotak berskor 0,5 ke atas (syarat tracker memulai jejak):

| Lebar plat | Siang: plat | baru | lama | Malam: plat | baru | lama |
|---|---|---|---|---|---|---|
| di bawah 70 px | 18 | 15 | 6 | 21 | 21 | 3 |
| 70–85 px | 90 | 88 | 54 | 90 | 86 | 12 |
| 85–100 px | 458 | 449 | 340 | 838 | 836 | 575 |
| 100–120 px | 1.216 | 1.216 | 1.100 | 1.776 | 1.770 | 1.454 |
| 120–150 px | 133 | 129 | 127 | 144 | 144 | 106 |
| 150 px ke atas | 24 | 24 | 20 | 16 | 16 | 4 |

Seluruhnya 4.824 plat: detektor terpasang 4.794, detektor lama 3.801. Di malam hari 2.873 dan 2.154 dari 2.885. Deteksi bukan lagi yang membatasi, sampai lebar 70 piksel.

### Bacaan satu frame menurut lebar plat

Key frame dari kunjungan yang platnya diketahui, dan berapa bacaan OCR-nya sama dengan bacaan mantap kunjungan itu:

| Lebar plat | Siang: dibaca | sama | Malam: dibaca | sama |
|---|---|---|---|---|
| di bawah 70 px | 73 | 4 (5%) | 104 | 14 (13%) |
| 70–85 px | 110 | 70 (64%) | 189 | 63 (33%) |
| 85–100 px | 593 | 363 (61%) | 933 | 700 (75%) |
| 100–120 px | 1.421 | 1.139 (80%) | 1.882 | 1.647 (88%) |
| 120–150 px | 124 | 116 (94%) | 157 | 125 (80%) |
| 150 px ke atas | 16 | 13 (81%) | 42 | 7 (17%) |

Di siang hari plat yang lebih besar terbaca lebih baik. Di malam hari tidak: plat yang dekat dengan kamera memutih oleh pantulan lampu inframerah, dan lampu depan truk menyilaukan lensa.

### Kunjungan dan event

282 kunjungan punya bacaan mantap. Semuanya menghasilkan event. Pada 245 teks event sama dengan bacaan mantap, pada 32 berbeda satu atau dua huruf, pada 5 berbeda lebih jauh (event ber-confidence 0,42 sampai 0,43).

| Lebar plat saat berdiri | Kunjungan | Event sama | |
|---|---|---|---|
| di bawah 95 px | 39 | 28 | 72% |
| 95–105 px | 91 | 75 | 82% |
| 105–115 px | 113 | 106 | 94% |
| 115 px ke atas | 39 | 36 | 92% |

| Confidence event | Kunjungan | Event sama | |
|---|---|---|---|
| 0,95 ke atas | 182 | 179 | 98% |
| 0,85–0,95 | 51 | 40 | 78% |
| 0,60–0,85 | 30 | 18 | 60% |
| di bawah 0,60 | 17 | 8 | 47% |

Lebar plat saat truk berdiri, per kamera (median; persentil ke-10 sampai ke-90): cam-1 104 (95–110) siang dan 102 (90–112) malam; cam-2 110 (96–122) dan 107 (98–116); cam-3 104 (94–111) dan 98 (84–120); cam-4 115 (79–118) dan 110 (101–132). Hampir semuanya di dalam area fokus (277 dari 282).

Dari 360 event pada jam itu, 78 tidak punya kunjungan berbacaan mantap di belakangnya: plat yang bacaannya berubah-ubah, truk yang hanya lewat, atau kunjungan yang terlalu singkat untuk tiga bacaan.

### Yang terdeteksi tetapi tidak pernah terbaca

Dari 347 benda yang mendapat kotak kuat di empat key frame atau lebih, 65 tidak pernah memberi tiga bacaan yang sama; di cam-1 malam 22 dari 74. Dua puluh di antaranya (cam-1, malam) dilihat: 10 plat belakang tertutup lumpur, 2 plat yang memutih total, 6 bukan plat (slogan di bak truk, logo CIMC, deret lampu, lampu belakang), 2 terlalu kecil atau gelap.

### Mengapa event salah

Dari potongan gambar yang dilihat:

- **Huruf mirip pada plat sekitar 100 piksel.** `BK8566MF` menjadi `HF`, `BK8065VK` menjadi `WK`, `BK8829ML` menjadi `TL`, `BK8359XC` menjadi `TC`, `BK8013LO` menjadi `LD`, `BB8170ME` menjadi `YE`. Pada `BK8566MF` 35 dari 36 key frame terbaca benar, tetapi event-nya diputuskan dari dua bacaan pertama dengan confidence 0,995 dan tidak pernah diperiksa lagi.
- **Silau dan inframerah di malam hari.** Lampu depan truk (cam-2), dan plat reflektif yang memutih saat dekat.
- **Plat terhalang.** Batang bull bar menutup bagian atas plat, lubang rangka, kait.
- **Plat kotor atau rusak.**

## Saran

1. **Baca terus selama truk berdiri.** Sekarang event diputuskan dari 2 sampai 5 bacaan pertama, lalu kendaraan itu tidak dibaca lagi. Bila service tetap membaca sesekali dan `/plate/{n}` mengikuti suara terbanyak, kesalahan seperti `HF` untuk `MF` terkoreksi selagi truk masih di timbangan. Ini perubahan kode; hasilnya bisa diukur dengan memutar rekaman ini lewat pipeline sebelum dan sesudah.
2. **Atur kamera untuk malam.** Kecilkan atau otomatiskan kekuatan inframerah (Smart IR), nyalakan kompensasi cahaya kuat (HLC), batasi waktu eksposur. Ini pengaturan di kamera, bukan di alpr-id.
3. **Buat plat lebih besar saat truk berhenti.** Di bawah 105 piksel event sama pada 103 dari 130 kunjungan, dari 105 piksel ke atas pada 142 dari 152. Caranya garis berhenti 10 sampai 15% lebih dekat, atau lensa yang lebih panjang (6 mm sebagai ganti 4 mm memperbesar plat 1,5 kali). Kerjakan sesudah nomor 2, sebab di malam hari plat yang lebih dekat sekarang justru lebih sering memutih.
4. **Latih OCR pada huruf yang tertukar.** Bahannya sudah ada: crop kecil dari kunjungan yang plat besarnya terbaca jelas, diperiksa orang lewat halaman periksa label, lalu fine-tune seperti di Fase 4.
5. **Pakai confidence.** Event 0,95 ke atas sama pada 179 dari 182 kunjungan, di bawah 0,85 pada 26 dari 47. Aplikasi desktop bisa menandai bacaan ber-confidence rendah untuk dicek orang, tanpa menahannya.
6. **Plat tertutup lumpur tidak bisa diperbaiki perangkat lunak.** Kira-kira separuh dari kotak yang tidak pernah terbaca di malam hari adalah plat seperti itu.
7. **Tambahkan slogan, logo, dan deret lampu sebagai contoh bukan-plat** pada training detektor berikutnya. Mereka jarang menjadi event, tetapi memakan waktu OCR.

## Batasan

- Satu malam dan dua potong siang. Tidak ada hujan dan tidak ada siang penuh.
- Tidak ada label manusia; lihat bagian cara mengukur.
- Satu key frame tiap 6 detik: truk yang sedang mendekat hanya tertangkap dua atau tiga kali, jadi baris di bawah 85 piksel berisi sedikit kunjungan.
- Plat yang tidak diberi kotak oleh kedua detektor tidak terlihat oleh pengukuran ini.
