# alpr-id

Pembaca plat nomor kendaraan Indonesia dari stream CCTV (RTSP) di timbangan, dengan REST API yang mengembalikan JSON. Menggantikan layanan lama di `ai_cctv`.

## Status

| Fase | Isi | Status |
|---|---|---|
| 1 | Audit folder lama, rencana, struktur proyek | selesai |
| 2 | Runtime dengan model pretrained (baseline), API, benchmark | selesai, terpasang di PC ROG |
| 3 | Pipeline dataset dan training | selesai (456 crop pertama sudah dilabeli) |
| 4 | Fine-tune, perbandingan dengan baseline, optimasi kecepatan | selesai 15 November 2026; model OCR baru terpasang di PC ROG, detektor tetap |

Dokumen: [audit dan rencana](docs/fase-1-audit-dan-rencana.md), [hasil Fase 2 dan benchmark](docs/fase-2-baseline-dan-benchmark.md), [dataset dan training](docs/fase-3-dataset-dan-training.md), [fine-tune dan perbandingan](docs/fase-4-finetune-dan-perbandingan.md), [keputusan arsitektur](CLAUDE.md).

Service memakai detektor YOLOv8n lama dan, sejak 15 November 2026, model OCR hasil fine-tune `models/ocr/finetuned/timbangan-20261115.onnx`. Pada 280 crop berlabel manusia dari kamera timbangan, dibaca satu per satu tanpa voting, model `cct-s-v2-global` dari hub membaca 228 dengan benar. Dengan resep fine-tune yang menghasilkan model baru itu, angkanya 244 pada evaluasi silang, yaitu tiap crop dibaca model yang belum pernah melihat platnya. Rinciannya di dokumen Fase 4.

## Menjalankan

Kebutuhan: Windows 10/11, [uv](https://docs.astral.sh/uv/) 0.13 atau lebih baru (Python 3.12 dipasang otomatis oleh uv), dan akses LAN ke kamera.

1. Siapkan lingkungan. Pakai `gpu` di PC ber-GPU NVIDIA, `cpu` selain itu:
   ```powershell
   cd "D:\Asisten IT\Project\Timbangan\alpr-id"
   uv sync --frozen --extra cpu
   ```
   Di PC ber-GPU, ganti `--extra cpu` dengan `--extra gpu` di semua perintah dalam dokumen ini. Perintah ber-`--extra cpu` di sana akan mengganti ONNX Runtime GPU dengan versi CPU.
2. Salin `.env.example` menjadi `.env`, isi URL RTSP keempat kamera, set `ALPR_EXTRA` ke `cpu` atau `gpu`, dan isi `ALPR_ADMIN_TOKEN` bila kamera akan dikelola dari browser.
3. Salin `config\app.example.yaml` menjadi `config\app.yaml`.
4. Siapkan model di folder `models\` (lihat bagian Model).
5. Klik dua kali **`start-alpr.bat`**. Jendela tetap terbuka selama service berjalan; tutup jendelanya untuk berhenti.

Service menjawab di `http://<IP-PC>:5000`. Log ada di `data\logs\alpr.log`.

## Halaman browser

| Alamat | Isi |
|---|---|
| `http://<IP-PC>:5000/ui` | Pemantauan: gambar terbaru tiap kamera, status sambungan, plat yang sedang terlihat, empat pembacaan terakhir per kamera, dan daftar pembacaan terbaru semua kamera. Klik gambar untuk memperbesar, klik satu pembacaan untuk melihat crop, frame, dan rinciannya. Di rincian itu, seret kotak di atas frame untuk memperbesar bagian tersebut (bisa berulang, sampai 8 kali); klik dua kali atau tombol "Tampilan penuh" mengembalikannya. |
| `http://<IP-PC>:5000/ui/cameras` | Kelola kamera: daftar kamera beserta sumbernya, dan formulir untuk menambah, mengubah, atau menghapus kamera. |
| `http://<IP-PC>:5000/ui/regions` | Kode wilayah: daftar kode huruf depan plat yang diterima, berapa kali tiap kode terbaca di lokasi ini, dan formulir untuk menambah atau menghapus kode. |

Halaman pemantauan bisa dibuka siapa saja di jaringan, sama seperti rute `/plate/{n}` dan `/video/{n}`. Menambah, mengubah, menguji, dan menghapus kamera memerlukan token `ALPR_ADMIN_TOKEN` dari `.env` (minimal 8 karakter). Token diketik sekali di halaman kamera dan disimpan browser sampai tab ditutup, atau seterusnya bila "Ingat di browser ini" dicentang. Tanpa token di `.env`, tindakan itu ditolak.

Menambah kamera dari halaman:

1. Isi ID (misalnya `cam-5`), keterangan, dan alamat stream. Alamat bisa diisi per bagian (IP, port, nama pengguna, sandi, jalur) atau ditempel utuh. Hanya `rtsp://` dan `rtsps://` yang diterima.
2. Tekan **Uji sambungan**. Bila berhasil, codec, resolusi, dan satu gambar kamera muncul. Bila kamera menolak login tiga kali dalam sepuluh menit, pengujian ke kamera itu ditahan dulu supaya akunnya tidak dikunci kamera.
3. Tarik kotak di gambar untuk menentukan area baca, atau isi angkanya dalam persen.
4. Isi nomor rute lama bila kamera itu perlu dijawab di `/plate/{n}` dan `/video/{n}`, lalu **Simpan kamera**. Kamera langsung dibaca tanpa restart.

### Kapan plat dibaca, dan kode wilayah yang tidak ada

Plat dibaca saat kendaraannya berhenti, supaya bacaan dan foto yang disimpan bersama event tidak buram. "Berhenti" berarti kotak plat bergeser kurang dari 12% lebarnya selama 0,75 detik (`ocr.still_window_s`, `ocr.still_tolerance`). Selama plat terlihat tetapi belum dibaca, halaman pemantauan menulis "Plat terlihat, belum dibaca".

Kendaraan yang lewat tanpa pernah berhenti dibaca satu kali dari frame terbaiknya, sebagai cadangan. Bacaan saat berhenti selalu didahulukan:

- bacaan saat lewat ditahan dulu `ocr.passing_quiet_s` (20 detik) setelah jejaknya ditutup, dan dibuang bila dalam waktu itu ada kendaraan yang berhenti di kamera yang sama, karena itu truk yang sama yang baru masuk;
- jejak baru yang muncul dalam 20 detik setelah kendaraan yang berhenti terakhir terlihat tidak dibaca, karena itu truk yang sama yang sedang pergi;
- bacaan saat lewat hanya menjadi event bila berpola plat umum dan keyakinannya minimal `voting.min_passing_confidence` (0,8).

Event menyimpan `stopped`: `true` bila dibaca saat berhenti, `false` bila dibaca saat lewat (di halaman bertanda "Dibaca saat kendaraan lewat, tidak berhenti"), `null` untuk event sebelum versi 0.4.2.

Pilihannya ada di `ocr.read_when`:

| Nilai | Arti |
|---|---|
| `stopped_or_passing` (bawaan) | saat berhenti, ditambah satu bacaan cadangan untuk kendaraan yang lewat tanpa berhenti |
| `stopped` | hanya saat plat diam; kendaraan yang tidak berhenti di depan sebuah kamera tidak dilaporkan kamera itu |
| `always` | langsung saat plat terlihat, seperti versi 0.3 |

Teks tanpa angka sama sekali, misalnya tulisan "SATPAM" di rompi petugas yang ikut terdeteksi, tidak pernah menjadi event.

Kode wilayah yang terbaca tetapi tidak ada di daftar, misalnya `BF 8418 FD`, diganti dengan kode di daftar yang paling didukung bacaan OCR pada huruf itu (di contoh ini `BK`). Yang dibandingkan hanya kode yang sama panjangnya. Kode yang sering terbaca di lokasi mendapat bobot lebih, paling banyak tiga kali (`postprocess.region_local_boost`). Event semacam itu bertanda `region_corrected: true`, `raw` tetap berisi bacaan aslinya, dan `confidence` dihitung ulang dengan dukungan OCR untuk huruf pengganti. Bacaan yang meleset ke kode lain yang juga ada di daftar (misalnya `BK` terbaca `BE`) tidak bisa dikenali dengan cara ini. Fitur ini dimatikan dengan `postprocess.fix_unknown_region: false`.

Daftar kode disimpan di `data\alpr.sqlite3` dan diubah lewat `/ui/regions`; perubahan langsung berlaku tanpa restart. Database baru berisi 61 kode bawaan. Kode bawaan yang dihapus tidak kembali sendiri saat restart; tombol "Pulihkan kode bawaan" mengembalikannya.

Kamera dari halaman kamera disimpan di `data\alpr.sqlite3`, termasuk URL dengan sandinya, jadi berkas itu perlu dijaga seperti `.env`. Empat kamera di `config\app.yaml` tetap diatur lewat berkas itu dan tampil di halaman sebagai baca-saja. Semua kamera memakai detektor dan OCR yang sama; kolom "Diproses" di kedua halaman menunjukkan apakah tiap kamera masih mencapai fps yang diminta setelah kamera bertambah.

### Jalan otomatis saat PC menyala

Buka PowerShell sebagai Administrator di folder proyek:

```powershell
powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1                    # daftar dan jalankan
powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1 -Action Status     # lihat keadaan
powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1 -Action Stop       # hentikan sementara
powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1 -Action Uninstall  # hapus
```

Task bernama `ALPR-ID` berjalan 30 detik setelah PC menyala, tanpa jendela dan tanpa perlu login, dan dijalankan ulang oleh Task Scheduler bila berhenti. Jangan menjalankan `start-alpr.bat` selagi task aktif, karena keduanya memakai port yang sama.

## Model

Folder `models\` tidak masuk git. Isinya:

| File | Asal |
|---|---|
| `models\detector\plat.onnx` | `best.pt` layanan lama (YOLOv8n, kelas `plat`), diekspor ke ONNX |
| `models\ocr\cct_s_v2_global.onnx` dan `cct_s_v2_global_plate_config.yaml` | fast-plate-ocr `cct-s-v2-global-model` |
| `models\detector\yolo-v9-t-640-license-plates-end2end.onnx` | open-image-models, detektor pembanding |

Untuk PC lain, cara termudah adalah menyalin folder `models\` dari laptop pengembangan. Untuk membuatnya dari awal:

```powershell
uv run --frozen --extra cpu python scripts/fetch_models.py      # OCR dan detektor pembanding
Copy-Item "..\ai_cctv\project0\best.pt" models\detector\src\    # salinan, folder lama tidak diubah
uv run --project training python training/detector/export_onnx.py --weights models/detector/src/best.pt --nms true --out models/detector/plat.onnx
```

## API

| Rute | Isi |
|---|---|
| `GET /health` | status service, jumlah kamera tersambung, provider ONNX Runtime |
| `GET /cameras` | status tiap stream: tersambung, fps, fps yang diproses, latency, area baca, asal kamera (`config` atau `ui`), plat yang sedang terlihat (`current`), dan jumlah plat yang sedang diikuti, terbaca atau belum (`plates_in_view`) |
| `GET /regions` | daftar kode wilayah yang diterima: `code`, `name`, `builtin`, dan `seen` (berapa kali terbaca di lokasi ini) |
| `GET /cameras/{id}/current` | plat terbaik yang sedang terlihat di kamera itu |
| `GET /cameras/{id}/snapshot.jpg?width=` | satu JPEG frame terakhir dengan kotak plat; `width` 160 sampai 1920, dibulatkan ke kelipatan 160 |
| `GET /cameras/{id}/video` | MJPEG kamera itu, untuk kamera tanpa nomor rute lama |
| `GET /events?camera=&since=&limit=` | riwayat pembacaan, terbaru dulu. `since` berupa ISO 8601 atau detik epoch |
| `GET /events/stream` | Server-Sent Events, satu pesan `plate` per kendaraan |
| `POST /recognize` | unggah gambar (field `file`), hasil deteksi dan bacaan tanpa tracking |
| `GET /media/...` | crop plat dan snapshot frame yang dirujuk `crop_url` dan `snapshot_url` |
| `GET /docs` | dokumentasi interaktif bawaan FastAPI |

Contoh event:

```json
{"event_id": "e0f3d984fa99475ebc05ea6e3a6dd5ef", "camera_id": "cam-3", "plate": "BK8985HF",
 "plate_formatted": "BK 8985 HF", "confidence": 0.9997, "format_valid": true,
 "timestamp": "2026-11-13T16:02:38+07:00", "track_id": 0, "bbox": [1262, 662, 1374, 700],
 "crop_url": "/media/2026-11-13/cam-3/160238_e0f3d984_crop.jpg",
 "snapshot_url": "/media/2026-11-13/cam-3/160238_e0f3d984_frame.jpg",
 "region_known": true, "alternatives": [], "reads": 2, "raw": "BK8985HF"}
```

- `format_valid` bernilai `false` untuk plat di luar pola 1-2 huruf, 1-4 angka, 0-3 huruf (TNI, Polri, CD). Platnya tetap dikembalikan apa adanya.
- `region_known` menandai apakah kode wilayahnya ada di daftar kode (`/ui/regions`).
- `region_corrected` bernilai `true` bila kode yang terbaca tidak ada di daftar dan sudah diganti dengan kode terdekat; `raw` memuat bacaan sebelum diganti.
- `alternatives` berisi bacaan lain bila huruf mirip (H, M, W) nyaris sama kuat.
- Satu kendaraan menghasilkan satu event per kamera. Plat yang sama di kamera yang sama dalam 120 detik dianggap kendaraan yang sama.

### Rute kelola kamera

Semua memerlukan header `Authorization: Bearer <ALPR_ADMIN_TOKEN>`, kecuali `/admin/check`. Tanpa header atau dengan token salah jawabannya 401; bila token belum diatur di server, 403.

| Rute | Isi |
|---|---|
| `GET /admin/check` | `{"configured": true, "authorized": false}`: apakah server punya token, dan apakah token di header benar |
| `GET /admin/cameras` | asal tiap kamera dan URL-nya dengan sandi disamarkan |
| `POST /cameras` | tambah kamera. Body: `id`, `url`, dan opsional `description`, `legacy_id`, `roi` (`[x1, y1, x2, y2]` pecahan 0 sampai 1), `process_fps` (1 sampai 15). Jawaban 201 |
| `PUT /cameras/{id}` | ganti pengaturan kamera yang ditambahkan lewat API. `url` boleh dikosongkan untuk memakai yang tersimpan; isian lain yang tidak dikirim kembali ke nilai bawaan |
| `DELETE /cameras/{id}` | hentikan dan hapus kamera yang ditambahkan lewat API. Riwayat event dan gambarnya tetap. Jawaban 204 |
| `POST /cameras/test` | buka `url` (atau URL tersimpan milik `camera_id`) sekali. Jawaban berisi `ok`, `code`, `codec`, `width`, `height`, `fps`, dan `snapshot` (data URL JPEG) |
| `POST /regions` | tambah kode wilayah. Body: `code` (satu atau dua huruf) dan opsional `name`. Jawaban 201 |
| `DELETE /regions/{code}` | hapus kode dari daftar. Jawaban 204 |
| `POST /regions/restore` | masukkan lagi kode bawaan yang pernah dihapus. Jawaban `{"added": n}` |

Kesalahan rute ini berbentuk `{"detail": {"code": "...", "message": "..."}}`. Nilai `code`: `duplicate_id`, `duplicate_legacy_id`, `camera_limit`, `config_camera`, `duplicate_code`, `special_code` (409), `unknown_camera`, `unknown_code` (404), `bad_token` (401), `admin_disabled` (403), `login_locked` (429, dengan header `Retry-After`). Hasil uji sambungan memakai `code`: `ok`, `login_rejected`, `not_found`, `unreachable`, `timeout`, `no_video`, `error`. Isian yang tidak valid dijawab 422 dalam bentuk bawaan FastAPI.

### Rute lama untuk aplikasi desktop dan web registrasi

| Rute | Isi |
|---|---|
| `GET /plate/{n}` | `{"plate": "BK 8985 HF", "raw": "BK8985HF", "fps": 20.0, "status": "online"}`. `plate` kosong bila tidak ada plat. `status`: `online`, `lagging`, atau `offline` |
| `GET /video/{n}` | MJPEG dengan kotak dan teks plat, untuk `<img>` |

`n` adalah nomor kamera lama: 1 Outgoing masuk, 2 Outgoing keluar, 3 Incoming masuk, 4 Incoming keluar. Format `plate` (berspasi atau rapat) diatur lewat `api.legacy_plate_format` di config.

## Alat bantu

```powershell
uv run --frozen --extra cpu --env-file .env python scripts/probe_rtsp.py           # codec, resolusi, fps tiap kamera
uv run --frozen --extra cpu --env-file .env python scripts/record_rtsp.py --seconds 600   # rekam tanpa encode ulang
uv run --frozen --extra cpu alpr-id recognize gambar.jpg                           # baca plat dari satu gambar
uv run --frozen --extra cpu alpr-id benchmark --live 120                           # latency empat kamera asli
uv run --frozen --extra cpu alpr-id benchmark data\samples\rec-xxx --save-crops    # latency dan event dari rekaman
uv run --frozen --extra cpu pytest                                                 # test
```

Hasil benchmark tersimpan di `data\benchmarks\`.

## Docker

`Dockerfile` dan `docker-compose.yml` disediakan untuk host Linux atau WSL2. Keduanya belum diuji, karena Docker tidak terpasang di laptop pengembangan. Di PC Windows, jalur utamanya `start-alpr.bat` dan Task Scheduler.

## Struktur

```
alpr-id/
├── src/alpr_id/
│   ├── capture/       reader RTSP per kamera, frame terbaru, reconnect, ROI
│   ├── detect/        detektor plat lewat ONNX Runtime
│   ├── track/         ByteTrack, satu tracker per kamera
│   ├── ocr/           fast-plate-ocr, skor mutu crop
│   ├── postprocess/   format plat, koreksi karakter, voting
│   ├── pipeline/      alur per kamera, engine, event
│   ├── storage/       SQLite untuk event dan kamera tambahan, file crop dan snapshot
│   ├── api/           FastAPI dan halaman browser (static/)
│   └── bench/         benchmark dari rekaman dan dari kamera
├── training/          lingkungan terpisah untuk training dan export model
├── scripts/           probe, perekam, pengunduh model
├── tests/             pytest
├── config/            app.example.yaml
├── start-alpr.bat, install-autostart.ps1
├── Dockerfile, docker-compose.yml
├── data/              database, media, log, rekaman (tidak masuk git)
└── models/            bobot ONNX (tidak masuk git)
```
