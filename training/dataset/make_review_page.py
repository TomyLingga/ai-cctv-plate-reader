"""Build one HTML file in which a person checks the labels of plate crops.

The page needs no server and no internet: the crops are inside the file. The reviewer
corrects the text where it is wrong, marks covered or unreadable plates, and downloads
`labels.csv`. Put that file in `<dataset>/review/labels.csv` and run build_ocr_dataset.py
again: human labels replace the automatic ones, and only they count as the test set.

Usage:
    uv run --frozen --extra cpu python training/dataset/make_review_page.py --dataset data/dataset/ocr
    uv run --frozen --extra cpu python training/dataset/make_review_page.py --dataset data/dataset/ocr --splits test,val,none
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
from pathlib import Path

PAGE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Periksa label plat</title>
<style>
  :root { color-scheme: dark; --bg:#0f172a; --card:#1b2336; --line:#334155; --strong:#64748b; --text:#f8fafc; --muted:#94a3b8; --ok:#22c55e; --warn:#fbbf24; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 "Segoe UI",system-ui,sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:12px 16px; display:flex; flex-wrap:wrap; gap:12px 24px; align-items:center; }
  h1 { font-size:18px; margin:0; }
  main { max-width:1100px; margin:0 auto; padding:16px; }
  .how { color:var(--muted); max-width:80ch; }
  .how b { color:var(--text); }
  button, input, select { font:inherit; color:inherit; }
  button { min-height:40px; padding:0 16px; border-radius:6px; border:1px solid var(--strong); background:transparent; cursor:pointer; font-weight:600; }
  button.primary { background:var(--ok); border-color:var(--ok); color:#0b1220; }
  :focus-visible { outline:2px solid #fff; outline-offset:2px; }
  .item { display:grid; grid-template-columns:minmax(0,340px) minmax(0,1fr); gap:12px 20px; align-items:center; padding:12px; margin:8px 0; background:var(--card); border:1px solid var(--line); border-left:4px solid var(--strong); border-radius:6px; }
  .item[data-done="true"] { border-left-color:var(--ok); }
  .item img { width:100%; height:auto; image-rendering:auto; background:#000; border-radius:4px; }
  .fields { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:flex-end; }
  label { display:flex; flex-direction:column; gap:4px; font-size:13px; font-weight:600; }
  input[type=text] { min-height:44px; width:13ch; padding:0 10px; font:700 22px/1 "Cascadia Mono",Consolas,monospace; letter-spacing:.04em; text-transform:uppercase; background:var(--bg); border:1px solid var(--strong); border-radius:6px; }
  select { min-height:44px; padding:0 8px; background:var(--bg); border:1px solid var(--strong); border-radius:6px; }
  .meta { color:var(--muted); font-size:13px; flex-basis:100%; }
  .meta .differs { color:var(--warn); }
  @media (max-width:700px) { .item { grid-template-columns:minmax(0,1fr); } }
</style>
</head>
<body>
<header>
  <h1>Periksa label plat</h1>
  <span id="progress" role="status"></span>
  <label style="flex-direction:row;align-items:center;gap:8px;font-weight:400"><input type="checkbox" id="todo-only"> Tampilkan yang belum dicek saja</label>
  <button class="primary" id="download" type="button">Unduh labels.csv</button>
</header>
<main>
  <p class="how">Cocokkan tulisan di kotak dengan plat pada gambar. <b>Betulkan bila salah</b>, lalu tekan <b>Enter</b> untuk menandai sudah dicek dan pindah ke gambar berikutnya. Tulis tanpa spasi, tanpa bulan dan tahun di baris bawah plat. Bila sebagian plat tertutup atau terpotong, pilih "Tertutup atau terpotong". Bila tidak bisa dibaca sama sekali, pilih "Tidak terbaca". Bila gambarnya bukan plat (lampu, tulisan di rompi, bagian bak), pilih "Bukan plat". Hasil tersimpan di browser ini; setelah selesai, tekan "Unduh labels.csv".</p>
  <div id="list"></div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = 'alpr.review.' + __KEY__;
let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
const list = document.getElementById('list');
const todoOnly = document.getElementById('todo-only');

function store() { try { localStorage.setItem(KEY, JSON.stringify(saved)); } catch (e) {} }
function paintProgress() {
  const done = ITEMS.filter((it) => saved[it.image] && saved[it.image].done).length;
  document.getElementById('progress').textContent = done + ' dari ' + ITEMS.length + ' sudah dicek';
}
function render() {
  list.replaceChildren();
  ITEMS.forEach((it, index) => {
    const state = saved[it.image] || { plate: it.label, status: 'ok', done: false };
    if (todoOnly.checked && state.done) return;
    const row = document.createElement('div');
    row.className = 'item';
    row.dataset.done = String(state.done);
    const img = document.createElement('img');
    img.src = it.data; img.alt = 'Potongan gambar plat ' + (index + 1);
    const fields = document.createElement('div'); fields.className = 'fields';
    const plateLabel = document.createElement('label'); plateLabel.textContent = 'Plat';
    const plate = document.createElement('input'); plate.type = 'text'; plate.value = state.plate; plate.maxLength = 10; plate.autocomplete = 'off'; plate.spellcheck = false;
    plateLabel.append(plate);
    const statusLabel = document.createElement('label'); statusLabel.textContent = 'Keadaan plat';
    const status = document.createElement('select');
    [['ok', 'Terlihat utuh'], ['covered', 'Tertutup atau terpotong'], ['unreadable', 'Tidak terbaca'], ['not_plate', 'Bukan plat']].forEach(([value, text]) => {
      const option = document.createElement('option'); option.value = value; option.textContent = text; status.append(option);
    });
    status.value = state.status; statusLabel.append(status);
    const meta = document.createElement('div'); meta.className = 'meta';
    meta.textContent = (index + 1) + '. ' + it.camera + ' · ' + it.date + ' ' + it.time + ' · OCR membaca ';
    const own = document.createElement('span'); own.textContent = it.own || '(kosong)';
    if (it.own !== it.label) own.className = 'differs';
    meta.append(own, ' (' + Math.round(it.conf * 100) + '%)' + (it.labelled ? '' : ' · belum ada label, isian awal dari bacaan OCR'));
    const commit = (done) => {
      saved[it.image] = { plate: plate.value.toUpperCase().replace(/[^A-Z0-9]/g, ''), status: status.value, done };
      plate.value = saved[it.image].plate; row.dataset.done = String(done); store(); paintProgress();
    };
    plate.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault(); commit(true);
      const inputs = [...list.querySelectorAll('input[type=text]')];
      const next = inputs[inputs.indexOf(plate) + 1];
      if (next) { next.focus(); next.select(); }
    });
    plate.addEventListener('change', () => commit(row.dataset.done === 'true'));
    status.addEventListener('change', () => commit(true));
    fields.append(plateLabel, statusLabel, meta);
    row.append(img, fields);
    list.append(row);
  });
  paintProgress();
}
todoOnly.addEventListener('change', render);
document.getElementById('download').addEventListener('click', () => {
  const lines = ['image,plate,status'];
  ITEMS.forEach((it) => { const s = saved[it.image]; if (s && s.done) lines.push([it.image, s.plate, s.status].join(',')); });
  const link = document.createElement('a');
  link.href = URL.createObjectURL(new Blob([lines.join('\\r\\n') + '\\r\\n'], { type: 'text/csv' }));
  link.download = 'labels.csv'; link.click();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, required=True, help="folder written by build_ocr_dataset.py")
    parser.add_argument("--raw", type=Path, default=Path("data/dataset/raw/rog"), help="folder with media/, to shorten image names")
    parser.add_argument("--splits", default="test,val", help="comma separated: test, val, train, none (crops without a label)")
    args = parser.parse_args()

    wanted = {name.strip() for name in args.splits.split(",")}
    media = (args.raw / "media").resolve()
    items = []
    with (args.dataset / "all.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in wanted:
                continue
            path = Path(row["image_path"])
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            items.append({
                "image": path.relative_to(media).as_posix(),
                "data": f"data:image/jpeg;base64,{data}",
                # Without a label the reviewer starts from what the OCR read, not from nothing.
                "label": row["plate_text"] or row["own_text"],
                "labelled": bool(row["plate_text"]),
                "own": row["own_text"],
                "conf": float(row["own_confidence"]),
                "camera": row["camera"], "date": row["date"], "time": row["time"],
            })
    # Crops of one plate next to each other: the eye checks a run of the same text quickly.
    items.sort(key=lambda it: (it["label"] or "~", it["camera"], it["date"], it["time"]))

    out = args.dataset / "review"
    out.mkdir(parents=True, exist_ok=True)
    page = PAGE.replace("__ITEMS__", json.dumps(items)).replace("__KEY__", json.dumps(html.escape(str(sorted(wanted)))))
    (out / "review.html").write_text(page, encoding="utf-8")
    print(f"{len(items)} crops ({', '.join(sorted(wanted))}) -> {out / 'review.html'}  ({(out / 'review.html').stat().st_size / 1e6:.1f} MB)")
    print(f"After checking, save the downloaded labels.csv as {out / 'labels.csv'} and run build_ocr_dataset.py again.")


if __name__ == "__main__":
    main()
