// Camera page: list of cameras, admin token, and the form that adds or changes a camera.
import { initAdminCard } from './admin.js';
import { ApiError, api, el, fmt0, fmt1, icon, initShell, toast, wireDialog, withBusy } from './common.js';

const LIST_POLL_MS = 3000;
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/;

const TEST_MESSAGES = {
  login_rejected:
    'Kamera menolak nama pengguna atau sandi. Periksa keduanya sebelum mencoba lagi: beberapa kali salah berturut-turut bisa membuat kamera mengunci akunnya.',
  not_found: 'Kamera menjawab, tetapi jalur stream tidak ditemukan. Periksa bagian jalur, misalnya /Streaming/Channels/101.',
  unreachable: 'Kamera tidak bisa dihubungi. Periksa alamat IP dan port, dan pastikan PC service satu jaringan dengan kamera.',
  timeout: 'Kamera tidak menjawab dalam batas waktu. Periksa alamat IP, port, dan kabel jaringannya.',
  no_video: 'Alamat ini tidak mengirim video.',
  error: 'Uji sambungan gagal.',
};

const ERROR_MESSAGES = {
  network: 'Service tidak bisa dihubungi. Coba lagi sebentar lagi.',
  bad_token: 'Token admin ditolak. Masukkan lagi di bagian atas halaman.',
  admin_disabled: 'Token admin belum diatur di PC service. Isi ALPR_ADMIN_TOKEN di berkas .env, lalu jalankan ulang service.',
  duplicate_id: 'ID ini sudah dipakai kamera lain.',
  duplicate_legacy_id: 'Nomor rute lama ini sudah dipakai kamera lain.',
  camera_limit: 'Jumlah kamera sudah mencapai batas service ini.',
  config_camera: 'Kamera ini diatur di config/app.yaml dan hanya bisa diubah di sana.',
  unknown_camera: 'Kamera ini sudah tidak ada. Muat ulang halaman.',
};

const $ = (id) => document.getElementById(id);
const form = $('camera-form');
const f = {
  id: $('f-id'),
  description: $('f-description'),
  host: $('f-host'),
  port: $('f-port'),
  user: $('f-user'),
  password: $('f-password'),
  path: $('f-path'),
  url: $('f-url'),
  fps: $('f-fps'),
  legacy: $('f-legacy'),
  roi: [$('f-roi-left'), $('f-roi-top'), $('f-roi-right'), $('f-roi-bottom')],
};
const rows = $('camera-rows');
const stage = $('roi-stage');
const roiBox = $('roi-box');
const roiImg = $('roi-img');
const testButton = $('test-button');
const saveButton = $('save-button');
const confirmDialog = $('confirm-dialog');

let cameras = [];
let sources = new Map(); // camera id -> { origin, url, url_env }, only with a valid token
let listSignature = '';
const dynamicCells = new Map(); // camera id -> { processed, status, statusText }
let editing = null; // id of the camera being changed, or null while adding
let pendingDelete = null;

const refreshService = initShell();
wireDialog(confirmDialog);
refreshService();

// -- messages -------------------------------------------------------------------------------

function messageFor(error) {
  if (error instanceof ApiError) {
    if (error.code === 'login_locked') {
      const minutes = Math.max(Math.ceil((error.retryAfter || 600) / 60), 1);
      return `Kamera ini baru saja menolak login beberapa kali. Tunggu sekitar ${minutes} menit sebelum menguji lagi supaya akunnya tidak dikunci kamera.`;
    }
    return ERROR_MESSAGES[error.code] || error.message;
  }
  return 'Terjadi kesalahan yang tidak dikenal.';
}

function setFieldError(fieldId, message) {
  const note = $(`${fieldId}-error`);
  if (note) {
    note.replaceChildren(icon('alert'), el('span', { text: message }));
    note.hidden = false;
  }
  const input = $(fieldId);
  if (input) input.setAttribute('aria-invalid', 'true');
}

function clearErrors() {
  for (const note of form.querySelectorAll('.error')) note.hidden = true;
  for (const input of form.querySelectorAll('[aria-invalid]')) input.removeAttribute('aria-invalid');
  $('form-alert').hidden = true;
}

function showFormAlert(message) {
  $('form-alert-text').textContent = message;
  $('form-alert').hidden = false;
  $('form-alert').scrollIntoView({ block: 'nearest' });
}

function focusFirstInvalid() {
  const first = form.querySelector('[aria-invalid="true"]');
  if (first) first.focus();
}

// -- admin token ----------------------------------------------------------------------------

const admin = initAdminCard({
  texts: {
    ready: 'Kamera bisa ditambah, diubah, dan dihapus dari browser ini.',
    locked: 'Daftar kamera bisa dilihat tanpa token. Menambah, mengubah, menguji, dan menghapus kamera memerlukan token.',
  },
  async onChange(state) {
    $('form-locked').hidden = state.authorized;
    testButton.disabled = !state.authorized;
    saveButton.disabled = !state.authorized;
    if (!state.authorized) sources = new Map();
    listSignature = '';
    await refreshList();
  },
});
// -- camera list ----------------------------------------------------------------------------

function statusOf(camera) {
  if (camera.connected) return ['on', 'Tersambung'];
  return camera.last_error ? ['off', 'Terputus'] : ['idle', 'Menyambung…'];
}

function buildRow(camera) {
  const source = sources.get(camera.id);
  const fromConfig = camera.origin === 'config';
  const [state, stateText] = statusOf(camera);
  const statusText = el('span', { text: stateText });
  const status = el('span', { class: 'status', dataset: { state } }, el('span', { class: 'status-dot', 'aria-hidden': 'true' }), statusText);
  const processed = el('td', { 'data-label': 'Diproses', class: 'mono' });

  let address;
  if (!admin.state.authorized) address = el('span', { class: 'muted', text: 'Terlihat setelah token diisi' });
  else if (!source) address = el('span', { class: 'muted', text: '–' });
  else {
    address = el('span', { class: 'cell-url', text: source.url });
    if (source.url_env) address = el('span', {}, address, el('br'), el('span', { class: 'muted', text: `dari ${source.url_env} di .env` }));
  }

  let actions;
  if (fromConfig) {
    actions = el('span', { class: 'muted', text: 'Diubah lewat config/app.yaml' });
  } else {
    const edit = el('button', { type: 'button', class: 'btn small', 'aria-label': `Ubah kamera ${camera.id}` }, icon('pencil'), 'Ubah');
    const remove = el('button', { type: 'button', class: 'btn small', 'aria-label': `Hapus kamera ${camera.id}` }, icon('trash'), 'Hapus');
    edit.disabled = !admin.state.authorized;
    remove.disabled = !admin.state.authorized;
    edit.addEventListener('click', () => startEdit(camera.id));
    remove.addEventListener('click', () => askDelete(camera.id));
    actions = el('div', { class: 'row-actions' }, edit, remove);
  }

  dynamicCells.set(camera.id, { processed, status, statusText });
  return el(
    'tr',
    {},
    el('td', { 'data-label': 'Kamera' }, el('div', {}, el('strong', { class: 'mono', text: camera.id }), el('div', { class: 'muted', text: camera.description || 'Tanpa keterangan' }))),
    el('td', { 'data-label': 'Sumber', text: fromConfig ? 'config/app.yaml' : 'Halaman ini' }),
    el('td', { 'data-label': 'Alamat stream' }, address),
    el('td', { 'data-label': 'Rute lama', class: 'mono', text: camera.legacy_id ? `/plate/${camera.legacy_id}` : '–' }),
    processed,
    el('td', { 'data-label': 'Status' }, status),
    el('td', { 'data-label': 'Tindakan' }, actions),
  );
}

function renderList() {
  // Rows are rebuilt only when what they show has changed, so a poll does not take the
  // keyboard focus away from a button in the table.
  const signature = JSON.stringify([
    admin.state.authorized,
    cameras.map((camera) => [camera.id, camera.origin, camera.description, camera.legacy_id, (sources.get(camera.id) || {}).url]),
  ]);
  if (signature !== listSignature) {
    listSignature = signature;
    dynamicCells.clear();
    rows.replaceChildren(...cameras.map(buildRow));
  }
  for (const camera of cameras) {
    const cells = dynamicCells.get(camera.id);
    if (!cells) continue;
    const [state, stateText] = statusOf(camera);
    cells.status.dataset.state = state;
    cells.statusText.textContent = stateText;
    cells.status.title = !camera.connected && camera.last_error ? camera.last_error : '';
    cells.processed.textContent = camera.connected ? `${fmt1(camera.processed_fps)} / ${fmt0(camera.process_fps)} fps` : `– / ${fmt0(camera.process_fps)} fps`;
  }
  $('list-empty').hidden = cameras.length > 0;
  $('list-count').textContent = cameras.length ? `${cameras.length} kamera` : '';
}

async function refreshList() {
  try {
    cameras = await api('/cameras');
    if (admin.state.authorized) {
      try {
        sources = new Map((await api('/admin/cameras', { admin: true })).map((item) => [item.id, item]));
      } catch (error) {
        if (!admin.handleFailure(error)) throw error;
      }
    }
    renderList();
  } catch {
    $('list-count').textContent = 'Service tidak menjawab; daftar ini mungkin sudah lama.';
  }
}

// -- area of the picture that is read -------------------------------------------------------

function readRoi() {
  return f.roi.map((input) => Number(input.value));
}

function drawRoi() {
  const [left, top, right, bottom] = readRoi();
  const valid = [left, top, right, bottom].every((value) => Number.isFinite(value)) && left < right && top < bottom;
  roiBox.hidden = !valid;
  if (!valid) return;
  roiBox.style.left = `${left}%`;
  roiBox.style.top = `${top}%`;
  roiBox.style.width = `${right - left}%`;
  roiBox.style.height = `${bottom - top}%`;
}

function setRoi(values) {
  values.forEach((value, index) => {
    f.roi[index].value = String(value);
  });
  drawRoi();
}

function showStage(src) {
  roiImg.src = src;
  stage.hidden = false;
  $('roi-empty').hidden = true;
}

function hideStage() {
  stage.hidden = true;
  roiImg.removeAttribute('src');
  $('roi-empty').hidden = false;
}

roiImg.addEventListener('load', () => {
  stage.style.aspectRatio = `${roiImg.naturalWidth} / ${roiImg.naturalHeight}`;
  drawRoi();
});

for (const input of f.roi) input.addEventListener('input', drawRoi);
$('roi-reset').addEventListener('click', () => setRoi([0, 0, 100, 100]));

let drag = null;
stage.addEventListener('pointerdown', (event) => {
  if (event.button !== 0) return;
  drag = { x: event.clientX, y: event.clientY, rect: stage.getBoundingClientRect(), active: false };
  stage.setPointerCapture(event.pointerId);
});
stage.addEventListener('pointermove', (event) => {
  if (!drag) return;
  // A click or a shaky tap must not replace the area with a tiny one.
  if (!drag.active && Math.hypot(event.clientX - drag.x, event.clientY - drag.y) < 6) return;
  drag.active = true;
  const { rect } = drag;
  const across = (value) => Math.round(Math.min(Math.max(((value - rect.left) / rect.width) * 100, 0), 100));
  const down = (value) => Math.round(Math.min(Math.max(((value - rect.top) / rect.height) * 100, 0), 100));
  const xs = [across(drag.x), across(event.clientX)].sort((a, b) => a - b);
  const ys = [down(drag.y), down(event.clientY)].sort((a, b) => a - b);
  if (xs[1] - xs[0] >= 2 && ys[1] - ys[0] >= 2) setRoi([xs[0], ys[0], xs[1], ys[1]]);
});
for (const type of ['pointerup', 'pointercancel']) {
  stage.addEventListener(type, () => {
    drag = null;
  });
}

// -- form -----------------------------------------------------------------------------------

function urlMode() {
  return form.querySelector('input[name="url-mode"]:checked').value;
}

function paintUrlMode() {
  const full = urlMode() === 'full';
  $('url-parts').hidden = full;
  $('url-full').hidden = !full;
}

for (const radio of form.querySelectorAll('input[name="url-mode"]')) radio.addEventListener('change', paintUrlMode);

/** The address as typed, or '' when the address fields were left empty. */
function composeUrl() {
  if (urlMode() === 'full') return f.url.value.trim();
  const host = f.host.value.trim();
  if (!host) return '';
  const user = f.user.value.trim();
  const password = f.password.value;
  const login = user ? `${encodeURIComponent(user)}${password ? `:${encodeURIComponent(password)}` : ''}@` : '';
  const port = f.port.value.trim() || '554';
  let path = f.path.value.trim();
  if (path && !path.startsWith('/')) path = `/${path}`;
  const hostPart = host.includes(':') && !host.startsWith('[') ? `[${host}]` : host;
  return `rtsp://${login}${hostPart}:${port}${path}`;
}

function checkAddress(url) {
  const field = urlMode() === 'full' ? 'f-url' : 'f-host';
  if (!url) {
    if (editing) return true; // the stored address is kept
    setFieldError(field, 'Alamat kamera belum diisi.');
    return false;
  }
  if (urlMode() === 'full') {
    if (!/^rtsps?:\/\/[^/\s]+/i.test(url) || /\s/.test(url)) {
      setFieldError('f-url', 'Alamat harus diawali rtsp:// dan tidak boleh berisi spasi.');
      return false;
    }
    return true;
  }
  let ok = true;
  if (/[\s/@]/.test(f.host.value.trim())) {
    setFieldError('f-host', 'Isi alamat IP atau nama host saja, tanpa rtsp:// dan tanpa jalur.');
    ok = false;
  }
  const port = Number(f.port.value || 554);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    setFieldError('f-port', 'Port harus angka antara 1 dan 65535.');
    ok = false;
  }
  return ok;
}

/** Read and check the form. Returns { id, body } or null when something has to be corrected. */
function collect() {
  clearErrors();
  let ok = true;
  const id = f.id.value.trim();
  if (!editing) {
    if (!ID_PATTERN.test(id)) {
      setFieldError('f-id', 'Pakai huruf, angka, tanda hubung atau garis bawah, diawali huruf atau angka, paling banyak 32 karakter.');
      ok = false;
    } else if (cameras.some((camera) => camera.id === id)) {
      setFieldError('f-id', ERROR_MESSAGES.duplicate_id);
      ok = false;
    }
  }
  const url = composeUrl();
  if (!checkAddress(url)) ok = false;

  const fps = Number(f.fps.value);
  if (!f.fps.value.trim() || !Number.isFinite(fps) || fps < 1 || fps > 15) {
    setFieldError('f-fps', 'Isi angka antara 1 dan 15.');
    ok = false;
  }
  let legacy = null;
  if (f.legacy.value.trim()) {
    legacy = Number(f.legacy.value);
    if (!Number.isInteger(legacy) || legacy < 1 || legacy > 999) {
      setFieldError('f-legacy', 'Isi bilangan bulat antara 1 dan 999, atau kosongkan.');
      ok = false;
    } else if (cameras.some((camera) => camera.id !== editing && camera.legacy_id === legacy)) {
      setFieldError('f-legacy', ERROR_MESSAGES.duplicate_legacy_id);
      ok = false;
    }
  }
  const roi = readRoi();
  const roiFilled = f.roi.every((input) => input.value.trim() !== '');
  if (!roiFilled || roi.some((value) => !Number.isFinite(value) || value < 0 || value > 100)) {
    setFieldError('f-roi', 'Keempat angka area baca harus antara 0 dan 100.');
    ok = false;
  } else if (roi[0] >= roi[2] || roi[1] >= roi[3]) {
    setFieldError('f-roi', 'Kiri harus lebih kecil dari Kanan, dan Atas lebih kecil dari Bawah.');
    ok = false;
  }
  if (!ok) return null;
  const body = {
    description: f.description.value.trim(),
    legacy_id: legacy,
    roi: roi.map((value) => value / 100),
    process_fps: fps,
  };
  if (url) body.url = url;
  return { id: editing || id, body };
}

function showTest(kind, message) {
  const box = $('test-result');
  box.dataset.kind = kind;
  $('test-icon').replaceChildren(kind === 'pending' ? el('span', { class: 'spinner', 'aria-hidden': 'true' }) : icon(kind === 'ok' ? 'check' : 'alert'));
  $('test-text').textContent = message;
  box.hidden = false;
}

const CODEC_NAMES = { hevc: 'H.265', h264: 'H.264', mjpeg: 'MJPEG' };

async function runTest() {
  clearErrors();
  const url = composeUrl();
  if (!checkAddress(url)) {
    focusFirstInvalid();
    return;
  }
  const body = url ? { url } : { camera_id: editing };
  showTest('pending', 'Menghubungi kamera. Ini bisa makan waktu sampai 20 detik.');
  try {
    const result = await api('/cameras/test', { method: 'POST', body, admin: true });
    if (result.ok) {
      const codec = CODEC_NAMES[result.codec] || result.codec;
      const rate = result.fps ? `, ${fmt0(result.fps)} fps` : '';
      showTest('ok', `Tersambung: ${codec}, ${result.width} × ${result.height}${rate}. Kamera belum disimpan.`);
      if (result.snapshot) showStage(result.snapshot);
    } else {
      const detail = result.code === 'error' && result.detail ? ` Rincian: ${result.detail}` : '';
      showTest('error', (TEST_MESSAGES[result.code] || TEST_MESSAGES.error) + detail);
    }
  } catch (error) {
    if (admin.handleFailure(error)) $('test-result').hidden = true;
    else if (error.code === 'validation') showTest('error', 'Alamat kamera ditolak service. Periksa lagi penulisannya.');
    else showTest('error', messageFor(error));
  }
}

const FIELD_OF = { id: 'f-id', description: 'f-description', legacy_id: 'f-legacy', roi: 'f-roi', process_fps: 'f-fps' };

function handleSaveError(error) {
  if (admin.handleFailure(error)) {
    showFormAlert(messageFor(error));
    return;
  }
  if (error.code === 'duplicate_id') setFieldError('f-id', ERROR_MESSAGES.duplicate_id);
  else if (error.code === 'duplicate_legacy_id') setFieldError('f-legacy', ERROR_MESSAGES.duplicate_legacy_id);
  else if (error.code === 'validation' && error.fields) {
    for (const name of Object.keys(error.fields)) {
      const field = name === 'url' ? (urlMode() === 'full' ? 'f-url' : 'f-host') : FIELD_OF[name];
      if (field) setFieldError(field, 'Isian ini ditolak service. Periksa lagi nilainya.');
    }
    showFormAlert('Kamera belum disimpan karena ada isian yang ditolak.');
  } else {
    showFormAlert(messageFor(error));
  }
  focusFirstInvalid();
}

function resetForm() {
  editing = null;
  form.reset();
  clearErrors();
  f.id.readOnly = false;
  $('h-form').textContent = 'Tambah kamera';
  $('save-label').textContent = 'Simpan kamera';
  $('cancel-edit').hidden = true;
  $('url-keep').hidden = true;
  $('test-result').hidden = true;
  $('f-id-help').hidden = false;
  paintUrlMode();
  hideStage();
  drawRoi();
}

function startEdit(cameraId) {
  const camera = cameras.find((item) => item.id === cameraId);
  if (!camera) return;
  resetForm();
  editing = camera.id;
  $('h-form').textContent = `Ubah kamera ${camera.id}`;
  $('save-label').textContent = 'Simpan perubahan';
  $('cancel-edit').hidden = false;
  f.id.value = camera.id;
  f.id.readOnly = true;
  $('f-id-help').hidden = true;
  f.description.value = camera.description || '';
  f.fps.value = String(camera.process_fps);
  f.legacy.value = camera.legacy_id ? String(camera.legacy_id) : '';
  setRoi(camera.roi.map((value) => Math.round(value * 100)));
  const source = sources.get(camera.id);
  $('url-keep').textContent = `Alamat yang tersimpan: ${source ? source.url : 'tidak terlihat'}. Biarkan kolom alamat kosong untuk tetap memakainya; isi hanya bila alamat atau sandinya berubah.`;
  $('url-keep').hidden = false;
  showStage(`/cameras/${encodeURIComponent(camera.id)}/snapshot.jpg?width=960&t=${Date.now()}`);
  $('h-form').scrollIntoView({ block: 'start' });
  f.description.focus({ preventScroll: true });
}

function askDelete(cameraId) {
  pendingDelete = cameraId;
  $('confirm-text').textContent =
    `Kamera ${cameraId} berhenti dibaca dan hilang dari daftar. Riwayat pembacaan dan gambar yang sudah tersimpan tidak ikut terhapus.`;
  confirmDialog.showModal();
}

$('confirm-yes').addEventListener('click', () => {
  const cameraId = pendingDelete;
  withBusy($('confirm-yes'), async () => {
    try {
      await api(`/cameras/${encodeURIComponent(cameraId)}`, { method: 'DELETE', admin: true });
      toast(`Kamera ${cameraId} dihapus.`);
      if (editing === cameraId) resetForm();
    } catch (error) {
      admin.handleFailure(error);
      toast(messageFor(error), 'error');
    }
    confirmDialog.close();
    await refreshList();
  });
});

testButton.addEventListener('click', () => withBusy(testButton, runTest));
$('cancel-edit').addEventListener('click', resetForm);

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const data = collect();
  if (!data) {
    focusFirstInvalid();
    return;
  }
  withBusy(saveButton, async () => {
    try {
      if (editing) {
        await api(`/cameras/${encodeURIComponent(data.id)}`, { method: 'PUT', body: data.body, admin: true });
        toast(`Perubahan kamera ${data.id} disimpan.`);
      } else {
        await api('/cameras', { method: 'POST', body: { id: data.id, ...data.body }, admin: true });
        toast(`Kamera ${data.id} ditambahkan. Statusnya muncul di daftar dalam beberapa detik.`);
      }
      resetForm();
      await refreshList();
    } catch (error) {
      handleSaveError(error);
    }
  });
});

// -- start ----------------------------------------------------------------------------------

paintUrlMode();
drawRoi();
admin.check();
setInterval(() => {
  if (!document.hidden) refreshList();
}, LIST_POLL_MS);
setInterval(refreshService, 10000);
