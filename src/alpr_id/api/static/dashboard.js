// Monitoring page: every camera with its newest picture, the plate in view and its last readings.
import {
  LOW_CONFIDENCE, api, el, eventTime, flagTags, fmt0, fmt1, icon, initShell, percent, prefs, readingFlags, wireDialog,
} from './common.js';

const CAMERA_POLL_MS = 1000;
const HIDDEN_POLL_MS = 5000;
const HEALTH_POLL_MS = 5000;
const FEED_LIMIT = 80;
const RECENT_PER_CAMERA = 4;
const TICK_MS = 250;
const FRESH_MS = 2500;
const ZOOM_MAX = 8; // the saved frame is 2560 px wide and shown at about a third of that
const ZOOM_MIN_DRAG_PX = 8; // anything smaller is a click, not a selection

const grid = document.getElementById('camera-grid');
const template = document.getElementById('tpl-camera');
const noCameras = document.getElementById('no-cameras');
const summary = document.getElementById('summary');
const live = document.getElementById('live');
const liveText = document.getElementById('live-text');
const alertBox = document.getElementById('alert');
const alertText = document.getElementById('alert-text');
const refreshSelect = document.getElementById('refresh');
const feedList = document.getElementById('feed-list');
const feedEmpty = document.getElementById('feed-empty');
const feedFilter = document.getElementById('feed-filter');
const eventDialog = document.getElementById('event-dialog');
const cameraDialog = document.getElementById('camera-dialog');
const cameraDialogImg = document.getElementById('camera-dialog-img');

const cards = new Map(); // camera id -> card
const recent = new Map(); // camera id -> newest events of that camera
const fresh = new Set(); // event ids that arrived a moment ago
let feed = [];
let filter = '';
let refreshMs = Number(prefs.get('refreshMs') ?? 1000);
let camerasOk = false;
let streamOk = false;
let streamDropped = false;
let eventsStored = null;
let dialogCamera = null;
let dialogPart = null; // enlarged part of the camera frame as fractions, null for the whole frame
const dialogTarget = { img: cameraDialogImg, loading: false, objectUrl: null };

const refreshService = initShell();
wireDialog(eventDialog);
wireDialog(cameraDialog);
const resetZoom = wireZoom(
  document.getElementById('event-frame'),
  document.getElementById('event-snapshot'),
  document.getElementById('event-zoom-reset'),
);
eventDialog.addEventListener('close', resetZoom);
const resetLiveZoom = wireLiveZoom(
  document.getElementById('camera-frame'),
  cameraDialogImg,
  document.getElementById('camera-zoom-reset'),
  (part) => {
    dialogPart = part;
    if (dialogCamera) loadSnapshot(dialogTarget, dialogCamera, 1280, dialogPart); // do not wait for the next tick
  },
);
cameraDialog.addEventListener('close', () => {
  dialogCamera = null;
  resetLiveZoom();
});

// -- pictures -------------------------------------------------------------------------------

async function loadSnapshot(target, cameraId, width, part = null) {
  if (target.loading) return;
  target.loading = true;
  try {
    const crop = part ? `&crop=${part.map((value) => value.toFixed(4)).join(',')}` : '';
    const response = await fetch(`/cameras/${encodeURIComponent(cameraId)}/snapshot.jpg?width=${width}${crop}`, { cache: 'no-store' });
    if (!response.ok) return;
    const url = URL.createObjectURL(await response.blob());
    const previous = target.objectUrl;
    target.objectUrl = url;
    target.img.src = url;
    if (previous) URL.revokeObjectURL(previous);
  } catch {
    // The last picture stays; the status of the camera says what is wrong.
  } finally {
    target.loading = false;
  }
}

// Cards scrolled out of view do not fetch pictures.
const observer = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    const card = cards.get(entry.target.dataset.camera);
    if (card) card.inView = entry.isIntersecting;
  }
});

setInterval(() => {
  if (document.hidden) return;
  if (dialogCamera) {
    loadSnapshot(dialogTarget, dialogCamera, 1280, dialogPart);
    return;
  }
  if (refreshMs === 0) return;
  const now = performance.now();
  for (const card of cards.values()) {
    if (!card.inView || now - card.lastShot < refreshMs) continue;
    card.lastShot = now;
    loadSnapshot(card, card.id, 640);
  }
}, TICK_MS);

// -- readings -------------------------------------------------------------------------------

function readRow(event, { showCamera }) {
  const label = event.plate_formatted || event.plate || 'Tidak terbaca';
  const where = showCamera ? `${event.camera_id} · ${eventTime(event.timestamp)}` : eventTime(event.timestamp);
  const crop = el('img', { class: 'crop', alt: '', loading: 'lazy', width: 96, height: 32 });
  if (event.crop_url) crop.src = event.crop_url;
  // Readings that deserve a second look carry a mark; the reasons are in the label and the detail view.
  const notes = readingFlags(event);
  const mark = notes.length ? el('span', { class: 'mark', title: notes.join('. ') }, icon('alert')) : null;
  const spoken = `${label}, ${where}, keyakinan ${percent(event.confidence)}.${notes.length ? ` ${notes.join('. ')}.` : ''} Buka rincian.`;
  const button = el(
    'button',
    { type: 'button', class: 'read-btn', 'aria-label': spoken },
    crop,
    el('span', { class: 'read-main' }, el('span', { class: 'plate-sm' }, label, mark), el('span', { class: 'read-sub', text: where })),
    el('span', { class: 'conf', text: percent(event.confidence), dataset: { level: event.confidence < LOW_CONFIDENCE ? 'low' : 'ok' } }),
  );
  button.addEventListener('click', () => openEvent(event));
  return el('li', { class: 'read-item', dataset: { fresh: fresh.has(event.event_id) ? 'true' : 'false' } }, button);
}

function renderFeed() {
  feedList.replaceChildren(...feed.map((event) => readRow(event, { showCamera: true })));
  feedEmpty.hidden = feed.length > 0;
  feedEmpty.textContent = filter ? `Belum ada pembacaan dari ${filter}.` : 'Belum ada pembacaan.';
}

function renderRecent(cameraId) {
  const card = cards.get(cameraId);
  if (!card) return;
  const events = recent.get(cameraId) || [];
  card.list.replaceChildren(...events.map((event) => readRow(event, { showCamera: false })));
  card.listEmpty.hidden = events.length > 0;
}

async function loadFeed() {
  const query = filter ? `&camera=${encodeURIComponent(filter)}` : '';
  try {
    feed = await api(`/events?limit=${FEED_LIMIT}${query}`);
    renderFeed();
  } catch {
    // The banner of the camera poll already reports an unreachable service.
  }
}

async function loadRecent(cameraId) {
  try {
    recent.set(cameraId, await api(`/events?camera=${encodeURIComponent(cameraId)}&limit=${RECENT_PER_CAMERA}`));
    renderRecent(cameraId);
  } catch {
    // Same as above.
  }
}

function addEvent(event) {
  fresh.add(event.event_id);
  setTimeout(() => fresh.delete(event.event_id), FRESH_MS);
  const mine = (recent.get(event.camera_id) || []).filter((item) => item.event_id !== event.event_id);
  recent.set(event.camera_id, [event, ...mine].slice(0, RECENT_PER_CAMERA));
  renderRecent(event.camera_id);
  if (!filter || filter === event.camera_id) {
    feed = [event, ...feed.filter((item) => item.event_id !== event.event_id)].slice(0, FEED_LIMIT);
    renderFeed();
  }
  if (eventsStored !== null) eventsStored += 1;
}

function openEvent(event) {
  const label = event.plate_formatted || event.plate || 'Tidak terbaca';
  document.getElementById('event-title').textContent = `Pembacaan ${label}`;
  document.getElementById('event-plate').textContent = label;
  const crop = document.getElementById('event-crop');
  crop.hidden = !event.crop_url;
  if (event.crop_url) crop.src = event.crop_url;
  document.getElementById('event-flags').replaceChildren(...flagTags(event));
  const frame = document.getElementById('event-frame');
  resetZoom();
  frame.hidden = !event.snapshot_url;
  document.getElementById('event-zoom-hint').hidden = !event.snapshot_url;
  if (event.snapshot_url) document.getElementById('event-snapshot').src = event.snapshot_url;
  const facts = [
    ['Kamera', event.camera_id],
    ['Waktu', `${event.timestamp.slice(0, 10)} ${event.timestamp.slice(11, 19)}`],
    ['Keyakinan', percent(event.confidence)],
    ['Jumlah bacaan OCR', event.reads],
    ['Bacaan mentah', event.raw || '–'],
    ['Nomor lacak', event.track_id],
    ['ID event', event.event_id],
  ];
  document.getElementById('event-facts').replaceChildren(
    ...facts.map(([name, value]) => el('div', {}, el('dt', { text: name }), el('dd', { text: String(value ?? '–') }))),
  );
  eventDialog.showModal();
}

// -- enlarging a picture --------------------------------------------------------------------

/**
 * Lets the user drag a rectangle over `frame` with the mouse or a pen. `onSelect` gets the
 * rectangle in frame pixels once the button is released. Returns the function that drops a
 * drag in progress.
 */
function wireDragBox(frame, onSelect) {
  const box = el('div', { class: 'zoom-box', hidden: true });
  frame.append(box);
  let start = null;

  function cancel() {
    start = null;
    box.hidden = true;
  }

  function pointIn(event) {
    const rect = frame.getBoundingClientRect();
    return {
      x: Math.min(Math.max(event.clientX - rect.left, 0), rect.width),
      y: Math.min(Math.max(event.clientY - rect.top, 0), rect.height),
    };
  }

  frame.addEventListener('pointerdown', (event) => {
    if (event.pointerType === 'touch' || event.button !== 0 || event.target.closest('button')) return;
    event.preventDefault(); // otherwise the browser starts dragging the image itself
    start = pointIn(event);
    frame.setPointerCapture(event.pointerId);
  });

  frame.addEventListener('pointermove', (event) => {
    if (!start) return;
    const now = pointIn(event);
    box.style.left = `${Math.min(start.x, now.x)}px`;
    box.style.top = `${Math.min(start.y, now.y)}px`;
    box.style.width = `${Math.abs(now.x - start.x)}px`;
    box.style.height = `${Math.abs(now.y - start.y)}px`;
    box.hidden = false;
  });

  frame.addEventListener('pointerup', (event) => {
    if (!start) return;
    const end = pointIn(event);
    const width = Math.abs(end.x - start.x);
    const height = Math.abs(end.y - start.y);
    const left = Math.min(start.x, end.x);
    const top = Math.min(start.y, end.y);
    cancel();
    if (width < ZOOM_MIN_DRAG_PX || height < ZOOM_MIN_DRAG_PX) return;
    onSelect({ left, top, width, height });
  });

  frame.addEventListener('pointercancel', cancel);
  return cancel;
}

/**
 * Lets the user drag a rectangle over the picture in `frame`; that part then fills the frame.
 * Dragging again enlarges further. A double click or `resetButton` shows the whole picture.
 * Returns the function that shows the whole picture.
 */
function wireZoom(frame, img, resetButton) {
  // Where the picture sits: its top left corner in frame pixels, and how much it is enlarged.
  let view = { scale: 1, x: 0, y: 0 };

  function show() {
    img.style.transform = view.scale === 1 ? '' : `translate(${view.x}px, ${view.y}px) scale(${view.scale})`;
    resetButton.hidden = view.scale === 1;
  }

  const cancel = wireDragBox(frame, ({ left, top, width, height }) => {
    // Not clientWidth: that is rounded, and the rounding error grows with the enlargement.
    const { width: frameWidth, height: frameHeight } = frame.getBoundingClientRect();
    // The selection measured on the picture as it is without enlargement.
    const picture = {
      x: (left - view.x) / view.scale,
      y: (top - view.y) / view.scale,
      width: width / view.scale,
      height: height / view.scale,
    };
    const scale = Math.min(frameWidth / picture.width, frameHeight / picture.height, ZOOM_MAX);
    // Centre the selection, then keep the picture from leaving an empty edge in the frame.
    const x = frameWidth / 2 - scale * (picture.x + picture.width / 2);
    const y = frameHeight / 2 - scale * (picture.y + picture.height / 2);
    view = {
      scale,
      x: Math.min(0, Math.max(frameWidth - frameWidth * scale, x)),
      y: Math.min(0, Math.max(frameHeight - frameHeight * scale, y)),
    };
    show();
  });

  function reset() {
    cancel();
    view = { scale: 1, x: 0, y: 0 };
    show();
  }

  frame.addEventListener('dblclick', reset);
  resetButton.addEventListener('click', reset);
  return reset;
}

/**
 * The same dragging for a live camera picture. Here the enlarged part is not the small
 * picture stretched: `onChange` gets the part as fractions [x1, y1, x2, y2] of the camera
 * frame (null for the whole frame), and the caller asks the service for that part, cut from
 * the full-resolution frame. Returns the function that shows the whole frame.
 */
function wireLiveZoom(frame, img, resetButton, onChange) {
  let part = null;

  function set(next) {
    part = next;
    resetButton.hidden = part === null;
    onChange(part);
  }

  const cancel = wireDragBox(frame, ({ left, top, width, height }) => {
    // Where the picture really is inside the frame: `object-fit: contain` can leave bars.
    const outer = frame.getBoundingClientRect();
    const inner = img.getBoundingClientRect();
    const ratio = img.naturalWidth && img.naturalHeight ? img.naturalWidth / img.naturalHeight : 16 / 9;
    const shownWidth = Math.min(inner.width, inner.height * ratio);
    const shownHeight = shownWidth / ratio;
    const originX = inner.left - outer.left + (inner.width - shownWidth) / 2;
    const originY = inner.top - outer.top + (inner.height - shownHeight) / 2;
    const unit = (value) => Math.min(Math.max(value, 0), 1);
    // The selection as fractions of what is shown now, then of the whole camera frame.
    const [px1, py1, px2, py2] = part ?? [0, 0, 1, 1];
    const x1 = px1 + unit((left - originX) / shownWidth) * (px2 - px1);
    const x2 = px1 + unit((left + width - originX) / shownWidth) * (px2 - px1);
    const y1 = py1 + unit((top - originY) / shownHeight) * (py2 - py1);
    const y2 = py1 + unit((top + height - originY) / shownHeight) * (py2 - py1);
    // Equal fractions of width and height keep the shape of the camera frame, so the part
    // fills the picture without bars. Grown around the middle of the selection, kept inside.
    const side = Math.max(x2 - x1, y2 - y1, 1 / ZOOM_MAX);
    if (side >= 0.98) {
      set(null);
      return;
    }
    const startX = Math.min(Math.max((x1 + x2) / 2 - side / 2, 0), 1 - side);
    const startY = Math.min(Math.max((y1 + y2) / 2 - side / 2, 0), 1 - side);
    set([startX, startY, startX + side, startY + side]);
  });

  function reset() {
    cancel();
    if (part !== null) set(null);
  }

  frame.addEventListener('dblclick', reset);
  resetButton.addEventListener('click', reset);
  return reset;
}

// -- camera cards ---------------------------------------------------------------------------

function createCard(camera) {
  const root = template.content.firstElementChild.cloneNode(true);
  root.dataset.camera = camera.id;
  const card = {
    id: camera.id,
    root,
    desc: root.querySelector('.camera-desc'),
    status: root.querySelector('.status'),
    statusText: root.querySelector('.status-text'),
    view: root.querySelector('.camera-view'),
    img: root.querySelector('.camera-view img'),
    note: root.querySelector('.camera-note'),
    plate: root.querySelector('.plate'),
    meta: root.querySelector('.reading-meta'),
    flags: root.querySelector('.flags'),
    metrics: Object.fromEntries([...root.querySelectorAll('[data-metric]')].map((node) => [node.dataset.metric, node])),
    list: root.querySelector('.read-list'),
    listEmpty: root.querySelector('.recent-empty'),
    inView: true,
    loading: false,
    lastShot: 0,
    objectUrl: null,
    plateText: null,
    flagKey: '',
    description: '',
  };
  root.querySelector('.camera-id').textContent = camera.id;
  card.img.alt = `Gambar terbaru ${camera.id}`;
  card.view.setAttribute('aria-label', `Perbesar gambar ${camera.id}`);
  card.view.addEventListener('click', () => openCamera(card));
  grid.append(root);
  cards.set(camera.id, card);
  observer.observe(root);
  feedFilter.append(el('option', { value: camera.id, text: camera.id }));
  loadSnapshot(card, camera.id, 640);
  loadRecent(camera.id);
  return card;
}

function removeCard(card) {
  observer.unobserve(card.root);
  card.root.remove();
  if (card.objectUrl) URL.revokeObjectURL(card.objectUrl);
  cards.delete(card.id);
  recent.delete(card.id);
  for (const option of feedFilter.options) {
    if (option.value === card.id) option.remove();
  }
}

function setMetric(node, text, warn = false) {
  node.textContent = text;
  node.dataset.level = warn ? 'warn' : 'ok';
}

function updateCard(card, camera) {
  card.description = camera.description || '';
  card.desc.textContent = camera.description || 'Tanpa keterangan';

  const state = camera.connected ? 'on' : camera.last_error ? 'off' : 'idle';
  card.status.dataset.state = state;
  card.statusText.textContent = { on: 'Tersambung', off: 'Terputus', idle: 'Menyambung…' }[state];

  let note = '';
  if (state === 'off') note = `Terputus: ${camera.last_error}`;
  else if (refreshMs === 0) note = 'Gambar dijeda';
  card.note.textContent = note;
  card.note.hidden = !note;

  const current = camera.current;
  const text = current && current.visible && current.plate ? current.plate_formatted || current.plate : null;
  // A plate that is followed but not read yet belongs to a vehicle that is still rolling:
  // reads start once it stands still.
  const idle = camera.plates_in_view > 0 ? 'Plat terlihat, belum dibaca' : 'Tidak ada kendaraan';
  if (text !== card.plateText || (!text && card.plate.textContent !== idle)) {
    card.plateText = text;
    card.plate.textContent = text || idle;
    card.plate.dataset.empty = text ? 'false' : 'true';
    card.plate.dataset.fresh = 'false';
    if (text) {
      void card.plate.offsetWidth; // restart the highlight for a new plate
      card.plate.dataset.fresh = 'true';
    }
  }
  if (text) {
    card.meta.replaceChildren(el('strong', { text: percent(current.confidence) }), ` · ${current.reads} bacaan`);
  } else {
    card.meta.replaceChildren();
  }
  const tags = text ? flagTags(current) : [];
  const flagKey = tags.map((tag) => tag.textContent).join('|');
  if (flagKey !== card.flagKey) {
    card.flagKey = flagKey;
    card.flags.replaceChildren(...tags);
  }

  // Units are in the labels of the page, so the values stay short enough for a narrow card.
  const latency = camera.latency_ms && camera.latency_ms.total;
  setMetric(card.metrics.fps, camera.connected ? fmt1(camera.fps) : '–');
  setMetric(
    card.metrics.processed,
    camera.connected ? `${fmt1(camera.processed_fps)} / ${fmt0(camera.process_fps)}` : '–',
    camera.connected && camera.processed_fps < camera.process_fps * 0.8,
  );
  setMetric(card.metrics.latency, latency ? fmt0(latency.p50_ms) : '–', Boolean(latency) && latency.p95_ms > 100);
  card.metrics.latency.title = latency
    ? `Separuh frame selesai dalam ${fmt0(latency.p50_ms)} ms, 95% dalam ${fmt0(latency.p95_ms)} ms`
    : '';
  setMetric(
    card.metrics.lag,
    camera.lag_s === null || camera.lag_s === undefined ? '–' : fmt0(camera.lag_s * 1000),
    camera.lag_s > 1,
  );
}

function openCamera(card) {
  dialogCamera = card.id;
  document.getElementById('camera-dialog-title').textContent = card.description ? `${card.id} · ${card.description}` : card.id;
  document.getElementById('camera-dialog-note').textContent =
    'Gambar diperbarui empat kali per detik selama jendela ini terbuka. Kotak hijau menandai plat yang terdeteksi. '
    + 'Seret di gambar untuk memperbesar bagian itu; klik dua kali untuk kembali ke tampilan penuh.';
  cameraDialogImg.alt = `Gambar terbaru ${card.id}`;
  if (card.objectUrl) cameraDialogImg.src = card.img.src;
  cameraDialog.showModal();
}

function syncCards(list) {
  for (const node of grid.querySelectorAll('.skeleton')) node.remove();
  const seen = new Set();
  for (const camera of list) {
    seen.add(camera.id);
    updateCard(cards.get(camera.id) || createCard(camera), camera);
  }
  for (const card of [...cards.values()]) {
    if (!seen.has(card.id)) removeCard(card);
  }
  noCameras.hidden = list.length > 0;
  if (filter && !seen.has(filter)) {
    filter = '';
    feedFilter.value = '';
    loadFeed();
  }
  const connected = list.filter((camera) => camera.connected).length;
  const stored = eventsStored === null ? '' : ` · ${fmt0(eventsStored)} pembacaan tersimpan`;
  summary.textContent = `${connected} dari ${list.length} kamera tersambung${stored}`;
}

// -- connection state -----------------------------------------------------------------------

function paintLive() {
  const on = camerasOk && streamOk;
  live.dataset.state = on ? 'on' : camerasOk || streamOk ? 'idle' : 'off';
  if (on) liveText.textContent = `Langsung · ${new Date().toTimeString().slice(0, 8)}`;
  else if (!camerasOk) liveText.textContent = 'Terputus dari service';
  else liveText.textContent = 'Menyambung ke aliran pembacaan…';
}

async function pollCameras() {
  try {
    syncCards(await api('/cameras'));
    camerasOk = true;
    alertBox.hidden = true;
  } catch {
    camerasOk = false;
    alertText.textContent = 'Service tidak bisa dihubungi. Data di halaman ini mungkin sudah lama; halaman mencoba menyambung lagi sendiri.';
    alertBox.hidden = false;
  }
  paintLive();
  setTimeout(pollCameras, document.hidden ? HIDDEN_POLL_MS : CAMERA_POLL_MS);
}

async function pollHealth() {
  const health = await refreshService();
  if (health) eventsStored = health.events_stored;
  setTimeout(pollHealth, HEALTH_POLL_MS);
}

function connectStream() {
  const source = new EventSource('/events/stream');
  source.addEventListener('open', () => {
    streamOk = true;
    if (streamDropped) {
      // Readings made while the stream was down are fetched again.
      streamDropped = false;
      loadFeed();
      for (const id of cards.keys()) loadRecent(id);
    }
    paintLive();
  });
  source.addEventListener('error', () => {
    streamOk = false;
    streamDropped = true;
    paintLive();
  });
  source.addEventListener('plate', (message) => addEvent(JSON.parse(message.data)));
}

// -- controls -------------------------------------------------------------------------------

if (![...refreshSelect.options].some((option) => Number(option.value) === refreshMs)) refreshMs = 1000;
refreshSelect.value = String(refreshMs);
refreshSelect.addEventListener('change', () => {
  refreshMs = Number(refreshSelect.value);
  prefs.set('refreshMs', String(refreshMs));
});

feedFilter.addEventListener('change', () => {
  filter = feedFilter.value;
  loadFeed();
});

pollHealth();
pollCameras();
loadFeed();
connectStream();
