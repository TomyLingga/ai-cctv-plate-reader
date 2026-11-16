// Shared by both pages: icons, API calls, admin token, formatting, toasts, page frame.

// Lucide icon paths (ISC licence), 24 x 24, drawn with the stroke set in app.css.
const ICONS = {
  video: '<path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5"/><rect x="2" y="6" width="14" height="12" rx="2"/>',
  dashboard: '<rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>',
  scan: '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/>',
  plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
  check: '<path d="M20 6 9 17l-5-5"/>',
  x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
  pencil: '<path d="M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z"/><path d="m15 5 4 4"/>',
  trash: '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
  eye: '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
  eyeOff: '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
  moon: '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  lock: '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
  activity: '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
  mapPin: '<path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/><circle cx="12" cy="10" r="3"/>',
};

export function icon(name) {
  const span = document.createElement('span');
  span.className = 'icon';
  span.setAttribute('aria-hidden', 'true');
  // Fixed strings from the table above, never data from the service.
  span.innerHTML = `<svg viewBox="0 0 24 24">${ICONS[name] || ''}</svg>`;
  return span;
}

/** Replace every <span data-icon="name"> inside `root` with the drawn icon. */
export function hydrateIcons(root = document) {
  for (const holder of root.querySelectorAll('[data-icon]')) {
    holder.replaceWith(icon(holder.dataset.icon));
  }
}

/** Small element builder: el('p', { class: 'muted', text: '...' }, child, ...). */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === 'text') node.textContent = value;
    else if (key === 'class') node.className = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined && child !== false));
  return node;
}

// -- storage that may be blocked ------------------------------------------------------------

function read(store, key) {
  try {
    return store.getItem(key);
  } catch {
    return null;
  }
}

function write(store, key, value) {
  try {
    if (value === null) store.removeItem(key);
    else store.setItem(key, value);
  } catch {
    // Blocked storage only means the value is not remembered.
  }
}

export const prefs = {
  get: (key) => read(localStorage, `alpr.${key}`),
  set: (key, value) => write(localStorage, `alpr.${key}`, value),
};

// -- admin token ----------------------------------------------------------------------------

const TOKEN_KEY = 'alpr.adminToken';

export const adminToken = {
  get: () => read(sessionStorage, TOKEN_KEY) || read(localStorage, TOKEN_KEY) || '',
  remembered: () => Boolean(read(localStorage, TOKEN_KEY)),
  set(value, remember) {
    write(sessionStorage, TOKEN_KEY, value);
    write(localStorage, TOKEN_KEY, remember ? value : null);
  },
  clear() {
    write(sessionStorage, TOKEN_KEY, null);
    write(localStorage, TOKEN_KEY, null);
  },
};

// -- API ------------------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(status, code, message, { fields = null, retryAfter = null } = {}) {
    super(message);
    this.status = status;
    this.code = code;
    this.fields = fields;
    this.retryAfter = retryAfter;
  }
}

export async function api(path, { method = 'GET', body, admin = false, signal } = {}) {
  const headers = {};
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  if (admin && adminToken.get()) headers.Authorization = `Bearer ${adminToken.get()}`;
  let response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      cache: 'no-store',
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(0, 'network', 'Service tidak bisa dihubungi.');
  }
  if (response.status === 204) return null;
  let data = null;
  try {
    data = await response.json();
  } catch {
    // An empty or non-JSON body is handled through the status below.
  }
  if (response.ok) return data;
  const detail = data ? data.detail : null;
  if (Array.isArray(detail)) {
    // FastAPI validation errors: one entry per field, the field name is the end of `loc`.
    const fields = {};
    for (const item of detail) fields[item.loc[item.loc.length - 1]] = item.msg;
    throw new ApiError(response.status, 'validation', 'Ada isian yang belum benar.', { fields });
  }
  if (detail && typeof detail === 'object') {
    const retryAfter = Number(response.headers.get('Retry-After')) || null;
    throw new ApiError(response.status, detail.code, detail.message, { retryAfter });
  }
  throw new ApiError(response.status, 'http', typeof detail === 'string' ? detail : `HTTP ${response.status}`);
}

// -- formatting -----------------------------------------------------------------------------

const number1 = new Intl.NumberFormat('id-ID', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const number0 = new Intl.NumberFormat('id-ID', { maximumFractionDigits: 0 });
const dayMonth = new Intl.DateTimeFormat('id-ID', { day: 'numeric', month: 'short' });

export const fmt1 = (value) => (value === null || value === undefined ? '–' : number1.format(value));
export const fmt0 = (value) => (value === null || value === undefined ? '–' : number0.format(value));
export const percent = (value) => (value === null || value === undefined ? '–' : `${number1.format(value * 100)}%`);

/** Clock time of an event as the service wrote it, with the date when it is not today. */
export function eventTime(timestamp) {
  const date = timestamp.slice(0, 10);
  const clock = timestamp.slice(11, 19);
  const now = new Date();
  const today = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  if (date === today) return clock;
  const [year, month, day] = date.split('-').map(Number);
  return `${dayMonth.format(new Date(year, month - 1, day))} ${clock}`;
}

export const LOW_CONFIDENCE = 0.7;

/** Notes an operator should see next to a reading. Works for events and for current plates. */
export function readingFlags(reading) {
  const flags = [];
  if (reading.format_valid === false) flags.push('Di luar pola plat umum');
  else if (reading.region_known === false) flags.push('Kode wilayah tidak dikenal');
  // The code as read was not on the list, and the nearest listed code took its place.
  if (reading.region_corrected) flags.push(reading.raw ? `Kode wilayah dikoreksi (OCR membaca ${reading.raw})` : 'Kode wilayah dikoreksi');
  if (reading.stopped === false) flags.push('Dibaca saat kendaraan lewat, tidak berhenti');
  if (typeof reading.confidence === 'number' && reading.confidence < LOW_CONFIDENCE) flags.push('Keyakinan rendah');
  if (reading.alternatives && reading.alternatives.length) flags.push(`Bisa juga: ${reading.alternatives.join(', ')}`);
  return flags;
}

export function flagTags(reading) {
  return readingFlags(reading).map((text) => el('span', { class: 'tag', dataset: { kind: 'warn' } }, icon('alert'), text));
}

// -- toasts ---------------------------------------------------------------------------------

export function toast(message, kind = 'ok') {
  let region = document.getElementById('toasts');
  if (!region) {
    region = el('div', { id: 'toasts', class: 'toasts', role: 'status', 'aria-live': 'polite' });
    document.body.append(region);
  }
  const node = el('div', { class: 'toast', dataset: { kind } }, icon(kind === 'error' ? 'alert' : 'check'), el('span', { text: message }));
  region.append(node);
  setTimeout(() => node.remove(), kind === 'error' ? 8000 : 5000);
}

/** Show a busy state on a button while `work` runs; the button cannot be pressed twice. */
export async function withBusy(button, work) {
  if (button.dataset.busy === 'true') return undefined;
  const spinner = el('span', { class: 'spinner', 'aria-hidden': 'true' });
  button.dataset.busy = 'true';
  button.setAttribute('aria-disabled', 'true');
  button.prepend(spinner);
  try {
    return await work();
  } finally {
    spinner.remove();
    delete button.dataset.busy;
    button.removeAttribute('aria-disabled');
  }
}

/** Modal dialogs close on a click outside their box and on every [data-close] button. */
export function wireDialog(dialog) {
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog || event.target.closest('[data-close]')) dialog.close();
  });
}

// -- page frame -----------------------------------------------------------------------------

function providerName(providers) {
  if (!providers || !providers.length) return '';
  if (providers[0].startsWith('CUDA')) return 'GPU (CUDA)';
  if (providers[0].startsWith('Tensorrt')) return 'GPU (TensorRT)';
  if (providers[0].startsWith('OpenVINO')) return 'OpenVINO';
  return 'CPU';
}

/** Theme switch and the service line in the top bar. Returns the function that refreshes that line. */
export function initShell() {
  hydrateIcons();
  const root = document.documentElement;
  const toggle = document.getElementById('theme-toggle');
  const paint = () => {
    const light = root.dataset.theme === 'light';
    toggle.replaceChildren(icon(light ? 'moon' : 'sun'));
    toggle.setAttribute('aria-label', light ? 'Pakai tampilan gelap' : 'Pakai tampilan terang');
  };
  toggle.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
    prefs.set('theme', root.dataset.theme);
    paint();
  });
  paint();

  const chip = document.getElementById('service-chip');
  return async function refreshService() {
    try {
      const health = await api('/health');
      chip.textContent = `v${health.version} · ${providerName(health.providers)}`;
      return health;
    } catch {
      chip.textContent = 'Service tidak menjawab';
      return null;
    }
  };
}
