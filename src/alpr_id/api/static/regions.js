// Region code page: the list of codes accepted as known, with adding and removing.
import { initAdminCard } from './admin.js';
import { ApiError, api, el, fmt0, icon, initShell, toast, wireDialog, withBusy } from './common.js';

const LIST_POLL_MS = 10000;
const CODE_PATTERN = /^[A-Z]{1,2}$/;

const ERROR_MESSAGES = {
  network: 'Service tidak bisa dihubungi. Coba lagi sebentar lagi.',
  bad_token: 'Token admin ditolak. Masukkan lagi di bagian atas halaman.',
  admin_disabled: 'Token admin belum diatur di PC service. Isi ALPR_ADMIN_TOKEN di berkas .env, lalu jalankan ulang service.',
  duplicate_code: 'Kode ini sudah ada di daftar.',
  special_code: 'Kode ini kode diplomatik atau pejabat negara, dan selalu diterima tanpa perlu didaftarkan.',
  unknown_code: 'Kode ini sudah tidak ada di daftar.',
};

const $ = (id) => document.getElementById(id);
const form = $('region-form');
const codeInput = $('f-code');
const nameInput = $('f-name');
const filterInput = $('filter');
const rows = $('region-rows');
const addButton = $('add-button');
const restoreButton = $('restore-button');
const confirmDialog = $('confirm-dialog');

let regions = [];
let listSignature = '';
let pending = null; // what the open confirmation is about: { kind: 'remove', code } or { kind: 'restore' }

const refreshService = initShell();
wireDialog(confirmDialog);
refreshService();

const admin = initAdminCard({
  texts: {
    ready: 'Kode bisa ditambah dan dihapus dari browser ini. Perubahan langsung berlaku untuk semua kamera.',
    locked: 'Daftar kode bisa dilihat tanpa token. Menambah dan menghapus kode memerlukan token.',
  },
  async onChange(state) {
    $('form-locked').hidden = state.authorized;
    addButton.disabled = !state.authorized;
    restoreButton.disabled = !state.authorized;
    listSignature = '';
    await refreshList();
  },
});

function messageFor(error) {
  return error instanceof ApiError ? ERROR_MESSAGES[error.code] || error.message : 'Terjadi kesalahan yang tidak dikenal.';
}

function setCodeError(message) {
  $('f-code-error').replaceChildren(icon('alert'), el('span', { text: message }));
  $('f-code-error').hidden = false;
  codeInput.setAttribute('aria-invalid', 'true');
}

function clearErrors() {
  $('f-code-error').hidden = true;
  codeInput.removeAttribute('aria-invalid');
  $('form-alert').hidden = true;
}

// -- list -----------------------------------------------------------------------------------

function buildRow(region) {
  const remove = el('button', { type: 'button', class: 'btn small', 'aria-label': `Hapus kode ${region.code}` }, icon('trash'), 'Hapus');
  remove.disabled = !admin.state.authorized;
  remove.addEventListener('click', () => askRemove(region));
  return el(
    'tr',
    {},
    el('td', { 'data-label': 'Kode' }, el('strong', { class: 'mono', text: region.code })),
    el('td', { 'data-label': 'Wilayah', text: region.name || '–' }),
    el('td', { 'data-label': 'Asal', text: region.builtin ? 'Bawaan' : 'Ditambahkan' }),
    el('td', { 'data-label': 'Terbaca di sini', class: 'mono', text: region.seen ? `${fmt0(region.seen)} kali` : '–' }),
    el('td', { 'data-label': 'Tindakan' }, remove),
  );
}

function renderList() {
  const query = filterInput.value.trim().toLowerCase();
  const shown = query
    ? regions.filter((region) => region.code.toLowerCase().startsWith(query) || region.name.toLowerCase().includes(query))
    : regions;
  // Rebuilt only when something shown has changed, so a refresh does not take the keyboard
  // focus away from a button in the table.
  const signature = JSON.stringify([admin.state.authorized, shown]);
  if (signature !== listSignature) {
    listSignature = signature;
    rows.replaceChildren(...shown.map(buildRow));
  }
  const added = regions.filter((region) => !region.builtin).length;
  $('list-count').textContent = regions.length
    ? `${regions.length} kode${added ? `, ${added} di antaranya ditambahkan` : ''}${query ? `; ${shown.length} cocok dengan pencarian` : ''}`
    : '';
  const empty = $('list-empty');
  empty.hidden = shown.length > 0;
  empty.textContent = regions.length
    ? 'Tidak ada kode yang cocok dengan pencarian.'
    : 'Daftar kosong, jadi tidak ada kode yang dianggap sah dan tidak ada yang dikoreksi. Pulihkan kode bawaan atau tambah kode sendiri.';
}

async function refreshList() {
  try {
    regions = await api('/regions');
    renderList();
  } catch {
    $('list-count').textContent = 'Service tidak menjawab; daftar ini mungkin sudah lama.';
  }
}

filterInput.addEventListener('input', renderList);

// -- add ------------------------------------------------------------------------------------

codeInput.addEventListener('input', () => {
  const upper = codeInput.value.toUpperCase();
  if (upper !== codeInput.value) codeInput.value = upper;
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  clearErrors();
  const code = codeInput.value.trim().toUpperCase();
  if (!CODE_PATTERN.test(code)) {
    setCodeError('Isi satu atau dua huruf, tanpa angka atau spasi.');
    codeInput.focus();
    return;
  }
  if (regions.some((region) => region.code === code)) {
    setCodeError(ERROR_MESSAGES.duplicate_code);
    codeInput.focus();
    return;
  }
  withBusy(addButton, async () => {
    try {
      await api('/regions', { method: 'POST', body: { code, name: nameInput.value.trim() }, admin: true });
      toast(`Kode ${code} ditambahkan dan langsung berlaku.`);
      form.reset();
      await refreshList();
      codeInput.focus();
    } catch (error) {
      if (admin.handleFailure(error)) {
        $('form-alert-text').textContent = messageFor(error);
        $('form-alert').hidden = false;
      } else if (error.code === 'duplicate_code' || error.code === 'special_code' || error.code === 'validation') {
        setCodeError(error.code === 'validation' ? 'Kode ditolak service. Isi satu atau dua huruf.' : messageFor(error));
        codeInput.focus();
      } else {
        $('form-alert-text').textContent = messageFor(error);
        $('form-alert').hidden = false;
      }
    }
  });
});

// -- remove and restore ---------------------------------------------------------------------

function askRemove(region) {
  pending = { kind: 'remove', code: region.code };
  $('confirm-title').textContent = `Hapus kode ${region.code}`;
  $('confirm-yes-label').textContent = 'Hapus kode';
  const seen = region.seen ? ` Kode ini sudah terbaca ${fmt0(region.seen)} kali di lokasi ini.` : '';
  $('confirm-text').textContent =
    `Setelah dihapus, plat berkode ${region.code} dianggap salah baca dan kodenya diganti dengan kode terdekat yang masih ada di daftar.${seen}`;
  confirmDialog.showModal();
}

restoreButton.addEventListener('click', () => {
  pending = { kind: 'restore' };
  $('confirm-title').textContent = 'Pulihkan kode bawaan';
  $('confirm-yes-label').textContent = 'Pulihkan';
  $('confirm-text').textContent =
    'Kode bawaan yang pernah dihapus dimasukkan lagi ke daftar. Kode yang Anda tambahkan sendiri tidak berubah.';
  confirmDialog.showModal();
});

$('confirm-yes').addEventListener('click', () => {
  const action = pending;
  withBusy($('confirm-yes'), async () => {
    try {
      if (action.kind === 'remove') {
        await api(`/regions/${encodeURIComponent(action.code)}`, { method: 'DELETE', admin: true });
        toast(`Kode ${action.code} dihapus dari daftar.`);
      } else {
        const result = await api('/regions/restore', { method: 'POST', admin: true });
        toast(result.added ? `${result.added} kode bawaan dipulihkan.` : 'Tidak ada kode bawaan yang hilang.');
      }
    } catch (error) {
      admin.handleFailure(error);
      toast(messageFor(error), 'error');
    }
    confirmDialog.close();
    await refreshList();
  });
});

// -- start ----------------------------------------------------------------------------------

admin.check();
setInterval(() => {
  if (!document.hidden) refreshList();
}, LIST_POLL_MS);
setInterval(refreshService, 10000);
