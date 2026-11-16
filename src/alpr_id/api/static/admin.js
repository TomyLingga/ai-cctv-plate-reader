// The "Token admin" card that every management page carries. The markup is in the page;
// this module wires it and keeps track of whether changes are allowed from this browser.
import { ApiError, adminToken, api, el, icon, withBusy } from './common.js';

const $ = (id) => document.getElementById(id);

/** Show/hide buttons next to password-like inputs: <button data-reveal="input-id">. */
function wireReveals() {
  for (const button of document.querySelectorAll('[data-reveal]')) {
    const input = $(button.dataset.reveal);
    const label = button.getAttribute('aria-label');
    const paint = () => {
      const shown = input.type === 'text';
      button.replaceChildren(icon(shown ? 'eyeOff' : 'eye'));
      button.setAttribute('aria-pressed', String(shown));
      button.setAttribute('aria-label', shown ? label.replace('Tampilkan', 'Sembunyikan') : label);
    };
    button.addEventListener('click', () => {
      input.type = input.type === 'text' ? 'password' : 'text';
      paint();
    });
    paint();
  }
}

/**
 * `texts.ready` and `texts.locked` say what the token allows on this page. `onChange(state)`
 * runs after every change of `state` ({ configured, authorized }) and may be async.
 * Returns { state, check, handleFailure }.
 */
export function initAdminCard({ texts, onChange }) {
  const state = { configured: false, authorized: false };

  function paint(rejected = false) {
    const status = $('auth-status');
    const explain = $('auth-explain');
    $('auth-form').hidden = !state.configured || state.authorized;
    $('auth-done').hidden = !state.authorized;
    if (!state.configured) {
      status.dataset.state = 'off';
      $('auth-status-text').textContent = 'Belum diatur';
      explain.textContent =
        'Token admin belum diatur di PC service, jadi perubahan dari halaman ini belum bisa dilakukan. ' +
        'Isi ALPR_ADMIN_TOKEN (paling sedikit 8 karakter) di berkas .env, lalu jalankan ulang service.';
    } else if (state.authorized) {
      status.dataset.state = 'on';
      $('auth-status-text').textContent = 'Token diterima';
      explain.textContent = texts.ready;
    } else {
      status.dataset.state = 'idle';
      $('auth-status-text').textContent = 'Perlu token';
      explain.textContent = texts.locked;
    }
    const note = $('token-error');
    note.hidden = !rejected;
    if (rejected) {
      note.replaceChildren(icon('alert'), el('span', { text: 'Token ditolak. Periksa lagi nilainya.' }));
      $('token').setAttribute('aria-invalid', 'true');
    } else {
      $('token').removeAttribute('aria-invalid');
    }
  }

  async function check({ submitted = false } = {}) {
    let result;
    try {
      result = await api('/admin/check', { admin: true });
    } catch {
      $('auth-status').dataset.state = 'off';
      $('auth-status-text').textContent = 'Service tidak menjawab';
      return;
    }
    state.configured = result.configured;
    state.authorized = result.authorized;
    const hadToken = Boolean(adminToken.get());
    if (!state.authorized && hadToken) adminToken.clear();
    paint(!state.authorized && state.configured && (submitted || hadToken));
    await onChange(state);
  }

  /** A write that came back 401 or 403 means the token was changed on the service. */
  function handleFailure(error) {
    if (!(error instanceof ApiError) || (error.code !== 'bad_token' && error.code !== 'admin_disabled')) return false;
    adminToken.clear();
    state.authorized = false;
    state.configured = error.code !== 'admin_disabled';
    paint(state.configured);
    onChange(state);
    return true;
  }

  $('auth-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const value = $('token').value.trim();
    if (!value) {
      $('token-error').replaceChildren(icon('alert'), el('span', { text: 'Token belum diisi.' }));
      $('token-error').hidden = false;
      $('token').setAttribute('aria-invalid', 'true');
      $('token').focus();
      return;
    }
    adminToken.set(value, $('token-remember').checked);
    withBusy($('auth-submit'), async () => {
      await check({ submitted: true });
      if (state.authorized) $('token').value = '';
      else $('token').focus();
    });
  });

  $('auth-forget').addEventListener('click', () => {
    adminToken.clear();
    state.authorized = false;
    paint();
    onChange(state);
  });

  wireReveals();
  $('token-remember').checked = adminToken.remembered();
  return { state, check, handleFailure };
}
