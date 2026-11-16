// Runs before the page is drawn, so the stored theme is there from the first paint.
(function () {
  var stored = null;
  try {
    stored = localStorage.getItem('alpr.theme');
  } catch (err) {
    // Storage can be blocked; the system preference is used then.
  }
  var light = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  document.documentElement.dataset.theme = stored || (light ? 'light' : 'dark');
})();
