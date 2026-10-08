document.documentElement.classList.add('js-enabled');

// Start at the intro rather than restoring the previous scroll position.
if ('scrollRestoration' in history) {
  history.scrollRestoration = 'manual';
}
window.scrollTo({ top: 0, left: 0, behavior: 'instant' });

document.addEventListener('DOMContentLoaded', function () {
  document.body.classList.add('landing-active');
  window.scrollTo({ top: 0, left: 0, behavior: 'instant' });

  document.querySelectorAll('.is-placeholder').forEach(function (link) {
    link.addEventListener('click', function (event) {
      event.preventDefault();
    });
  });

  var hasLeftLanding = false;
  var touchStartY = null;

  window.addEventListener('pageshow', function () {
    if (!hasLeftLanding) {
      window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
    }
  });

  function leaveLanding() {
    if (hasLeftLanding) return;
    hasLeftLanding = true;
    document.body.classList.remove('landing-active');
    document.body.classList.add('landing-transition');

    window.setTimeout(function () {
      document.body.classList.remove('landing-transition');
      document.body.classList.add('landing-complete');
    }, 820);
  }

  window.addEventListener('wheel', function (event) {
    if (!hasLeftLanding && event.deltaY > 0) {
      event.preventDefault();
      leaveLanding();
    }
  }, { passive: false });

  window.addEventListener('touchstart', function (event) {
    touchStartY = event.touches[0] ? event.touches[0].clientY : null;
  }, { passive: true });

  window.addEventListener('touchend', function (event) {
    var touch = event.changedTouches[0];
    if (!hasLeftLanding && touchStartY !== null && touch && touchStartY - touch.clientY > 20) {
      leaveLanding();
    }
    touchStartY = null;
  }, { passive: true });

  window.addEventListener('keydown', function (event) {
    if (!hasLeftLanding && ['ArrowDown', 'PageDown', ' ', 'Spacebar'].includes(event.key)) {
      event.preventDefault();
      leaveLanding();
    }
  });
});
