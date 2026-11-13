/** Application shell: builds the DOM and wires keyboard, controllers and audio together. */

import { AudioController, type AudioState } from './AudioController';
import { KeyboardController } from './KeyboardController';
import { PianoController } from './PianoController';
import { PianoKeyboard } from './PianoKeyboard';
import { clampVelocity, midiToFrequency, midiToNoteName } from './noteUtils';

const TEMPLATE = `
  <header class="app-header">
    <h1 class="app-title"><span class="piano-icon" aria-hidden="true">🎹</span> pianosim</h1>
    <p class="app-subtitle">Browser piano — real-time physical modeling synthesis</p>
  </header>

  <main class="main-content">
    <section class="controls-bar" aria-label="Piano controls">
      <div class="status-indicator" role="status">
        <span class="status-dot" id="status-dot"></span>
        <span id="audio-status" class="status-text">Audio: Ready to start</span>
      </div>
      <button type="button" id="start-audio-btn" class="btn btn-primary">Start Audio Engine</button>

      <label class="control velocity-control" for="velocity-slider">
        <span class="control-label">Velocity <output id="velocity-value">80</output></span>
        <input type="range" id="velocity-slider" min="1" max="127" step="1" value="80" />
      </label>

      <div class="control octave-control" aria-label="Computer keyboard octave">
        <button type="button" id="octave-down" class="btn btn-small" aria-label="Octave down" title="Octave down (Z)">−</button>
        <span class="control-label">Octave <output id="octave-display">C4</output></span>
        <button type="button" id="octave-up" class="btn btn-small" aria-label="Octave up" title="Octave up (X)">+</button>
      </div>

      <button type="button" id="pedal-badge" class="pedal-badge" aria-pressed="false" aria-label="Sustain pedal (hold, or hold Space)">
        <span id="pedal-text">Sustain: OFF</span> <kbd>Space</kbd>
      </button>
    </section>

    <section class="keyboard-section" aria-label="Piano keyboard">
      <div class="keyboard-scroll" id="keyboard-scroll">
        <div class="keyboard-bed">
          <div id="piano-keys" role="group" aria-label="88-key piano keyboard"></div>
        </div>
      </div>
      <input type="range" id="scroll-range" class="scroll-range" min="0" max="1000" value="500" aria-label="Scroll keyboard" />
    </section>

    <section class="info-card">
      <p class="freq-preview" id="freq-preview" aria-live="polite">Press a key to see its note and fundamental frequency.</p>
      <ul class="help-list">
        <li><kbd>A</kbd>–<kbd>;</kbd> white keys, <kbd>W E T Y U O P</kbd> black keys (A = C4)</li>
        <li><kbd>Z</kbd> / <kbd>X</kbd> octave down / up</li>
        <li>Hold <kbd>Space</kbd> for the sustain pedal</li>
        <li>Mouse / touch: click, chord with multiple fingers, or drag across keys for a glissando</li>
      </ul>
    </section>
  </main>

  <footer class="app-footer">
    <p>pianosim · TypeScript digital-waveguide piano model on Web Audio · <a href="https://github.com/likr-sandbox/pianosim" target="_blank" rel="noopener noreferrer">GitHub</a></p>
  </footer>
`;

function must<T extends Element>(root: ParentNode, sel: string): T {
  const el = root.querySelector<T>(sel);
  if (!el) throw new Error(`Missing element ${sel}`);
  return el;
}

export function initApp(): void {
  const app = document.querySelector<HTMLDivElement>('#app');
  if (!app) {
    console.error('Failed to find #app container');
    return;
  }
  app.innerHTML = TEMPLATE;

  const statusDot = must<HTMLElement>(app, '#status-dot');
  const audioStatus = must<HTMLElement>(app, '#audio-status');
  const startBtn = must<HTMLButtonElement>(app, '#start-audio-btn');
  const velSlider = must<HTMLInputElement>(app, '#velocity-slider');
  const velValue = must<HTMLElement>(app, '#velocity-value');
  const octaveDisplay = must<HTMLElement>(app, '#octave-display');
  const octDown = must<HTMLButtonElement>(app, '#octave-down');
  const octUp = must<HTMLButtonElement>(app, '#octave-up');
  const pedalBadge = must<HTMLButtonElement>(app, '#pedal-badge');
  const pedalText = must<HTMLElement>(app, '#pedal-text');
  const freqPreview = must<HTMLElement>(app, '#freq-preview');
  const scroller = must<HTMLElement>(app, '#keyboard-scroll');
  const scrollRange = must<HTMLInputElement>(app, '#scroll-range');

  // ---- audio ----
  const audio = new AudioController();
  const renderAudioState = (s: AudioState, detail?: string) => {
    const on = s === 'starting' || s === 'running';
    statusDot.classList.toggle('active', on);
    statusDot.classList.toggle('error', s === 'error');
    startBtn.disabled = on;
    startBtn.textContent = s === 'running' ? 'Audio Active' : s === 'starting' ? 'Starting…' : 'Start Audio Engine';
    audioStatus.textContent =
      s === 'idle'
        ? 'Audio: Ready to start'
        : s === 'starting'
          ? 'Audio: Starting…'
          : s === 'running'
            ? 'Audio: Engine Initialized'
            : `Audio: Failed to start${detail ? ` (${detail})` : ''}`;
  };
  audio.onStateChange(renderAudioState);
  startBtn.addEventListener('click', () => void audio.start());

  // ---- keyboard + controllers ----
  const getVelocity = () => clampVelocity(Number(velSlider.value));
  let controller: PianoController;
  const keyboard = new PianoKeyboard({
    container: must<HTMLElement>(app, '#piano-keys'),
    onNoteOn: (m, v, src) => controller.noteOn(m, v, src),
    onNoteOff: (m, src) => controller.noteOff(m, src),
    getVelocity,
  });
  controller = new PianoController(audio, keyboard);

  controller.onStrike = ({ midi, velocity }) => {
    const f = midiToFrequency(midi);
    freqPreview.textContent = `Active Key: ${midiToNoteName(midi)} (MIDI ${midi}) → Calculated Frequency: ${f.toFixed(2)} Hz · velocity ${velocity}`;
  };

  let uiPedal = false;
  let keyPedal = false;
  const syncPedal = () => controller.setPedal(uiPedal || keyPedal);
  controller.onPedalChange = (down) => {
    pedalBadge.classList.toggle('active', down);
    pedalBadge.setAttribute('aria-pressed', String(down));
    pedalText.textContent = down ? 'Sustain: ON' : 'Sustain: OFF';
  };

  const kbd = new KeyboardController({
    onNoteOn: (m, src) => controller.noteOn(m, getVelocity(), src),
    onNoteOff: (m, src) => controller.noteOff(m, src),
    onPedal: (down) => {
      keyPedal = down;
      syncPedal();
    },
    onOctaveChange: () => updateOctave(),
  });
  function updateOctave(): void {
    const shift = kbd.getOctaveShift();
    octaveDisplay.textContent = midiToNoteName(60 + shift * 12);
    keyboard.setHints(kbd.getHints());
  }
  updateOctave();
  octDown.addEventListener('click', () => kbd.setOctaveShift(kbd.getOctaveShift() - 1));
  octUp.addEventListener('click', () => kbd.setOctaveShift(kbd.getOctaveShift() + 1));

  // On-screen pedal: hold to sustain.
  pedalBadge.addEventListener('mousedown', (e) => e.preventDefault());
  pedalBadge.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    uiPedal = true;
    syncPedal();
  });
  const pedalUp = () => {
    if (!uiPedal) return;
    uiPedal = false;
    syncPedal();
  };
  window.addEventListener('pointerup', pedalUp);
  window.addEventListener('pointercancel', pedalUp);

  velSlider.addEventListener('input', () => {
    velValue.textContent = String(getVelocity());
  });

  // First gesture anywhere unlocks audio (autoplay policy), without waiting for a note.
  const unlock = () => void audio.start();
  window.addEventListener('pointerdown', unlock, { once: true, capture: true });
  window.addEventListener('keydown', unlock, { once: true, capture: true });

  // ---- horizontal scrolling on narrow screens ----
  const syncRange = () => {
    const max = scroller.scrollWidth - scroller.clientWidth;
    scrollRange.hidden = max <= 1;
    if (max > 1) scrollRange.value = String(Math.round((scroller.scrollLeft / max) * 1000));
  };
  scrollRange.addEventListener('input', () => {
    const max = scroller.scrollWidth - scroller.clientWidth;
    scroller.scrollLeft = (Number(scrollRange.value) / 1000) * max;
  });
  scroller.addEventListener('scroll', syncRange, { passive: true });
  window.addEventListener('resize', syncRange);
  // Start around middle C when the keyboard overflows the viewport.
  const c4 = keyboard.getKeyElement(60);
  if (c4 && scroller.scrollWidth > scroller.clientWidth) {
    scroller.scrollLeft = c4.offsetLeft - scroller.clientWidth / 2;
  }
  syncRange();

  window.addEventListener('blur', () => {
    keyPedal = false;
    uiPedal = false;
    syncPedal();
  });
}
