import './style.css';

/**
 * Standard MIDI to frequency calculation formula:
 * f = 440 * 2^((midiNote - 69) / 12)
 */
export function getNoteFrequency(midiNote: number): number {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
}

/**
 * Key definition for the placeholder interactive keyboard
 */
interface KeyDef {
  midi: number;
  note: string;
  isBlack: boolean;
  leftOffset?: number; // CSS left position for black keys
}

const PLACEHOLDER_OCTAVE: KeyDef[] = [
  { midi: 60, note: 'C4', isBlack: false },
  { midi: 61, note: 'C#4', isBlack: true, leftOffset: 34 },
  { midi: 62, note: 'D4', isBlack: false },
  { midi: 63, note: 'D#4', isBlack: true, leftOffset: 82 },
  { midi: 64, note: 'E4', isBlack: false },
  { midi: 65, note: 'F4', isBlack: false },
  { midi: 66, note: 'F#4', isBlack: true, leftOffset: 178 },
  { midi: 67, note: 'G4', isBlack: false },
  { midi: 68, note: 'G#4', isBlack: true, leftOffset: 226 },
  { midi: 69, note: 'A4', isBlack: false },
  { midi: 70, note: 'A#4', isBlack: true, leftOffset: 274 },
  { midi: 71, note: 'B4', isBlack: false },
];

export function initApp(): void {
  const app = document.querySelector<HTMLDivElement>('#app');
  if (!app) {
    console.error('Failed to find #app container');
    return;
  }

  app.innerHTML = `
    <header class="app-header">
      <h1 class="app-title"><span class="piano-icon" aria-hidden="true">🎹</span> pianosim</h1>
      <p class="app-subtitle">Browser Piano Physical Modeling Simulation</p>
    </header>

    <main class="main-content">
      <section class="controls-bar" aria-label="Piano Controls">
        <div class="status-indicator">
          <span class="status-dot" id="status-dot"></span>
          <span id="audio-status" class="status-text">Audio: Ready to start</span>
        </div>
        <button type="button" id="start-audio-btn" class="btn btn-primary">Start Audio Engine</button>
        <div class="pedal-badge" id="pedal-badge" aria-label="Sustain pedal indicator">
          Sustain: OFF <kbd>[Space]</kbd>
        </div>
      </section>

      <section class="keyboard-section" aria-label="Piano Keyboard">
        <div class="keyboard-bed">
          <div class="piano-keys" id="piano-keys" role="group" aria-label="Interactive Keyboard Prototype">
            ${PLACEHOLDER_OCTAVE.map((k) => {
              if (k.isBlack) {
                return `
                  <button type="button"
                    class="key key-black"
                    data-midi="${k.midi}"
                    data-note="${k.note}"
                    style="left: ${k.leftOffset}px;"
                    aria-label="${k.note}">
                    <span class="key-label">${k.note}</span>
                  </button>
                `;
              }
              return `
                <button type="button"
                  class="key key-white"
                  data-midi="${k.midi}"
                  data-note="${k.note}"
                  aria-label="${k.note}">
                  <span class="key-label">${k.note}</span>
                </button>
              `;
            }).join('')}
          </div>
        </div>
      </section>

      <section class="info-card">
        <h2>Milestone 1: Project Scaffolding Active</h2>
        <p>Vite + TypeScript + Vitest toolchain verified. Interactive virtual keyboard and Digital Waveguide physical modeling synthesizer will connect in Milestones 2 &amp; 3.</p>
        <p class="freq-preview" id="freq-preview">Select any key above to inspect fundamental frequency calculation.</p>
      </section>
    </main>

    <footer class="app-footer">
      <p>pianosim · Pure TypeScript &amp; Web Audio Physical Modeling Engine · <a href="https://github.com/likr-sandbox/pianosim" target="_blank" rel="noopener noreferrer">GitHub Repository</a></p>
    </footer>
  `;

  // Attach interactive listeners for the prototype keyboard
  const keysContainer = app.querySelector<HTMLElement>('#piano-keys');
  const freqPreview = app.querySelector<HTMLElement>('#freq-preview');
  const audioStatus = app.querySelector<HTMLElement>('#audio-status');
  const statusDot = app.querySelector<HTMLElement>('#status-dot');
  const startBtn = app.querySelector<HTMLButtonElement>('#start-audio-btn');

  if (startBtn && audioStatus && statusDot) {
    startBtn.addEventListener('click', () => {
      audioStatus.textContent = 'Audio: Engine Initialized';
      statusDot.classList.add('active');
      startBtn.textContent = 'Audio Active';
      startBtn.disabled = true;
    });
  }

  if (keysContainer && freqPreview) {
    const handleKeyActivation = (button: HTMLButtonElement, active: boolean) => {
      button.classList.toggle('active', active);
      if (active) {
        const midi = Number(button.dataset.midi);
        const note = button.dataset.note ?? '';
        const freq = getNoteFrequency(midi);
        freqPreview.textContent = `Active Key: ${note} (MIDI ${midi}) → Calculated Frequency: ${freq.toFixed(2)} Hz`;
      }
    };

    keysContainer.addEventListener('pointerdown', (e) => {
      const target = (e.target as HTMLElement).closest<HTMLButtonElement>('.key');
      if (target) {
        handleKeyActivation(target, true);
        target.setPointerCapture(e.pointerId);
      }
    });

    keysContainer.addEventListener('pointerup', (e) => {
      const target = (e.target as HTMLElement).closest<HTMLButtonElement>('.key');
      if (target) {
        handleKeyActivation(target, false);
      }
    });

    keysContainer.addEventListener('pointercancel', (e) => {
      const target = (e.target as HTMLElement).closest<HTMLButtonElement>('.key');
      if (target) {
        handleKeyActivation(target, false);
      }
    });
  }
}

// Auto-run when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
