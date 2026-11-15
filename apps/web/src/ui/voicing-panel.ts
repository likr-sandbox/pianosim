/** Voicing parameter control panel for hammer hardness, unison detuning, stretch tuning, and string scaling. */

export interface VoicingState {
  hammerHardness: number;
  unisonDetune: number;
  stretchTuning: number;
  stringLengthScale: number;
}

export type VoicingChangeCallback = (state: VoicingState) => void;

export class VoicingPanel {
  private readonly container: HTMLElement;
  private readonly onChange: VoicingChangeCallback;

  private hammerSlider!: HTMLInputElement;
  private unisonSlider!: HTMLInputElement;
  private stretchSlider!: HTMLInputElement;
  private lengthSlider!: HTMLInputElement;

  private hammerValEl!: HTMLElement;
  private unisonValEl!: HTMLElement;
  private stretchValEl!: HTMLElement;
  private lengthValEl!: HTMLElement;

  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, onChange: VoicingChangeCallback) {
    this.container = container;
    this.onChange = onChange;
    this.render();
    this.bindEvents();
  }

  getState(): VoicingState {
    return {
      hammerHardness: parseFloat(this.hammerSlider.value),
      unisonDetune: parseFloat(this.unisonSlider.value),
      stretchTuning: parseFloat(this.stretchSlider.value),
      stringLengthScale: parseFloat(this.lengthSlider.value),
    };
  }

  setState(state: Partial<VoicingState>): void {
    if (state.hammerHardness !== undefined) {
      this.hammerSlider.value = String(state.hammerHardness);
      this.hammerValEl.textContent = state.hammerHardness.toFixed(2);
    }
    if (state.unisonDetune !== undefined) {
      this.unisonSlider.value = String(state.unisonDetune);
      this.unisonValEl.textContent = state.unisonDetune.toFixed(2);
    }
    if (state.stretchTuning !== undefined) {
      this.stretchSlider.value = String(state.stretchTuning);
      this.stretchValEl.textContent = state.stretchTuning.toFixed(2);
    }
    if (state.stringLengthScale !== undefined) {
      this.lengthSlider.value = String(state.stringLengthScale);
      this.lengthValEl.textContent = state.stringLengthScale.toFixed(2);
    }
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.container.innerHTML = '';
  }

  private render(): void {
    this.container.innerHTML = `
      <section class="voicing-panel" aria-label="Physical Voicing Parameters">
        <h3 class="panel-title">Voicing & Scale</h3>
        <div class="panel-grid">
          <label class="control-row" for="slider-hammer-hardness">
            <span class="control-label">Hammer Hardness</span>
            <input
              id="slider-hammer-hardness"
              data-testid="voicing-hammer"
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value="0.5"
            />
            <span id="val-voicing-hammer" class="val-mono">0.50</span>
          </label>

          <label class="control-row" for="slider-unison-detune">
            <span class="control-label">Unison Detune</span>
            <input
              id="slider-unison-detune"
              data-testid="voicing-unison"
              type="range"
              min="0.0"
              max="4.0"
              step="0.05"
              value="1.0"
            />
            <span id="val-voicing-unison" class="val-mono">1.00</span>
          </label>

          <label class="control-row" for="slider-stretch">
            <span class="control-label">Stretch Tuning</span>
            <input
              id="slider-stretch"
              data-testid="voicing-stretch"
              type="range"
              min="0.0"
              max="2.0"
              step="0.05"
              value="1.0"
            />
            <span id="val-voicing-stretch" class="val-mono">1.00</span>
          </label>

          <label class="control-row" for="slider-string-scale">
            <span class="control-label">String Scale</span>
            <input
              id="slider-string-scale"
              data-testid="voicing-length"
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value="1.0"
            />
            <span id="val-voicing-length" class="val-mono">1.00</span>
          </label>
        </div>
      </section>
    `;

    this.hammerSlider = this.container.querySelector('#slider-hammer-hardness')!;
    this.unisonSlider = this.container.querySelector('#slider-unison-detune')!;
    this.stretchSlider = this.container.querySelector('#slider-stretch')!;
    this.lengthSlider = this.container.querySelector('#slider-string-scale')!;

    this.hammerValEl = this.container.querySelector('#val-voicing-hammer')!;
    this.unisonValEl = this.container.querySelector('#val-voicing-unison')!;
    this.stretchValEl = this.container.querySelector('#val-voicing-stretch')!;
    this.lengthValEl = this.container.querySelector('#val-voicing-length')!;
  }

  private bindEvents(): void {
    const onInput = () => {
      this.hammerValEl.textContent = parseFloat(this.hammerSlider.value).toFixed(2);
      this.unisonValEl.textContent = parseFloat(this.unisonSlider.value).toFixed(2);
      this.stretchValEl.textContent = parseFloat(this.stretchSlider.value).toFixed(2);
      this.lengthValEl.textContent = parseFloat(this.lengthSlider.value).toFixed(2);
      this.onChange(this.getState());
    };

    this.hammerSlider.addEventListener('input', onInput);
    this.unisonSlider.addEventListener('input', onInput);
    this.stretchSlider.addEventListener('input', onInput);
    this.lengthSlider.addEventListener('input', onInput);

    this.cleanupHandlers.push(() => {
      this.hammerSlider.removeEventListener('input', onInput);
      this.unisonSlider.removeEventListener('input', onInput);
      this.stretchSlider.removeEventListener('input', onInput);
      this.lengthSlider.removeEventListener('input', onInput);
    });
  }
}
