/** Voicing parameter control panel for hammer hardness, unison detuning, stretch tuning, string scaling, and concert pitch (A4). */

import {
  A4_FREQ_DEFAULT,
  A4_FREQ_MAX,
  A4_FREQ_MIN,
  A4_FREQ_STEP,
} from '../audio/abi';

export const A4_QUICK_PRESETS = [415, 432, 440, 442] as const;

export interface VoicingState {
  hammerHardness: number;
  unisonDetune: number;
  stretchTuning: number;
  stringLengthScale: number;
  a4FreqHz: number;
}

export type VoicingChangeCallback = (state: VoicingState) => void;

function sanitizeA4Hz(raw: number): number {
  if (!Number.isFinite(raw)) {
    return A4_FREQ_DEFAULT;
  }
  const clamped = Math.max(A4_FREQ_MIN, Math.min(A4_FREQ_MAX, raw));
  return Math.round(clamped * 10) / 10;
}

export class VoicingPanel {
  private readonly container: HTMLElement;
  private readonly onChange: VoicingChangeCallback;

  private hammerSlider!: HTMLInputElement;
  private unisonSlider!: HTMLInputElement;
  private stretchSlider!: HTMLInputElement;
  private lengthSlider!: HTMLInputElement;
  private a4Slider!: HTMLInputElement;
  private a4NumberInput!: HTMLInputElement;
  private a4ChipBtns: HTMLButtonElement[] = [];

  private hammerValEl!: HTMLElement;
  private unisonValEl!: HTMLElement;
  private stretchValEl!: HTMLElement;
  private lengthValEl!: HTMLElement;

  private a4FreqHz = A4_FREQ_DEFAULT;
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
      a4FreqHz: this.a4FreqHz,
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
    if (state.a4FreqHz !== undefined) {
      this.applyA4Display(sanitizeA4Hz(state.a4FreqHz));
    }
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.container.innerHTML = '';
  }

  private applyA4Display(hz: number, updateNumberField = true): void {
    this.a4FreqHz = hz;
    const formatted = hz.toFixed(1);
    this.a4Slider.value = formatted;
    if (updateNumberField) {
      this.a4NumberInput.value = formatted;
    }
    for (const btn of this.a4ChipBtns) {
      const chipHz = Number(btn.dataset.a4);
      const isActive = Math.abs(hz - chipHz) < 0.05;
      btn.classList.toggle('active', isActive);
      btn.setAttribute('aria-pressed', String(isActive));
    }
  }

  private render(): void {
    const chipsHtml = A4_QUICK_PRESETS.map((hz) => {
      const active = hz === A4_FREQ_DEFAULT;
      return `<button
        type="button"
        class="btn-a4-chip${active ? ' active' : ''}"
        data-testid="voicing-a4-chip-${hz}"
        data-a4="${hz}"
        aria-pressed="${active}"
      >${hz} Hz</button>`;
    }).join('');

    this.container.innerHTML = `
      <section class="voicing-panel" aria-label="Physical Voicing Parameters">
        <h3 class="panel-title">Voicing & Scale</h3>
        <div class="panel-grid">
          <div class="control-row control-row-a4">
            <div class="a4-header-row">
              <label class="control-label" for="slider-a4-freq">Concert Pitch (A4)</label>
              <div class="a4-input-group">
                <input
                  id="input-a4-freq"
                  data-testid="voicing-a4-number"
                  class="a4-number-input val-mono"
                  type="number"
                  min="${A4_FREQ_MIN.toFixed(1)}"
                  max="${A4_FREQ_MAX.toFixed(1)}"
                  step="${A4_FREQ_STEP}"
                  value="${A4_FREQ_DEFAULT.toFixed(1)}"
                  aria-label="Concert Pitch A4 (Hz)"
                />
                <span class="a4-unit">Hz</span>
              </div>
            </div>
            <input
              id="slider-a4-freq"
              data-testid="voicing-a4"
              type="range"
              min="${A4_FREQ_MIN.toFixed(1)}"
              max="${A4_FREQ_MAX.toFixed(1)}"
              step="${A4_FREQ_STEP}"
              value="${A4_FREQ_DEFAULT.toFixed(1)}"
            />
            <div class="a4-chips-row" role="group" aria-label="Quick Concert Pitch Presets">
              ${chipsHtml}
            </div>
          </div>

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
    this.a4Slider = this.container.querySelector('#slider-a4-freq')!;
    this.a4NumberInput = this.container.querySelector('#input-a4-freq')!;
    this.a4ChipBtns = Array.from(this.container.querySelectorAll<HTMLButtonElement>('.btn-a4-chip'));

    this.hammerValEl = this.container.querySelector('#val-voicing-hammer')!;
    this.unisonValEl = this.container.querySelector('#val-voicing-unison')!;
    this.stretchValEl = this.container.querySelector('#val-voicing-stretch')!;
    this.lengthValEl = this.container.querySelector('#val-voicing-length')!;
  }

  private bindEvents(): void {
    const onSliderInput = () => {
      this.hammerValEl.textContent = parseFloat(this.hammerSlider.value).toFixed(2);
      this.unisonValEl.textContent = parseFloat(this.unisonSlider.value).toFixed(2);
      this.stretchValEl.textContent = parseFloat(this.stretchSlider.value).toFixed(2);
      this.lengthValEl.textContent = parseFloat(this.lengthSlider.value).toFixed(2);
      this.onChange(this.getState());
    };

    const onA4SliderInput = () => {
      const hz = sanitizeA4Hz(parseFloat(this.a4Slider.value));
      this.applyA4Display(hz, true);
      this.onChange(this.getState());
    };

    const onA4NumberInput = () => {
      const parsed = parseFloat(this.a4NumberInput.value);
      if (Number.isFinite(parsed) && parsed >= A4_FREQ_MIN && parsed <= A4_FREQ_MAX) {
        const hz = sanitizeA4Hz(parsed);
        if (Math.abs(hz - this.a4FreqHz) >= 1e-4) {
          this.applyA4Display(hz, false);
          this.onChange(this.getState());
        }
      }
    };

    const onA4NumberCommit = () => {
      const parsed = parseFloat(this.a4NumberInput.value);
      const hz = Number.isFinite(parsed) ? sanitizeA4Hz(parsed) : this.a4FreqHz;
      const changed = Math.abs(hz - this.a4FreqHz) >= 1e-4;
      this.applyA4Display(hz, true);
      if (changed) {
        this.onChange(this.getState());
      }
    };

    this.hammerSlider.addEventListener('input', onSliderInput);
    this.unisonSlider.addEventListener('input', onSliderInput);
    this.stretchSlider.addEventListener('input', onSliderInput);
    this.lengthSlider.addEventListener('input', onSliderInput);
    this.a4Slider.addEventListener('input', onA4SliderInput);
    this.a4NumberInput.addEventListener('input', onA4NumberInput);
    this.a4NumberInput.addEventListener('change', onA4NumberCommit);
    this.a4NumberInput.addEventListener('blur', onA4NumberCommit);

    for (const btn of this.a4ChipBtns) {
      const onChipClick = () => {
        const hz = sanitizeA4Hz(Number(btn.dataset.a4));
        this.applyA4Display(hz, true);
        this.onChange(this.getState());
      };
      btn.addEventListener('click', onChipClick);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', onChipClick));
    }

    this.cleanupHandlers.push(() => {
      this.hammerSlider.removeEventListener('input', onSliderInput);
      this.unisonSlider.removeEventListener('input', onSliderInput);
      this.stretchSlider.removeEventListener('input', onSliderInput);
      this.lengthSlider.removeEventListener('input', onSliderInput);
      this.a4Slider.removeEventListener('input', onA4SliderInput);
      this.a4NumberInput.removeEventListener('input', onA4NumberInput);
      this.a4NumberInput.removeEventListener('change', onA4NumberCommit);
      this.a4NumberInput.removeEventListener('blur', onA4NumberCommit);
    });
  }
}
