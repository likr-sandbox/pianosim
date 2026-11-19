/** Performance & Physics Quality panel managing Full/Balanced/Realtime presets and 18 knobs. */

import {
  clonePerfConfig,
  DEFAULT_LIVE_PERF_PRESET,
  PERF_KNOB_SPECS,
  PERF_PRESET_BALANCED,
  PERF_PRESET_CONFIGS,
  PERF_PRESET_FULL,
  PERF_PRESET_LABELS,
  PERF_PRESET_REALTIME,
  PERF_PRESET_SLUGS,
  perfConfigsEqual,
  type PerfKnobKey,
  type PerfKnobSpec,
  type PerfPresetId,
  type PerfProfileConfig,
} from '../audio/abi';

const PRESET_ORDER: ReadonlyArray<PerfPresetId> = [
  PERF_PRESET_FULL,
  PERF_PRESET_BALANCED,
  PERF_PRESET_REALTIME,
];

export interface PerfPanelCallbacks {
  readonly onSelectPreset: (presetId: PerfPresetId) => void;
  readonly onKnobChange: <K extends PerfKnobKey>(key: K, value: PerfProfileConfig[K]) => void;
}

export class PerfPanel {
  private readonly container: HTMLElement;
  private readonly callbacks: PerfPanelCallbacks;
  private currentPreset: PerfPresetId;
  private currentConfig: PerfProfileConfig;
  private isCustom = false;

  private selectEl!: HTMLSelectElement;
  private resetBtn!: HTMLButtonElement;
  private customBadgeEl!: HTMLElement;
  private readonly presetBtns = new Map<PerfPresetId, HTMLButtonElement>();
  private readonly knobInputs = new Map<PerfKnobKey, HTMLInputElement | HTMLSelectElement>();
  private readonly knobVals = new Map<PerfKnobKey, HTMLElement>();
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(
    container: HTMLElement,
    callbacks: PerfPanelCallbacks,
    initialPreset: PerfPresetId = DEFAULT_LIVE_PERF_PRESET,
    initialConfig?: PerfProfileConfig,
  ) {
    this.container = container;
    this.callbacks = callbacks;
    this.currentPreset = initialPreset;
    this.currentConfig = clonePerfConfig(initialConfig ?? PERF_PRESET_CONFIGS[initialPreset]);
    this.isCustom = !perfConfigsEqual(this.currentConfig, PERF_PRESET_CONFIGS[initialPreset]);
    this.render();
    this.bindEvents();
    this.updateDom();
  }

  setState(presetId: PerfPresetId, config: PerfProfileConfig): void {
    this.currentPreset = presetId;
    this.currentConfig = clonePerfConfig(config);
    this.isCustom = !perfConfigsEqual(this.currentConfig, PERF_PRESET_CONFIGS[presetId]);
    this.updateDom();
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) off();
    this.cleanupHandlers.length = 0;
    this.presetBtns.clear();
    this.knobInputs.clear();
    this.knobVals.clear();
    this.container.innerHTML = '';
  }

  private renderKnobControl(spec: PerfKnobSpec): string {
    const val = this.currentConfig[spec.key];
    const testId = `perf-knob-${spec.snakeCase}`;
    if (spec.kind === 'bool') {
      return `
        <label class="perf-knob-row perf-knob-bool">
          <span class="control-label">${spec.label} <small class="perf-stage">${spec.stage}</small></span>
          <input type="checkbox" data-testid="${testId}" data-knob="${spec.key}" ${val ? 'checked' : ''} />
        </label>
      `;
    }
    if (spec.kind === 'select') {
      const opts = (spec.options ?? [])
        .map((o) => `<option value="${o}" ${Number(val) === o ? 'selected' : ''}>${o}</option>`)
        .join('');
      return `
        <label class="perf-knob-row">
          <span class="control-label">${spec.label} <small class="perf-stage">${spec.stage}</small></span>
          <select class="select-perf-knob" data-testid="${testId}" data-knob="${spec.key}">${opts}</select>
        </label>
      `;
    }
    return `
      <label class="perf-knob-row">
        <span class="control-label">${spec.label} <small class="perf-stage">${spec.stage}</small></span>
        <div class="perf-range-wrap">
          <input
            type="range"
            data-testid="${testId}"
            data-knob="${spec.key}"
            min="${spec.min}"
            max="${spec.max}"
            step="${spec.step}"
            value="${val}"
          />
          <span class="val-mono perf-knob-val" data-knob-val="${spec.key}">${val}</span>
        </div>
      </label>
    `;
  }

  private render(): void {
    const btnsHtml = PRESET_ORDER.map((id) => {
      const slug = PERF_PRESET_SLUGS[id];
      const active = !this.isCustom && id === this.currentPreset;
      return `<button
        type="button"
        class="btn-segmented${active ? ' active' : ''}"
        data-testid="perf-preset-${slug}"
        data-perf-preset="${id}"
        role="radio"
        aria-checked="${active ? 'true' : 'false'}"
      >${PERF_PRESET_LABELS[id]}</button>`;
    }).join('');

    const optsHtml = PRESET_ORDER.map((id) => {
      const slug = PERF_PRESET_SLUGS[id];
      return `<option value="${slug}">${PERF_PRESET_LABELS[id]}</option>`;
    }).join('');

    const knobsHtml = PERF_KNOB_SPECS.map((s) => this.renderKnobControl(s)).join('');

    this.container.innerHTML = `
      <section class="perf-panel" aria-label="Performance Quality Profile">
        <div class="perf-header">
          <div class="preset-label-group">
            <span class="preset-title">RT Performance Profile</span>
            <span data-testid="perf-custom-badge" class="badge-custom" hidden>Custom</span>
          </div>
          <div class="perf-header-actions">
            <select data-testid="perf-preset-select" class="select-preset" aria-label="Performance Preset">
              ${optsHtml}
            </select>
            <button type="button" data-testid="perf-reset" class="btn btn-secondary">Reset Preset</button>
          </div>
        </div>
        <div class="perf-controls-row">
          <div class="segmented-group perf-segmented" role="radiogroup" aria-label="Performance Presets">
            ${btnsHtml}
          </div>
        </div>
        <details class="perf-advanced" data-testid="perf-advanced-toggle">
          <summary class="perf-advanced-summary">Advanced 18-Knob Simplification Parameters</summary>
          <div class="perf-knob-grid">${knobsHtml}</div>
        </details>
      </section>
    `;

    this.selectEl = this.container.querySelector('[data-testid="perf-preset-select"]')!;
    this.resetBtn = this.container.querySelector('[data-testid="perf-reset"]')!;
    this.customBadgeEl = this.container.querySelector('[data-testid="perf-custom-badge"]')!;
    for (const id of PRESET_ORDER) {
      const btn = this.container.querySelector<HTMLButtonElement>(`[data-perf-preset="${id}"]`);
      if (btn) this.presetBtns.set(id, btn);
    }
    for (const spec of PERF_KNOB_SPECS) {
      const input = this.container.querySelector<HTMLInputElement | HTMLSelectElement>(
        `[data-knob="${spec.key}"]`,
      );
      if (input) this.knobInputs.set(spec.key, input);
      const valEl = this.container.querySelector<HTMLElement>(`[data-knob-val="${spec.key}"]`);
      if (valEl) this.knobVals.set(spec.key, valEl);
    }
  }

  private bindEvents(): void {
    for (const [id, btn] of this.presetBtns) {
      const onClick = () => {
        this.setState(id, PERF_PRESET_CONFIGS[id]);
        this.callbacks.onSelectPreset(id);
      };
      btn.addEventListener('click', onClick);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', onClick));
    }

    const onSelect = () => {
      const v = this.selectEl.value;
      const found = PRESET_ORDER.find((id) => PERF_PRESET_SLUGS[id] === v);
      if (found !== undefined) {
        this.setState(found, PERF_PRESET_CONFIGS[found]);
        this.callbacks.onSelectPreset(found);
      }
    };
    this.selectEl.addEventListener('change', onSelect);
    this.cleanupHandlers.push(() => this.selectEl.removeEventListener('change', onSelect));

    const onReset = () => {
      this.setState(this.currentPreset, PERF_PRESET_CONFIGS[this.currentPreset]);
      this.callbacks.onSelectPreset(this.currentPreset);
    };
    this.resetBtn.addEventListener('click', onReset);
    this.cleanupHandlers.push(() => this.resetBtn.removeEventListener('click', onReset));

    for (const spec of PERF_KNOB_SPECS) {
      const el = this.knobInputs.get(spec.key);
      if (!el) continue;
      const eventName = spec.kind === 'range' ? 'input' : 'change';
      const onChange = () => {
        let nextVal: boolean | number;
        if (spec.kind === 'bool') {
          nextVal = (el as HTMLInputElement).checked;
        } else {
          nextVal = Number(el.value);
        }
        this.currentConfig = { ...this.currentConfig, [spec.key]: nextVal };
        this.isCustom = !perfConfigsEqual(
          this.currentConfig,
          PERF_PRESET_CONFIGS[this.currentPreset],
        );
        this.updateDom();
        this.callbacks.onKnobChange(spec.key, nextVal as PerfProfileConfig[typeof spec.key]);
      };
      el.addEventListener(eventName, onChange);
      this.cleanupHandlers.push(() => el.removeEventListener(eventName, onChange));
    }
  }

  private updateDom(): void {
    this.customBadgeEl.hidden = !this.isCustom;
    for (const [id, btn] of this.presetBtns) {
      const active = !this.isCustom && id === this.currentPreset;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    }
    if (this.isCustom) {
      let customOpt = this.selectEl.querySelector<HTMLOptionElement>('option[value="custom"]');
      if (!customOpt) {
        customOpt = document.createElement('option');
        customOpt.value = 'custom';
        customOpt.textContent = 'Custom';
        this.selectEl.appendChild(customOpt);
      }
      this.selectEl.value = 'custom';
    } else {
      this.selectEl.querySelector('option[value="custom"]')?.remove();
      this.selectEl.value = PERF_PRESET_SLUGS[this.currentPreset];
    }
    for (const spec of PERF_KNOB_SPECS) {
      const el = this.knobInputs.get(spec.key);
      const val = this.currentConfig[spec.key];
      if (el) {
        if (spec.kind === 'bool') (el as HTMLInputElement).checked = Boolean(val);
        else el.value = String(val);
      }
      const valEl = this.knobVals.get(spec.key);
      if (valEl) valEl.textContent = String(val);
    }
  }
}
