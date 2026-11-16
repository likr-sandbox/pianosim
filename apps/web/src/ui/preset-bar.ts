/** UI component for selecting and managing physical parameter presets. */

import {
  DEFAULT_PRESET_ID,
  getPreset,
  PIANO_PRESETS,
  type ParameterPreset,
} from '../presets/parameter-presets';

export interface PresetBarCallbacks {
  readonly onSelectPreset: (preset: ParameterPreset) => void;
}

export class PresetBar {
  private container: HTMLElement | null = null;
  private readonly callbacks: PresetBarCallbacks;
  private currentPresetId: string = DEFAULT_PRESET_ID;
  private isCustom = false;

  private selectEl!: HTMLSelectElement;
  private descEl!: HTMLElement;
  private customBadgeEl!: HTMLElement;
  private readonly buttonMap = new Map<string, HTMLButtonElement>();
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(
    containerOrCallbacks: HTMLElement | PresetBarCallbacks,
    callbacks?: PresetBarCallbacks,
  ) {
    if (containerOrCallbacks instanceof HTMLElement) {
      this.container = containerOrCallbacks;
      this.callbacks = callbacks ?? { onSelectPreset: () => {} };
      this.render();
      this.bindEvents();
    } else {
      this.callbacks = containerOrCallbacks;
    }
  }

  mount(container: HTMLElement): void {
    this.container = container;
    this.render();
    this.bindEvents();
  }

  selectPreset(id: string): void {
    const preset = getPreset(id);
    if (!preset) {
      return;
    }
    this.currentPresetId = preset.id;
    this.isCustom = false;
    this.updateDom();
    this.callbacks.onSelectPreset(preset);
  }

  setCustom(): void {
    if (this.isCustom) {
      return;
    }
    this.isCustom = true;
    if (this.customBadgeEl) {
      this.customBadgeEl.hidden = false;
    }
    for (const btn of this.buttonMap.values()) {
      btn.classList.remove('active');
      btn.setAttribute('aria-checked', 'false');
    }
    if (this.selectEl) {
      let customOpt = this.selectEl.querySelector<HTMLOptionElement>('option[value="custom"]');
      if (!customOpt) {
        customOpt = document.createElement('option');
        customOpt.value = 'custom';
        customOpt.textContent = 'Custom';
        this.selectEl.appendChild(customOpt);
      }
      this.selectEl.value = 'custom';
    }
    if (this.descEl) {
      this.descEl.textContent = 'Custom modified physical parameters (Voicing / Acoustic Space).';
    }
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.buttonMap.clear();
    if (this.container) {
      this.container.innerHTML = '';
      this.container = null;
    }
  }

  private render(): void {
    if (!this.container) return;
    const initialPreset = getPreset(this.currentPresetId);

    this.container.innerHTML = `
      <section class="preset-bar-panel" aria-label="Physical Parameter Presets">
        <div class="preset-header">
          <div class="preset-label-group">
            <span class="preset-title">Piano Acoustic Model</span>
            <span id="badge-preset-custom" data-testid="preset-custom-badge" class="badge-custom" hidden>Custom</span>
          </div>
          <div class="preset-select-wrapper">
            <select id="select-preset" data-testid="preset-select" class="select-preset" aria-label="Preset Archetype">
              ${PIANO_PRESETS.map((p) => `<option value="${p.id}" ${p.id === this.currentPresetId ? 'selected' : ''}>${p.name}</option>`).join('')}
            </select>
          </div>
        </div>

        <div class="preset-controls-row">
          <div class="segmented-group preset-segmented" role="radiogroup" aria-label="Acoustic Preset Archetypes">
            ${PIANO_PRESETS.map((p) => `
              <button
                type="button"
                class="btn-segmented ${p.id === this.currentPresetId ? 'active' : ''}"
                data-testid="preset-${p.id}"
                data-preset-id="${p.id}"
                role="radio"
                aria-checked="${p.id === this.currentPresetId ? 'true' : 'false'}"
              >
                ${p.name}
              </button>
            `).join('')}
          </div>
        </div>

        <p id="preset-desc" data-testid="preset-description" class="preset-description">
          ${initialPreset?.description ?? ''}
        </p>
      </section>
    `;

    this.selectEl = this.container.querySelector('#select-preset')!;
    this.descEl = this.container.querySelector('#preset-desc')!;
    this.customBadgeEl = this.container.querySelector('#badge-preset-custom')!;

    for (const p of PIANO_PRESETS) {
      const btn = this.container.querySelector<HTMLButtonElement>(`[data-preset-id="${p.id}"]`);
      if (btn) {
        this.buttonMap.set(p.id, btn);
      }
    }
  }

  private bindEvents(): void {
    if (!this.container) return;

    // Dropdown change
    const onSelectChange = () => {
      const val = this.selectEl.value;
      if (val !== 'custom') {
        this.selectPreset(val);
      }
    };
    this.selectEl.addEventListener('change', onSelectChange);
    this.cleanupHandlers.push(() => this.selectEl.removeEventListener('change', onSelectChange));

    // Segmented button clicks
    for (const [id, btn] of this.buttonMap) {
      const onClick = () => {
        this.selectPreset(id);
      };
      btn.addEventListener('click', onClick);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', onClick));
    }
  }

  private updateDom(): void {
    const preset = getPreset(this.currentPresetId);
    if (!preset) return;

    if (this.customBadgeEl) {
      this.customBadgeEl.hidden = true;
    }
    const customOpt = this.selectEl.querySelector('option[value="custom"]');
    if (customOpt) {
      customOpt.remove();
    }
    if (this.selectEl) {
      this.selectEl.value = preset.id;
    }
    if (this.descEl) {
      this.descEl.textContent = preset.description;
    }
    for (const [id, btn] of this.buttonMap) {
      const active = id === preset.id;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    }
  }
}
