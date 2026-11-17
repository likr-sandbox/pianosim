/** Segmented UI selector for Light / Natural / Heavy touch velocity curves. */

import {
  DEFAULT_TOUCH_CURVE,
  TOUCH_CURVE_OPTIONS,
  type TouchCurveMode,
} from '../input/touch-curve';

export type TouchCurveChangeCallback = (mode: TouchCurveMode) => void;

export class TouchCurveSelector {
  private readonly container: HTMLElement;
  private readonly onChange: TouchCurveChangeCallback;
  private currentMode: TouchCurveMode = DEFAULT_TOUCH_CURVE;
  private readonly buttons = new Map<TouchCurveMode, HTMLButtonElement>();
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, onChange: TouchCurveChangeCallback) {
    this.container = container;
    this.onChange = onChange;
    this.render();
    this.bindEvents();
  }

  getMode(): TouchCurveMode {
    return this.currentMode;
  }

  setMode(mode: TouchCurveMode, emit = false): void {
    if (this.currentMode === mode && !emit) {
      return;
    }
    this.currentMode = mode;
    this.updateDom();
    if (emit) {
      this.onChange(mode);
    }
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.buttons.clear();
    this.container.innerHTML = '';
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="touch-curve-control">
        <span class="control-label">Touch Curve</span>
        <div
          class="segmented-group touch-curve-group"
          data-testid="touch-curve-group"
          role="radiogroup"
          aria-label="Touch Curve"
        >
          ${TOUCH_CURVE_OPTIONS.map((opt) => {
            const active = opt.mode === this.currentMode;
            return `
              <button
                type="button"
                class="btn-segmented ${active ? 'active' : ''}"
                data-testid="touch-curve-${opt.mode}"
                data-touch-curve="${opt.mode}"
                role="radio"
                aria-checked="${active ? 'true' : 'false'}"
              >
                ${opt.label}
              </button>
            `;
          }).join('')}
        </div>
      </div>
    `;

    for (const opt of TOUCH_CURVE_OPTIONS) {
      const btn = this.container.querySelector<HTMLButtonElement>(
        `[data-touch-curve="${opt.mode}"]`,
      );
      if (btn) {
        this.buttons.set(opt.mode, btn);
      }
    }
  }

  private bindEvents(): void {
    for (const [mode, btn] of this.buttons) {
      const onClick = () => {
        this.currentMode = mode;
        this.updateDom();
        this.onChange(mode);
      };
      btn.addEventListener('click', onClick);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', onClick));
    }
  }

  private updateDom(): void {
    for (const [mode, btn] of this.buttons) {
      const active = mode === this.currentMode;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    }
  }
}
