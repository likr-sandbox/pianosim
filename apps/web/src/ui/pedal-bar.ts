/** Three-pedal control bar with continuous half-pedaling slider and Sostenuto/Una Corda latch buttons. */

import type { PedalController } from '../input/pedal-controller';

export class PedalBar {
  private readonly container: HTMLElement;
  private readonly controller: PedalController;

  private sustainSlider!: HTMLInputElement;
  private sustainBtn!: HTMLButtonElement;
  private sostenutoBtn!: HTMLButtonElement;
  private unaCordaBtn!: HTMLButtonElement;

  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, controller: PedalController) {
    this.container = container;
    this.controller = controller;
    this.render();
    this.bindEvents();
  }

  updateState(sustain: number, sostenuto: boolean, unaCorda: number | boolean): void {
    if (this.sustainSlider) {
      this.sustainSlider.value = sustain.toFixed(2);
    }
    if (this.sustainBtn) {
      const isDown = sustain > 0.05;
      this.sustainBtn.setAttribute('aria-pressed', isDown ? 'true' : 'false');
      this.sustainBtn.classList.toggle('active', isDown);
      this.sustainBtn.textContent = isDown
        ? `Sustain: ${(sustain * 100).toFixed(0)}%`
        : 'Sustain: OFF';
    }
    if (this.sostenutoBtn) {
      this.sostenutoBtn.setAttribute('aria-pressed', sostenuto ? 'true' : 'false');
      this.sostenutoBtn.classList.toggle('active', sostenuto);
    }
    if (this.unaCordaBtn) {
      const unaVal = typeof unaCorda === 'number' ? unaCorda : unaCorda ? 1.0 : 0.0;
      const isUnaDown = unaVal > 0.02;
      this.unaCordaBtn.setAttribute('aria-pressed', isUnaDown ? 'true' : 'false');
      this.unaCordaBtn.classList.toggle('active', isUnaDown);
      this.unaCordaBtn.dataset.value = unaVal.toFixed(3);
      this.unaCordaBtn.textContent = isUnaDown
        ? `Una Corda: ${(unaVal * 100).toFixed(0)}%`
        : 'Una Corda (Soft)';
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
      <section class="pedal-bar" aria-label="Piano Pedals">
        <div class="pedal-group">
          <button
            id="pedal-una-corda"
            data-testid="pedal-una-corda"
            type="button"
            class="btn btn-toggle"
            aria-pressed="false"
          >
            Una Corda (Soft)
          </button>
          <button
            id="pedal-sostenuto"
            data-testid="pedal-sostenuto"
            type="button"
            class="btn btn-toggle"
            aria-pressed="false"
          >
            Sostenuto
          </button>
          <button
            id="btn-sustain"
            data-testid="sustain-toggle"
            type="button"
            class="btn btn-toggle"
            aria-pressed="false"
          >
            Sustain: OFF
          </button>
        </div>

        <div class="pedal-slider-group">
          <label for="slider-sustain" class="pedal-slider-label">
            <span>Half-Pedal Clearance</span>
            <input
              id="slider-sustain"
              data-testid="sustain-slider"
              type="range"
              min="0.0"
              max="1.0"
              step="0.01"
              value="0.0"
              class="pedal-slider"
            />
          </label>
        </div>
      </section>
    `;

    this.sustainSlider = this.container.querySelector('#slider-sustain')!;
    this.sustainBtn = this.container.querySelector('#btn-sustain')!;
    this.sostenutoBtn = this.container.querySelector('#pedal-sostenuto')!;
    this.unaCordaBtn = this.container.querySelector('#pedal-una-corda')!;
  }

  private bindEvents(): void {
    const onSliderInput = () => {
      const val = parseFloat(this.sustainSlider.value);
      this.controller.setSustain(val);
    };

    const onSustainClick = () => {
      this.controller.toggleSustain();
    };

    const onSostenutoClick = () => {
      this.controller.toggleSostenuto();
    };

    const onUnaCordaClick = () => {
      this.controller.toggleUnaCorda();
    };

    this.sustainSlider.addEventListener('input', onSliderInput);
    this.sustainBtn.addEventListener('click', onSustainClick);
    this.sostenutoBtn.addEventListener('click', onSostenutoClick);
    this.unaCordaBtn.addEventListener('click', onUnaCordaClick);

    this.cleanupHandlers.push(() => {
      this.sustainSlider.removeEventListener('input', onSliderInput);
      this.sustainBtn.removeEventListener('click', onSustainClick);
      this.sostenutoBtn.removeEventListener('click', onSostenutoClick);
      this.unaCordaBtn.removeEventListener('click', onUnaCordaClick);
    });
  }
}
