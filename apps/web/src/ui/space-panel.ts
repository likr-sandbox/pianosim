/** Space & Acoustics panel managing piano body archetype, lid state, listening perspective, and hall reverb. */

import {
  BODY_CONCERT_GRAND,
  BODY_OPTIONS,
  LID_CLOSED,
  LID_HALF,
  LID_OPEN,
  PERSPECTIVE_BINAURAL,
  PERSPECTIVE_CLOSE,
  PERSPECTIVE_HALL,
  PERSPECTIVE_PLAYER,
  REVERB_DECAY_DEFAULT,
  REVERB_DECAY_MAX,
  REVERB_DECAY_MIN,
  REVERB_DECAY_STEP,
  REVERB_MIX_DEFAULT,
  REVERB_MIX_MAX,
  REVERB_MIX_MIN,
  type BodyId,
  type LidState,
  type ListenerPerspective,
} from '../audio/abi';

export interface SpaceState {
  readonly bodyId: BodyId;
  readonly perspective: ListenerPerspective;
  readonly lidState: LidState;
  readonly reverbDecayS: number;
  readonly reverbMix: number;
}

export interface SpacePanelCallbacks {
  readonly onSpaceChange: (
    perspective: ListenerPerspective,
    lidState: LidState,
    bodyId: BodyId,
  ) => void;
  readonly onReverbChange: (decayS: number, mix: number) => void;
}

const clampVal = (raw: number, lo: number, hi: number, fallback: number): number =>
  Number.isFinite(raw) ? Math.max(lo, Math.min(hi, raw)) : fallback;

export class SpacePanel {
  private readonly container: HTMLElement;
  private readonly callbacks: SpacePanelCallbacks;

  private currentBody: BodyId = BODY_CONCERT_GRAND;
  private currentPerspective: ListenerPerspective = PERSPECTIVE_PLAYER;
  private currentLid: LidState = LID_OPEN;
  private currentReverbDecayS = REVERB_DECAY_DEFAULT;
  private currentReverbMix = REVERB_MIX_DEFAULT;

  private readonly bodyButtons = new Map<BodyId, HTMLButtonElement>();
  private readonly lidButtons = new Map<LidState, HTMLButtonElement>();
  private readonly perspectiveButtons = new Map<ListenerPerspective, HTMLButtonElement>();
  private reverbDecaySlider!: HTMLInputElement;
  private reverbMixSlider!: HTMLInputElement;
  private reverbDecayValEl!: HTMLElement;
  private reverbMixValEl!: HTMLElement;
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, callbacks: SpacePanelCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.render();
    this.bindEvents();
  }

  getBodyId(): BodyId {
    return this.currentBody;
  }

  getPerspective(): ListenerPerspective {
    return this.currentPerspective;
  }

  getLidState(): LidState {
    return this.currentLid;
  }

  getReverbDecayS(): number {
    return this.currentReverbDecayS;
  }

  getReverbMix(): number {
    return this.currentReverbMix;
  }

  setSpace(perspective: ListenerPerspective, lid: LidState, bodyId?: BodyId): void {
    this.currentPerspective = perspective;
    this.currentLid = lid;
    if (bodyId !== undefined) {
      this.currentBody = bodyId;
    }
    this.updateDomState();
  }

  setReverb(decayS: number, mix: number): void {
    this.currentReverbDecayS = clampVal(
      decayS,
      REVERB_DECAY_MIN,
      REVERB_DECAY_MAX,
      REVERB_DECAY_DEFAULT,
    );
    this.currentReverbMix = clampVal(mix, REVERB_MIX_MIN, REVERB_MIX_MAX, REVERB_MIX_DEFAULT);
    this.updateDomState();
  }

  setState(state: Partial<SpaceState>): void {
    if (state.bodyId !== undefined) this.currentBody = state.bodyId;
    if (state.perspective !== undefined) this.currentPerspective = state.perspective;
    if (state.lidState !== undefined) this.currentLid = state.lidState;
    if (state.reverbDecayS !== undefined) {
      this.currentReverbDecayS = clampVal(
        state.reverbDecayS,
        REVERB_DECAY_MIN,
        REVERB_DECAY_MAX,
        REVERB_DECAY_DEFAULT,
      );
    }
    if (state.reverbMix !== undefined) {
      this.currentReverbMix = clampVal(
        state.reverbMix,
        REVERB_MIX_MIN,
        REVERB_MIX_MAX,
        REVERB_MIX_DEFAULT,
      );
    }
    this.updateDomState();
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.bodyButtons.clear();
    this.lidButtons.clear();
    this.perspectiveButtons.clear();
    this.container.innerHTML = '';
  }

  private render(): void {
    const bodyButtonsHtml = BODY_OPTIONS.map((opt) => {
      const active = opt.id === this.currentBody;
      return `<button
        type="button"
        data-testid="${opt.testId}"
        data-body-id="${opt.id}"
        class="btn btn-segmented${active ? ' active' : ''}"
        aria-pressed="${active ? 'true' : 'false'}"
      >${opt.label}</button>`;
    }).join('');

    const initMixPct = Math.round(this.currentReverbMix * 100);

    this.container.innerHTML = `
      <section class="space-panel" aria-label="Acoustic Space and Perspective">
        <h3 class="panel-title">Acoustic Space</h3>
        <div class="space-controls">
          <div class="control-segment">
            <span class="segment-label">Piano Body</span>
            <div class="segmented-group" role="group" aria-label="Piano Body Archetype">
              ${bodyButtonsHtml}
            </div>
          </div>

          <div class="control-segment">
            <span class="segment-label">Grand Lid Position</span>
            <div class="segmented-group" role="radiogroup" aria-label="Grand Lid State">
              <button
                type="button"
                data-testid="lid-open"
                data-lid="open"
                class="btn btn-segmented active"
                role="radio"
                aria-checked="true"
              >Open</button>
              <button
                type="button"
                data-testid="lid-half"
                data-lid="half"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >Half</button>
              <button
                type="button"
                data-testid="lid-closed"
                data-lid="closed"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >Closed</button>
            </div>
          </div>

          <div class="control-segment">
            <span class="segment-label">Acoustic Perspective</span>
            <div class="segmented-group" role="radiogroup" aria-label="Listening Perspective">
              <button
                type="button"
                data-testid="perspective-player"
                data-perspective="player"
                class="btn btn-segmented active"
                role="radio"
                aria-checked="true"
              >Player</button>
              <button
                type="button"
                data-testid="perspective-close"
                data-perspective="close"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >Close Stereo</button>
              <button
                type="button"
                data-testid="perspective-hall"
                data-perspective="hall"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >Concert Hall</button>
              <button
                type="button"
                data-testid="perspective-binaural"
                data-perspective="binaural"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >Binaural HRTF</button>
            </div>
          </div>

          <div class="reverb-grid">
            <label class="control-row" for="slider-reverb-decay">
              <span class="control-label">Reverb Decay</span>
              <input
                id="slider-reverb-decay"
                data-testid="reverb-decay-slider"
                type="range"
                min="${REVERB_DECAY_MIN.toFixed(1)}"
                max="${REVERB_DECAY_MAX.toFixed(1)}"
                step="${REVERB_DECAY_STEP}"
                value="${this.currentReverbDecayS.toFixed(2)}"
              />
              <span id="val-reverb-decay" data-testid="reverb-decay-value" class="val-mono">${this.currentReverbDecayS.toFixed(2)} s</span>
            </label>

            <label class="control-row" for="slider-reverb-mix">
              <span class="control-label">Reverb Mix</span>
              <input
                id="slider-reverb-mix"
                data-testid="reverb-mix-slider"
                type="range"
                min="0"
                max="100"
                step="1"
                value="${initMixPct}"
              />
              <span id="val-reverb-mix" data-testid="reverb-mix-value" class="val-mono">${initMixPct}%</span>
            </label>
          </div>
        </div>
      </section>
    `;

    for (const opt of BODY_OPTIONS) {
      this.bodyButtons.set(
        opt.id,
        this.container.querySelector(`[data-body-id="${opt.id}"]`)!,
      );
    }

    this.lidButtons.set(LID_OPEN, this.container.querySelector('[data-lid="open"]')!);
    this.lidButtons.set(LID_HALF, this.container.querySelector('[data-lid="half"]')!);
    this.lidButtons.set(LID_CLOSED, this.container.querySelector('[data-lid="closed"]')!);

    this.perspectiveButtons.set(
      PERSPECTIVE_PLAYER,
      this.container.querySelector('[data-perspective="player"]')!,
    );
    this.perspectiveButtons.set(
      PERSPECTIVE_CLOSE,
      this.container.querySelector('[data-perspective="close"]')!,
    );
    this.perspectiveButtons.set(
      PERSPECTIVE_HALL,
      this.container.querySelector('[data-perspective="hall"]')!,
    );
    this.perspectiveButtons.set(
      PERSPECTIVE_BINAURAL,
      this.container.querySelector('[data-perspective="binaural"]')!,
    );

    this.reverbDecaySlider = this.container.querySelector('#slider-reverb-decay')!;
    this.reverbMixSlider = this.container.querySelector('#slider-reverb-mix')!;
    this.reverbDecayValEl = this.container.querySelector('#val-reverb-decay')!;
    this.reverbMixValEl = this.container.querySelector('#val-reverb-mix')!;
  }

  private bindEvents(): void {
    for (const [bodyId, btn] of this.bodyButtons.entries()) {
      const handler = () => {
        this.currentBody = bodyId;
        this.updateDomState();
        this.callbacks.onSpaceChange(this.currentPerspective, this.currentLid, this.currentBody);
      };
      btn.addEventListener('click', handler);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', handler));
    }

    for (const [lid, btn] of this.lidButtons.entries()) {
      const handler = () => {
        this.currentLid = lid;
        this.updateDomState();
        this.callbacks.onSpaceChange(this.currentPerspective, this.currentLid, this.currentBody);
      };
      btn.addEventListener('click', handler);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', handler));
    }

    for (const [persp, btn] of this.perspectiveButtons.entries()) {
      const handler = () => {
        this.currentPerspective = persp;
        this.updateDomState();
        this.callbacks.onSpaceChange(this.currentPerspective, this.currentLid, this.currentBody);
      };
      btn.addEventListener('click', handler);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', handler));
    }

    const onReverbInput = () => {
      const decayS = clampVal(
        parseFloat(this.reverbDecaySlider.value),
        REVERB_DECAY_MIN,
        REVERB_DECAY_MAX,
        REVERB_DECAY_DEFAULT,
      );
      const mixPct = clampVal(parseFloat(this.reverbMixSlider.value), 0, 100, 18);
      this.currentReverbDecayS = decayS;
      this.currentReverbMix = mixPct / 100;
      this.reverbDecayValEl.textContent = `${decayS.toFixed(2)} s`;
      this.reverbMixValEl.textContent = `${Math.round(mixPct)}%`;
      this.callbacks.onReverbChange(this.currentReverbDecayS, this.currentReverbMix);
    };

    this.reverbDecaySlider.addEventListener('input', onReverbInput);
    this.reverbMixSlider.addEventListener('input', onReverbInput);
    this.cleanupHandlers.push(() => {
      this.reverbDecaySlider.removeEventListener('input', onReverbInput);
      this.reverbMixSlider.removeEventListener('input', onReverbInput);
    });
  }

  private updateDomState(): void {
    for (const [bodyId, btn] of this.bodyButtons.entries()) {
      const active = bodyId === this.currentBody;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-pressed', active ? 'true' : 'false');
    }
    for (const [lid, btn] of this.lidButtons.entries()) {
      const active = lid === this.currentLid;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    }
    for (const [persp, btn] of this.perspectiveButtons.entries()) {
      const active = persp === this.currentPerspective;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-checked', active ? 'true' : 'false');
    }
    this.reverbDecaySlider.value = this.currentReverbDecayS.toFixed(2);
    this.reverbDecayValEl.textContent = `${this.currentReverbDecayS.toFixed(2)} s`;
    const mixPct = Math.round(this.currentReverbMix * 100);
    this.reverbMixSlider.value = String(mixPct);
    this.reverbMixValEl.textContent = `${mixPct}%`;
  }
}
