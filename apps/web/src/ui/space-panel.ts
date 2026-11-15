/** Space & Acoustics panel managing grand piano lid state and listening perspective. */

import {
  LID_CLOSED,
  LID_HALF,
  LID_OPEN,
  PERSPECTIVE_BINAURAL,
  PERSPECTIVE_CLOSE,
  PERSPECTIVE_HALL,
  PERSPECTIVE_PLAYER,
  type LidState,
  type ListenerPerspective,
} from '../audio/abi';

export type SpaceChangeCallback = (
  perspective: ListenerPerspective,
  lidState: LidState,
) => void;

export class SpacePanel {
  private readonly container: HTMLElement;
  private readonly onChange: SpaceChangeCallback;

  private currentPerspective: ListenerPerspective = PERSPECTIVE_PLAYER;
  private currentLid: LidState = LID_OPEN;

  private readonly lidButtons = new Map<LidState, HTMLButtonElement>();
  private readonly perspectiveButtons = new Map<ListenerPerspective, HTMLButtonElement>();
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, onChange: SpaceChangeCallback) {
    this.container = container;
    this.onChange = onChange;
    this.render();
    this.bindEvents();
  }

  getPerspective(): ListenerPerspective {
    return this.currentPerspective;
  }

  getLidState(): LidState {
    return this.currentLid;
  }

  setSpace(perspective: ListenerPerspective, lid: LidState): void {
    this.currentPerspective = perspective;
    this.currentLid = lid;
    this.updateDomState();
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.lidButtons.clear();
    this.perspectiveButtons.clear();
    this.container.innerHTML = '';
  }

  private render(): void {
    this.container.innerHTML = `
      <section class="space-panel" aria-label="Acoustic Space and Perspective">
        <h3 class="panel-title">Acoustic Space</h3>
        <div class="space-controls">
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
              >
                Open
              </button>
              <button
                type="button"
                data-testid="lid-half"
                data-lid="half"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >
                Half
              </button>
              <button
                type="button"
                data-testid="lid-closed"
                data-lid="closed"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >
                Closed
              </button>
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
              >
                Player
              </button>
              <button
                type="button"
                data-testid="perspective-close"
                data-perspective="close"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >
                Close Stereo
              </button>
              <button
                type="button"
                data-testid="perspective-hall"
                data-perspective="hall"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >
                Concert Hall
              </button>
              <button
                type="button"
                data-testid="perspective-binaural"
                data-perspective="binaural"
                class="btn btn-segmented"
                role="radio"
                aria-checked="false"
              >
                Binaural HRTF
              </button>
            </div>
          </div>
        </div>
      </section>
    `;

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
  }

  private bindEvents(): void {
    for (const [lid, btn] of this.lidButtons.entries()) {
      const handler = () => {
        this.currentLid = lid;
        this.updateDomState();
        this.onChange(this.currentPerspective, this.currentLid);
      };
      btn.addEventListener('click', handler);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', handler));
    }

    for (const [persp, btn] of this.perspectiveButtons.entries()) {
      const handler = () => {
        this.currentPerspective = persp;
        this.updateDomState();
        this.onChange(this.currentPerspective, this.currentLid);
      };
      btn.addEventListener('click', handler);
      this.cleanupHandlers.push(() => btn.removeEventListener('click', handler));
    }
  }

  private updateDomState(): void {
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
  }
}
