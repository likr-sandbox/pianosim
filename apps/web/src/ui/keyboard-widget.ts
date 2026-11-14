export interface KeyDef {
  readonly midi: number;
  readonly label: string;
  readonly keyHint: string;
  readonly code: string;
  readonly isBlack: boolean;
  readonly leftPct?: number;
}

export const OCTAVE_KEYS: ReadonlyArray<KeyDef> = [
  { midi: 60, label: 'C4', keyHint: 'A', code: 'KeyA', isBlack: false },
  { midi: 61, label: 'C#4', keyHint: 'W', code: 'KeyW', isBlack: true, leftPct: 8.5 },
  { midi: 62, label: 'D4', keyHint: 'S', code: 'KeyS', isBlack: false },
  { midi: 63, label: 'D#4', keyHint: 'E', code: 'KeyE', isBlack: true, leftPct: 21.0 },
  { midi: 64, label: 'E4', keyHint: 'D', code: 'KeyD', isBlack: false },
  { midi: 65, label: 'F4', keyHint: 'F', code: 'KeyF', isBlack: false },
  { midi: 66, label: 'F#4', keyHint: 'T', code: 'KeyT', isBlack: true, leftPct: 46.0 },
  { midi: 67, label: 'G4', keyHint: 'G', code: 'KeyG', isBlack: false },
  { midi: 68, label: 'G#4', keyHint: 'Y', code: 'KeyY', isBlack: true, leftPct: 58.5 },
  { midi: 69, label: 'A4', keyHint: 'H', code: 'KeyH', isBlack: false },
  { midi: 70, label: 'A#4', keyHint: 'U', code: 'KeyU', isBlack: true, leftPct: 71.0 },
  { midi: 71, label: 'B4', keyHint: 'J', code: 'KeyJ', isBlack: false },
  { midi: 72, label: 'C5', keyHint: 'K', code: 'KeyK', isBlack: false },
];

const CODE_TO_MIDI = new Map<string, number>(OCTAVE_KEYS.map((k) => [k.code, k.midi]));

export interface KeyboardCallbacks {
  onPress(midi: number): void;
  onRelease(midi: number): void;
  onToggleSustain(): void;
  onBlurReset(): void;
}

/** Interactive 1-octave C4-C5 keyboard with pointer capture and PC key repeat suppression. */
export class TestKeyboardWidget {
  private readonly keyEls = new Map<number, HTMLButtonElement>();
  private readonly pointerNotes = new Map<number, number>();
  private readonly keyNotes = new Map<string, number>();
  private readonly callbacks: KeyboardCallbacks;
  private readonly unbinders: Array<() => void> = [];

  constructor(container: HTMLElement, callbacks: KeyboardCallbacks) {
    this.callbacks = callbacks;
    this.renderKeys(container);
    this.bindGlobalKeys();
  }

  setKeyActive(midi: number, active: boolean): void {
    const el = this.keyEls.get(midi);
    if (el) {
      el.classList.toggle('is-active', active);
    }
  }

  clearInputs(): void {
    this.pointerNotes.clear();
    this.keyNotes.clear();
    for (const el of this.keyEls.values()) {
      el.classList.remove('is-active');
    }
  }

  destroy(): void {
    for (const unbind of this.unbinders) {
      unbind();
    }
    this.unbinders.length = 0;
  }

  private renderKeys(container: HTMLElement): void {
    for (const keyDef of OCTAVE_KEYS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = keyDef.isBlack ? 'piano-key key-black' : 'piano-key key-white';
      if (keyDef.isBlack && keyDef.leftPct !== undefined) {
        btn.style.left = `${keyDef.leftPct}%`;
      }
      btn.dataset.midi = String(keyDef.midi);
      btn.dataset.note = keyDef.label;
      btn.setAttribute('aria-label', `${keyDef.label} (${keyDef.keyHint})`);
      btn.innerHTML = `<span class="key-note">${keyDef.label}</span><span class="key-hint">${keyDef.keyHint}</span>`;
      container.appendChild(btn);
      this.keyEls.set(keyDef.midi, btn);
      this.bindKeyPointer(btn, keyDef.midi);
    }
  }

  private bindKeyPointer(btn: HTMLButtonElement, midi: number): void {
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === 'mouse') {
        return;
      }
      e.preventDefault();
      if (typeof btn.setPointerCapture === 'function') {
        try {
          btn.setPointerCapture(e.pointerId);
        } catch {
          // Synthetic pointer events in automated tests may lack an active capture target.
        }
      }
      this.pointerNotes.set(e.pointerId, midi);
      this.callbacks.onPress(midi);
    };

    const onPointerRelease = (e: PointerEvent) => {
      const heldMidi = this.pointerNotes.get(e.pointerId);
      if (heldMidi !== undefined) {
        this.pointerNotes.delete(e.pointerId);
        this.callbacks.onRelease(heldMidi);
      }
    };

    btn.addEventListener('pointerdown', onPointerDown);
    btn.addEventListener('pointerup', onPointerRelease);
    btn.addEventListener('pointercancel', onPointerRelease);
    btn.addEventListener('lostpointercapture', onPointerRelease);

    this.unbinders.push(
      () => btn.removeEventListener('pointerdown', onPointerDown),
      () => btn.removeEventListener('pointerup', onPointerRelease),
      () => btn.removeEventListener('pointercancel', onPointerRelease),
      () => btn.removeEventListener('lostpointercapture', onPointerRelease),
    );
  }

  private bindGlobalKeys(): void {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) {
        return;
      }
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')
      ) {
        return;
      }
      if (e.code === 'Space') {
        e.preventDefault();
        this.callbacks.onToggleSustain();
        return;
      }
      const midi = CODE_TO_MIDI.get(e.code);
      if (midi !== undefined && !this.keyNotes.has(e.code)) {
        e.preventDefault();
        this.keyNotes.set(e.code, midi);
        this.callbacks.onPress(midi);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const midi = this.keyNotes.get(e.code);
      if (midi !== undefined) {
        this.keyNotes.delete(e.code);
        this.callbacks.onRelease(midi);
      }
    };

    const onBlur = () => {
      this.callbacks.onBlurReset();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    this.unbinders.push(
      () => window.removeEventListener('keydown', onKeyDown),
      () => window.removeEventListener('keyup', onKeyUp),
      () => window.removeEventListener('blur', onBlur),
    );
  }
}
