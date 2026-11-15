/** PC keyboard chromatic mapping with octave transposition (Z/X), key repeat suppression, and blur safety. */

export interface PcKeyboardCallbacks {
  onPress(midiKey: number): void;
  onRelease(midiKey: number): void;
  onToggleSustain(): void;
  onOctaveChange?(octave: number): void;
}

const KEY_SEMITONES: Record<string, number> = {
  // White keys
  KeyA: 0,  // C
  KeyS: 2,  // D
  KeyD: 4,  // E
  KeyF: 5,  // F
  KeyG: 7,  // G
  KeyH: 9,  // A
  KeyJ: 11, // B
  KeyK: 12, // C (+1 octave)
  KeyL: 14, // D (+1 octave)
  Semicolon: 16, // E (+1 octave)

  // Black keys
  KeyW: 1,  // C#
  KeyE: 3,  // D#
  KeyT: 6,  // F#
  KeyY: 8,  // G#
  KeyU: 10, // A#
  KeyO: 13, // C# (+1 octave)
  KeyP: 15, // D# (+1 octave)
};

export class PcKeyboard {
  private readonly callbacks: PcKeyboardCallbacks;
  private octave = 4; // Default octave C4
  private readonly heldKeys = new Map<string, number>(); // code -> midiKey
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(callbacks: PcKeyboardCallbacks) {
    this.callbacks = callbacks;
    this.attachListeners();
  }

  getOctave(): number {
    return this.octave;
  }

  setOctave(octave: number): void {
    const clamped = Math.max(1, Math.min(7, Math.trunc(octave)));
    if (this.octave !== clamped) {
      this.octave = clamped;
      this.callbacks.onOctaveChange?.(this.octave);
    }
  }

  clear(): void {
    for (const midiKey of this.heldKeys.values()) {
      this.callbacks.onRelease(midiKey);
    }
    this.heldKeys.clear();
  }

  destroy(): void {
    this.clear();
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
  }

  private attachListeners(): void {
    const onKeyDown = (e: KeyboardEvent) => {
      // Suppress OS key-repeat events
      if (e.repeat) {
        return;
      }

      // Ignore if user is focused on form inputs
      const target = e.target as HTMLElement | null;
      if (target) {
        const tag = target.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
          return;
        }
      }

      // Ignore modifier keys
      if (e.ctrlKey || e.metaKey || e.altKey) {
        return;
      }

      // Octave shifts
      if (e.code === 'KeyZ') {
        this.setOctave(this.octave - 1);
        return;
      }
      if (e.code === 'KeyX') {
        this.setOctave(this.octave + 1);
        return;
      }

      // Sustain pedal toggle on Space
      if (e.code === 'Space') {
        e.preventDefault();
        this.callbacks.onToggleSustain();
        return;
      }

      const semitone = KEY_SEMITONES[e.code];
      if (semitone !== undefined) {
        const baseMidi = 12 * (this.octave + 1);
        const midi = baseMidi + semitone;
        if (midi >= 21 && midi <= 108) {
          this.heldKeys.set(e.code, midi);
          this.callbacks.onPress(midi);
        }
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const midi = this.heldKeys.get(e.code);
      if (midi !== undefined) {
        this.callbacks.onRelease(midi);
        this.heldKeys.delete(e.code);
      }
    };

    const onBlur = () => {
      this.clear();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onBlur);

    this.cleanupHandlers.push(() => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
    });
  }
}
