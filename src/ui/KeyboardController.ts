/**
 * KeyboardController: computer-keyboard -> piano mapping.
 *
 *   W E   T Y U   O P        black keys
 *  A S D F G H J K L ;       white keys (A = C4 at octave shift 0)
 *  Z / X = octave down / up,  Space (hold) = sustain pedal
 *
 * Uses KeyboardEvent.code (layout independent), ignores auto-repeat, remembers which
 * midi note each physical key started so octave shifts never leave stuck notes.
 */

export interface KeyboardControllerOptions {
  onNoteOn: (midiNote: number, source: string) => void;
  onNoteOff: (midiNote: number, source: string) => void;
  onPedal: (down: boolean) => void;
  onOctaveChange?: (shift: number) => void;
  minOctaveShift?: number;
  maxOctaveShift?: number;
  minMidi?: number;
  maxMidi?: number;
}

/** key code -> semitone offset from C4 (60) */
export const KEY_OFFSETS: Readonly<Record<string, number>> = {
  KeyA: 0,
  KeyW: 1,
  KeyS: 2,
  KeyE: 3,
  KeyD: 4,
  KeyF: 5,
  KeyT: 6,
  KeyG: 7,
  KeyY: 8,
  KeyH: 9,
  KeyU: 10,
  KeyJ: 11,
  KeyK: 12,
  KeyO: 13,
  KeyL: 14,
  KeyP: 15,
  Semicolon: 16,
};

export const KEY_LABELS: Readonly<Record<string, string>> = {
  Semicolon: ';',
};

export const BASE_NOTE = 60;

export class KeyboardController {
  private readonly opts: KeyboardControllerOptions;
  private readonly held = new Map<string, number>(); // code -> midi sounding
  private octaveShift = 0;
  private pedalDown = false;
  private readonly minShift: number;
  private readonly maxShift: number;
  private readonly minMidi: number;
  private readonly maxMidi: number;
  private readonly disposers: Array<() => void> = [];

  constructor(options: KeyboardControllerOptions) {
    this.opts = options;
    this.minShift = options.minOctaveShift ?? -3;
    this.maxShift = options.maxOctaveShift ?? 3;
    this.minMidi = options.minMidi ?? 21;
    this.maxMidi = options.maxMidi ?? 108;

    const kd = (e: KeyboardEvent) => this.onKeyDown(e);
    const ku = (e: KeyboardEvent) => this.onKeyUp(e);
    const blur = () => this.releaseAll();
    const vis = () => {
      if (document.hidden) this.releaseAll();
    };
    window.addEventListener('keydown', kd);
    window.addEventListener('keyup', ku);
    window.addEventListener('blur', blur);
    document.addEventListener('visibilitychange', vis);
    this.disposers.push(
      () => window.removeEventListener('keydown', kd),
      () => window.removeEventListener('keyup', ku),
      () => window.removeEventListener('blur', blur),
      () => document.removeEventListener('visibilitychange', vis),
    );
  }

  getOctaveShift(): number {
    return this.octaveShift;
  }

  setOctaveShift(shift: number): void {
    const next = Math.max(this.minShift, Math.min(this.maxShift, Math.round(shift)));
    if (next === this.octaveShift) return;
    this.octaveShift = next;
    this.opts.onOctaveChange?.(next);
  }

  /** Current midi -> key label map (for rendering hints). */
  getHints(): Map<number, string> {
    const map = new Map<number, string>();
    for (const [code, off] of Object.entries(KEY_OFFSETS)) {
      const midi = BASE_NOTE + this.octaveShift * 12 + off;
      if (midi >= this.minMidi && midi <= this.maxMidi) {
        map.set(midi, KEY_LABELS[code] ?? code.replace('Key', ''));
      }
    }
    return map;
  }

  private shouldIgnore(e: KeyboardEvent): boolean {
    if (e.ctrlKey || e.metaKey || e.altKey) return true;
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT')) return true;
    if (t && t.tagName === 'INPUT' && (t as HTMLInputElement).type !== 'range' && (t as HTMLInputElement).type !== 'checkbox') {
      return true;
    }
    return false;
  }

  private onKeyDown(e: KeyboardEvent): void {
    if (this.shouldIgnore(e)) return;
    const code = e.code;

    if (code === 'Space') {
      e.preventDefault(); // no page scroll / button activation
      if (e.repeat || this.pedalDown) return;
      this.pedalDown = true;
      this.opts.onPedal(true);
      return;
    }
    if (code === 'KeyZ' || code === 'KeyX') {
      if (!e.repeat) this.setOctaveShift(this.octaveShift + (code === 'KeyX' ? 1 : -1));
      return;
    }
    const off = KEY_OFFSETS[code];
    if (off === undefined) return;
    if (e.repeat || this.held.has(code)) return;
    const midi = BASE_NOTE + this.octaveShift * 12 + off;
    if (midi < this.minMidi || midi > this.maxMidi) return;
    this.held.set(code, midi);
    this.opts.onNoteOn(midi, `key:${code}`);
  }

  private onKeyUp(e: KeyboardEvent): void {
    const code = e.code;
    if (code === 'Space') {
      e.preventDefault();
      if (this.pedalDown) {
        this.pedalDown = false;
        this.opts.onPedal(false);
      }
      return;
    }
    const midi = this.held.get(code);
    if (midi === undefined) return;
    this.held.delete(code);
    this.opts.onNoteOff(midi, `key:${code}`);
  }

  releaseAll(): void {
    for (const [code, midi] of [...this.held]) {
      this.held.delete(code);
      this.opts.onNoteOff(midi, `key:${code}`);
    }
    if (this.pedalDown) {
      this.pedalDown = false;
      this.opts.onPedal(false);
    }
  }

  destroy(): void {
    this.releaseAll();
    this.disposers.splice(0).forEach((d) => d());
  }
}
