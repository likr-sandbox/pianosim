/**
 * PianoKeyboard: responsive semantic-DOM 88-key keyboard (A0-C8 by default).
 *
 * - Every key is a <button class="key key-white|key-black" data-midi aria-label>, emitted in
 *   ascending MIDI order (black keys interleaved) inside the container.
 * - Pointer input (mouse / touch / pen) with polyphonic multi-touch and glissando: while a
 *   pointer is down, moving it across keys (resolved with document.elementFromPoint) releases
 *   the previous key and presses the new one.
 * - The component only reports note on/off; visual state is driven through setActive().
 */

import { isBlackKey, midiToNoteName, pitchClass } from './noteUtils';

export interface PianoKeyboardOptions {
  container: HTMLElement;
  onNoteOn: (midiNote: number, velocity: number, source: string) => void;
  onNoteOff: (midiNote: number, source: string) => void;
  startNote?: number; // default 21 (A0)
  endNote?: number; // default 108 (C8)
  /** Velocity used for pointer strikes (defaults to 80). */
  getVelocity?: () => number;
}

const BLACK_WIDTH_RATIO = 0.62; // black key width relative to a white key

export class PianoKeyboard {
  private readonly container: HTMLElement;
  private readonly opts: PianoKeyboardOptions;
  private readonly startNote: number;
  private readonly endNote: number;
  private readonly keys = new Map<number, HTMLButtonElement>();
  /** pointerId -> midi currently held by that pointer */
  private readonly pointers = new Map<number, number>();
  private readonly disposers: Array<() => void> = [];

  constructor(options: PianoKeyboardOptions) {
    this.opts = options;
    this.container = options.container;
    this.startNote = options.startNote ?? 21;
    this.endNote = options.endNote ?? 108;
    this.render();
    this.bind();
  }

  private render(): void {
    const whiteNotes: number[] = [];
    for (let m = this.startNote; m <= this.endNote; m++) if (!isBlackKey(m)) whiteNotes.push(m);
    const whiteCount = whiteNotes.length;
    const whitePct = 100 / whiteCount;

    this.container.classList.add('piano-keys');
    this.container.style.setProperty('--white-count', String(whiteCount));
    this.container.replaceChildren();

    let whiteIndex = 0;
    const frag = document.createDocumentFragment();
    for (let m = this.startNote; m <= this.endNote; m++) {
      const black = isBlackKey(m);
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.tabIndex = -1;
      btn.className = `key ${black ? 'key-black' : 'key-white'}`;
      btn.dataset.midi = String(m);
      const name = midiToNoteName(m);
      btn.dataset.note = name;
      btn.setAttribute('aria-label', name);
      btn.setAttribute('aria-pressed', 'false');

      if (black) {
        const w = whitePct * BLACK_WIDTH_RATIO;
        // Centered on the boundary between the previous white key and the next one.
        btn.style.left = `${whiteIndex * whitePct - w / 2}%`;
        btn.style.width = `${w}%`;
      } else {
        btn.style.left = `${whiteIndex * whitePct}%`;
        btn.style.width = `${whitePct}%`;
        whiteIndex++;
        if (pitchClass(m) === 0) {
          const label = document.createElement('span');
          label.className = 'key-label';
          label.textContent = name;
          btn.appendChild(label);
        }
      }
      const hint = document.createElement('span');
      hint.className = 'key-hint';
      hint.setAttribute('aria-hidden', 'true');
      btn.appendChild(hint);

      this.keys.set(m, btn);
      frag.appendChild(btn);
    }
    this.container.appendChild(frag);
  }

  private keyFromElement(el: Element | null): number | null {
    const btn = el?.closest?.<HTMLElement>('.key');
    if (!btn || !this.container.contains(btn)) return null;
    const midi = Number(btn.dataset.midi);
    return Number.isFinite(midi) ? midi : null;
  }

  private sourceFor(pointerId: number): string {
    return `pointer:${pointerId}`;
  }

  private velocity(): number {
    return this.opts.getVelocity ? this.opts.getVelocity() : 80;
  }

  private press(pointerId: number, midi: number): void {
    this.pointers.set(pointerId, midi);
    this.opts.onNoteOn(midi, this.velocity(), this.sourceFor(pointerId));
  }

  private releasePointer(pointerId: number): void {
    const midi = this.pointers.get(pointerId);
    if (midi === undefined) return;
    this.pointers.delete(pointerId);
    this.opts.onNoteOff(midi, this.sourceFor(pointerId));
  }

  private bind(): void {
    const on = <K extends keyof HTMLElementEventMap>(
      target: HTMLElement,
      type: K,
      fn: (e: HTMLElementEventMap[K]) => void,
    ) => {
      target.addEventListener(type, fn);
      this.disposers.push(() => target.removeEventListener(type, fn));
    };
    const onWin = <K extends keyof WindowEventMap>(type: K, fn: (e: WindowEventMap[K]) => void) => {
      window.addEventListener(type, fn);
      this.disposers.push(() => window.removeEventListener(type, fn));
    };

    on(this.container, 'pointerdown', (e) => {
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      const midi = this.keyFromElement(e.target as Element);
      if (midi === null) return;
      e.preventDefault();
      // Touch pointers are implicitly captured to the start element; release so that
      // pointermove keeps working across keys (we resolve keys via elementFromPoint).
      try {
        (e.target as Element).releasePointerCapture?.(e.pointerId);
      } catch {
        /* synthetic pointer ids have nothing to release */
      }
      this.releasePointer(e.pointerId); // defensive: stale state for this pointer
      this.press(e.pointerId, midi);
    });

    on(this.container, 'mousedown', (e) => e.preventDefault()); // keep focus off the keys
    on(this.container, 'contextmenu', (e) => e.preventDefault());

    onWin('pointermove', (e) => {
      const current = this.pointers.get(e.pointerId);
      if (current === undefined) return;
      if (e.pointerType === 'mouse' && e.buttons === 0) {
        this.releasePointer(e.pointerId); // button released outside the window
        return;
      }
      const el = document.elementFromPoint(e.clientX, e.clientY);
      const midi = this.keyFromElement(el);
      if (midi === null || midi === current) return;
      this.releasePointer(e.pointerId);
      this.press(e.pointerId, midi);
    });

    const end = (e: PointerEvent) => this.releasePointer(e.pointerId);
    onWin('pointerup', end);
    onWin('pointercancel', end);
    onWin('blur', () => this.releaseAll());
  }

  /** Release every pointer-held note (focus loss etc.). */
  releaseAll(): void {
    for (const id of [...this.pointers.keys()]) this.releasePointer(id);
  }

  /** Visual state. `sustained` = released by the player but still ringing under the pedal. */
  setActive(midiNote: number, active: boolean, sustained = false): void {
    const btn = this.keys.get(midiNote);
    if (!btn) return;
    const lit = active || sustained;
    btn.classList.toggle('active', lit);
    btn.classList.toggle('sustained', sustained && !active);
    btn.setAttribute('aria-pressed', lit ? 'true' : 'false');
  }

  /** Show computer-keyboard hints on keys (midi -> label); other keys are cleared. */
  setHints(hints: ReadonlyMap<number, string>): void {
    for (const [midi, btn] of this.keys) {
      const el = btn.querySelector<HTMLElement>('.key-hint');
      if (!el) continue;
      const text = hints.get(midi) ?? '';
      el.textContent = text;
      btn.classList.toggle('has-hint', text !== '');
    }
  }

  getKeyElement(midiNote: number): HTMLButtonElement | undefined {
    return this.keys.get(midiNote);
  }

  destroy(): void {
    this.releaseAll();
    this.disposers.splice(0).forEach((d) => d());
    this.container.replaceChildren();
    this.keys.clear();
  }
}
