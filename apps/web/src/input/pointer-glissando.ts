/** Multi-touch and pointer glissando tracker for responsive 88-key piano keyboard. */

export interface PointerGlissandoCallbacks {
  onPress(midiKey: number, velocity?: number): void;
  onRelease(midiKey: number): void;
  getBaseVelocity?: () => number;
}

const GLISSANDO_HOLD_MS = 450;

export class PointerGlissando {
  private readonly container: HTMLElement;
  private readonly callbacks: PointerGlissandoCallbacks;
  private readonly activePointers = new Map<number, number>(); // pointerId -> midiKey
  private readonly pendingGlissando = new Map<number, number>(); // timerId -> midiKey
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, callbacks: PointerGlissandoCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.attachListeners();
  }

  clear(): void {
    for (const [timerId, midiKey] of this.pendingGlissando.entries()) {
      window.clearTimeout(timerId);
      this.callbacks.onRelease(midiKey);
    }
    this.pendingGlissando.clear();

    for (const midiKey of this.activePointers.values()) {
      this.callbacks.onRelease(midiKey);
    }
    this.activePointers.clear();
  }

  private clearPendingGlissando(): void {
    for (const [timerId, midiKey] of this.pendingGlissando.entries()) {
      window.clearTimeout(timerId);
      this.callbacks.onRelease(midiKey);
    }
    this.pendingGlissando.clear();
  }

  destroy(): void {
    this.clear();
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
  }

  private attachListeners(): void {
    const onPointerDown = (e: PointerEvent) => {
      // Only primary mouse button or touch/pen
      if (e.pointerType === 'mouse' && e.button !== 0 && e.buttons !== 1) {
        return;
      }
      const keyEl = this.findKeyElement(e.target as Element | null, e.clientX, e.clientY);
      if (!keyEl) {
        return;
      }
      const midi = this.extractMidiKey(keyEl);
      if (midi === null) {
        return;
      }

      const vel = this.computePointerVelocity(keyEl, e.clientX, e.clientY);
      this.activePointers.set(e.pointerId, midi);
      this.callbacks.onPress(midi, vel);
    };

    const onPointerMove = (e: PointerEvent) => {
      // If this pointer was not tracked and no buttons are down, skip
      if (!this.activePointers.has(e.pointerId) && e.buttons !== 1) {
        return;
      }

      const keyEl = this.findKeyElement(null, e.clientX, e.clientY);
      const currentMidi = this.activePointers.get(e.pointerId);

      if (!keyEl) {
        // Pointer moved outside the keyboard (e.g. off-canvas)
        if (currentMidi !== undefined) {
          this.callbacks.onRelease(currentMidi);
          this.activePointers.delete(e.pointerId);
        }
        return;
      }

      const nextMidi = this.extractMidiKey(keyEl);
      if (nextMidi === null) {
        if (currentMidi !== undefined) {
          this.callbacks.onRelease(currentMidi);
          this.activePointers.delete(e.pointerId);
        }
        return;
      }

      if (currentMidi === undefined) {
        // Dragged into keyboard with button held
        const vel = this.computePointerVelocity(keyEl, e.clientX, e.clientY);
        this.activePointers.set(e.pointerId, nextMidi);
        this.callbacks.onPress(nextMidi, vel);
      } else if (currentMidi !== nextMidi) {
        // Glissando transition to adjacent key: maintain voice polyphony during drag
        const vel = this.computePointerVelocity(keyEl, e.clientX, e.clientY);
        const timerId = window.setTimeout(() => {
          this.pendingGlissando.delete(timerId);
          this.callbacks.onRelease(currentMidi);
        }, GLISSANDO_HOLD_MS);
        this.pendingGlissando.set(timerId, currentMidi);
        this.activePointers.set(e.pointerId, nextMidi);
        this.callbacks.onPress(nextMidi, vel);
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      this.clearPendingGlissando();
      let midi = this.activePointers.get(e.pointerId);
      if (midi === undefined) {
        const keyEl = this.findKeyElement(e.target as Element | null, e.clientX, e.clientY);
        if (keyEl) {
          const targetMidi = this.extractMidiKey(keyEl);
          if (targetMidi !== null) {
            midi = targetMidi;
          }
        }
      }
      if (midi !== undefined) {
        this.callbacks.onRelease(midi);
        this.activePointers.delete(e.pointerId);
      }
    };

    const onPointerCancel = (e: PointerEvent) => {
      this.clearPendingGlissando();
      let midi = this.activePointers.get(e.pointerId);
      if (midi === undefined) {
        const keyEl = this.findKeyElement(e.target as Element | null, e.clientX, e.clientY);
        if (keyEl) {
          const targetMidi = this.extractMidiKey(keyEl);
          if (targetMidi !== null) {
            midi = targetMidi;
          }
        }
      }
      if (midi !== undefined) {
        this.callbacks.onRelease(midi);
        this.activePointers.delete(e.pointerId);
      }
    };

    const onWindowBlur = () => {
      this.clear();
    };

    this.container.addEventListener('pointerdown', onPointerDown, { capture: true });
    this.container.addEventListener('pointerup', onPointerUp, { capture: true });
    this.container.addEventListener('pointercancel', onPointerCancel, { capture: true });

    window.addEventListener('pointermove', onPointerMove, { passive: true, capture: true });
    window.addEventListener('pointerup', onPointerUp, { passive: true, capture: true });
    window.addEventListener('pointercancel', onPointerCancel, { passive: true, capture: true });
    window.addEventListener('blur', onWindowBlur);

    this.cleanupHandlers.push(() => {
      this.container.removeEventListener('pointerdown', onPointerDown, { capture: true });
      this.container.removeEventListener('pointerup', onPointerUp, { capture: true });
      this.container.removeEventListener('pointercancel', onPointerCancel, { capture: true });
      window.removeEventListener('pointermove', onPointerMove, { capture: true });
      window.removeEventListener('pointerup', onPointerUp, { capture: true });
      window.removeEventListener('pointercancel', onPointerCancel, { capture: true });
      window.removeEventListener('blur', onWindowBlur);
    });
  }

  private findKeyElement(target: Element | null, clientX: number, clientY: number): HTMLElement | null {
    if (target && target.closest) {
      const key = target.closest<HTMLElement>('.piano-key');
      if (key && this.container.contains(key)) {
        return key;
      }
    }
    if (typeof document !== 'undefined' && typeof document.elementFromPoint === 'function') {
      const el = document.elementFromPoint(clientX, clientY);
      if (el) {
        const key = el.closest<HTMLElement>('.piano-key');
        if (key && this.container.contains(key)) {
          return key;
        }
      }
    }
    return null;
  }

  private computePointerVelocity(
    keyEl: HTMLElement,
    clientX: number,
    clientY: number,
  ): number | undefined {
    // Preserve undefined on synthetic dispatchEvent({ buttons: 1 }) where clientX/Y === 0.
    if (clientX === 0 && clientY === 0) {
      return undefined;
    }
    const rect = keyEl.getBoundingClientRect();
    if (!rect || rect.height <= 0) {
      return undefined;
    }
    const yNorm = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
    const base = this.callbacks.getBaseVelocity ? this.callbacks.getBaseVelocity() : 100;
    return Math.max(1, Math.min(127, Math.round(base * (0.45 + 0.80 * yNorm))));
  }

  private extractMidiKey(el: HTMLElement): number | null {
    const raw = el.dataset.midi;
    if (!raw) {
      return null;
    }
    const midi = parseInt(raw, 10);
    return Number.isFinite(midi) && midi >= 21 && midi <= 108 ? midi : null;
  }
}
