/** Multi-touch and pointer glissando tracker for responsive 88-key piano keyboard. */

export interface PointerGlissandoCallbacks {
  onPress(midiKey: number): void;
  onRelease(midiKey: number): void;
}

export class PointerGlissando {
  private readonly container: HTMLElement;
  private readonly callbacks: PointerGlissandoCallbacks;
  private readonly activePointers = new Map<number, number>(); // pointerId -> midiKey
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, callbacks: PointerGlissandoCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.attachListeners();
  }

  clear(): void {
    for (const midiKey of this.activePointers.values()) {
      this.callbacks.onRelease(midiKey);
    }
    this.activePointers.clear();
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
      if (e.pointerType === 'mouse' && e.button !== 0) {
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

      this.activePointers.set(e.pointerId, midi);
      this.callbacks.onPress(midi);
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
        this.activePointers.set(e.pointerId, nextMidi);
        this.callbacks.onPress(nextMidi);
      } else if (currentMidi !== nextMidi) {
        // Glissando transition to adjacent key
        this.callbacks.onRelease(currentMidi);
        this.activePointers.set(e.pointerId, nextMidi);
        this.callbacks.onPress(nextMidi);
      }
    };

    const onPointerUp = (e: PointerEvent) => {
      const midi = this.activePointers.get(e.pointerId);
      if (midi !== undefined) {
        this.callbacks.onRelease(midi);
        this.activePointers.delete(e.pointerId);
      }
    };

    const onPointerCancel = (e: PointerEvent) => {
      const midi = this.activePointers.get(e.pointerId);
      if (midi !== undefined) {
        this.callbacks.onRelease(midi);
        this.activePointers.delete(e.pointerId);
      }
    };

    const onWindowBlur = () => {
      this.clear();
    };

    this.container.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove, { passive: true });
    window.addEventListener('pointerup', onPointerUp, { passive: true });
    window.addEventListener('pointercancel', onPointerCancel, { passive: true });
    window.addEventListener('blur', onWindowBlur);

    this.cleanupHandlers.push(() => {
      this.container.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerCancel);
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

  private extractMidiKey(el: HTMLElement): number | null {
    const raw = el.dataset.midi;
    if (!raw) {
      return null;
    }
    const midi = parseInt(raw, 10);
    return Number.isFinite(midi) && midi >= 21 && midi <= 108 ? midi : null;
  }
}
