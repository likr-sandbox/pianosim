/** Responsive 88-key interactive piano keyboard spanning A0 (MIDI 21) to C8 (MIDI 108). */

import { PointerGlissando } from '../input/pointer-glissando';

export interface KeyboardViewCallbacks {
  onPress(midiKey: number, velocity?: number): void;
  onRelease(midiKey: number): void;
  getBaseVelocity?: () => number;
}

const IS_BLACK_KEY = [
  false, // 0: C
  true,  // 1: C#
  false, // 2: D
  true,  // 3: D#
  false, // 4: E
  false, // 5: F
  true,  // 6: F#
  false, // 7: G
  true,  // 8: G#
  false, // 9: A
  true,  // 10: A#
  false, // 11: B
];

export class KeyboardView {
  private readonly container: HTMLElement;
  private readonly callbacks: KeyboardViewCallbacks;
  private readonly keyElements = new Map<number, HTMLElement>();
  private pointerGlissando: PointerGlissando | null = null;

  constructor(container: HTMLElement, callbacks: KeyboardViewCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.buildDom();
    this.pointerGlissando = new PointerGlissando(this.container, {
      onPress: (midi, vel) => this.callbacks.onPress(midi, vel),
      onRelease: (midi) => this.callbacks.onRelease(midi),
      getBaseVelocity: this.callbacks.getBaseVelocity,
    });
  }

  setKeyActive(midiKey: number, active: boolean): void {
    const el = this.keyElements.get(midiKey);
    if (!el) {
      return;
    }
    if (active) {
      el.classList.add('is-active', 'active');
      el.setAttribute('aria-pressed', 'true');
    } else {
      el.classList.remove('is-active', 'active');
      el.setAttribute('aria-pressed', 'false');
    }
  }

  clearActiveKeys(): void {
    for (const el of this.keyElements.values()) {
      el.classList.remove('is-active', 'active');
      el.setAttribute('aria-pressed', 'false');
    }
    this.pointerGlissando?.clear();
  }

  destroy(): void {
    this.pointerGlissando?.destroy();
    this.pointerGlissando = null;
    this.keyElements.clear();
    this.container.innerHTML = '';
  }

  private buildDom(): void {
    this.container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.className = 'piano-keyboard';
    wrapper.setAttribute('role', 'group');
    wrapper.setAttribute('aria-label', '88-Key Piano Keyboard (A0 to C8)');

    // 88 keys: MIDI 21 (A0) to 108 (C8)
    // 52 white keys, 36 black keys.
    // White keys form the baseline layout; black keys are positioned proportionally.
    let whiteKeyIndex = 0;
    const keyData: Array<{ midi: number; isBlack: boolean; whiteIndex: number }> = [];

    for (let midi = 21; midi <= 108; midi++) {
      const isBlack = IS_BLACK_KEY[midi % 12];
      if (isBlack) {
        keyData.push({ midi, isBlack: true, whiteIndex: whiteKeyIndex });
      } else {
        keyData.push({ midi, isBlack: false, whiteIndex: whiteKeyIndex });
        whiteKeyIndex++;
      }
    }

    const totalWhiteKeys = 52;

    for (const item of keyData) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.dataset.midi = String(item.midi);
      btn.setAttribute('aria-pressed', 'false');

      if (item.isBlack) {
        // Black key: must have BOTH .key-black AND .black-key for E2E selector satisfaction
        btn.className = 'piano-key key-black black-key';
        // Position relative to preceding white key
        const leftPercent = ((item.whiteIndex - 0.3) / totalWhiteKeys) * 100;
        btn.style.left = `${leftPercent}%`;
        btn.setAttribute('aria-label', `Key ${item.midi}`);
      } else {
        // White key: must have BOTH .key-white AND .white-key for E2E selector satisfaction
        btn.className = 'piano-key key-white white-key';
        const leftPercent = (item.whiteIndex / totalWhiteKeys) * 100;
        const widthPercent = (1 / totalWhiteKeys) * 100;
        btn.style.left = `${leftPercent}%`;
        btn.style.width = `${widthPercent}%`;
        btn.setAttribute('aria-label', `Key ${item.midi}`);
      }

      this.keyElements.set(item.midi, btn);
      wrapper.appendChild(btn);
    }

    this.container.appendChild(wrapper);
  }
}
