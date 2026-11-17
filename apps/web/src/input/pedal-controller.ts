/** Normalizes Damper/Sostenuto/UnaCorda pedal state across UI sliders, buttons, spacebar, and MIDI CC. */

import {
  cc7ToU16,
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  type PedalKind,
} from '../audio/abi';

export interface PedalControllerCallbacks {
  onPedalChange(pedalKind: PedalKind, valueU16: number): void;
  onStateUpdate?(state: { sustain: number; sostenuto: boolean; unaCorda: number }): void;
}

export class PedalController {
  private readonly callbacks: PedalControllerCallbacks;

  private sustainVal = 0.0; // 0.0 to 1.0
  private sostenutoEngaged = false;
  private unaCordaVal = 0.0; // 0.0 to 1.0

  constructor(callbacks: PedalControllerCallbacks) {
    this.callbacks = callbacks;
  }

  getSustain(): number {
    return this.sustainVal;
  }

  isSostenuto(): boolean {
    return this.sostenutoEngaged;
  }

  getUnaCorda(): number {
    return this.unaCordaVal;
  }

  isUnaCorda(): boolean {
    return this.unaCordaVal > 0.0;
  }

  setSustain(normalized: number): void {
    const clamped = Number.isFinite(normalized) ? Math.max(0.0, Math.min(1.0, normalized)) : 0.0;
    this.sustainVal = clamped;
    const u16 = Math.trunc(clamped * 65535);
    this.callbacks.onPedalChange(PEDAL_SUSTAIN, u16);
    this.emitState();
  }

  toggleSustain(): boolean {
    const next = this.sustainVal > 0.0 ? 0.0 : 1.0;
    this.setSustain(next);
    return next > 0.0;
  }

  setSostenuto(latched: boolean): void {
    this.sostenutoEngaged = latched;
    const u16 = latched ? 65535 : 0;
    this.callbacks.onPedalChange(PEDAL_SOSTENUTO, u16);
    this.emitState();
  }

  toggleSostenuto(): boolean {
    this.setSostenuto(!this.sostenutoEngaged);
    return this.sostenutoEngaged;
  }

  setUnaCorda(value: number | boolean, emitHost = true): void {
    const raw = typeof value === 'boolean' ? (value ? 1.0 : 0.0) : value;
    const clamped = Number.isFinite(raw) ? Math.max(0.0, Math.min(1.0, raw)) : 0.0;
    this.unaCordaVal = clamped;
    if (emitHost) {
      this.callbacks.onPedalChange(PEDAL_UNA_CORDA, Math.trunc(clamped * 65535));
    }
    this.emitState();
  }

  toggleUnaCorda(): boolean {
    const next = this.unaCordaVal > 0.0 ? 0.0 : 1.0;
    this.setUnaCorda(next);
    return next > 0.0;
  }

  handleMidiCc(controller: number, value7: number, emitHost = true): void {
    const val = Number.isFinite(value7) ? Math.max(0, Math.min(127, Math.trunc(value7))) : 0;
    if (controller === 64) {
      this.sustainVal = val / 127;
      if (emitHost) {
        this.callbacks.onPedalChange(PEDAL_SUSTAIN, cc7ToU16(val));
      }
      this.emitState();
    } else if (controller === 66) {
      this.sostenutoEngaged = val >= 64;
      if (emitHost) {
        this.callbacks.onPedalChange(PEDAL_SOSTENUTO, this.sostenutoEngaged ? 65535 : 0);
      }
      this.emitState();
    } else if (controller === 67) {
      this.unaCordaVal = val / 127;
      if (emitHost) {
        this.callbacks.onPedalChange(PEDAL_UNA_CORDA, cc7ToU16(val));
      }
      this.emitState();
    }
  }

  reset(): void {
    this.setSustain(0.0);
    this.setSostenuto(false);
    this.setUnaCorda(0.0);
  }

  private emitState(): void {
    this.callbacks.onStateUpdate?.({
      sustain: this.sustainVal,
      sostenuto: this.sostenutoEngaged,
      unaCorda: this.unaCordaVal,
    });
  }
}
