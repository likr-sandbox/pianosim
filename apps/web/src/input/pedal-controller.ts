/** Normalizes Damper/Sostenuto/UnaCorda pedal state across UI sliders, buttons, spacebar, and MIDI CC. */

import {
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  type PedalKind,
} from '../audio/abi';

export interface PedalControllerCallbacks {
  onPedalChange(pedalKind: PedalKind, valueU16: number): void;
  onStateUpdate?(state: { sustain: number; sostenuto: boolean; unaCorda: boolean }): void;
}

export class PedalController {
  private readonly callbacks: PedalControllerCallbacks;

  private sustainVal = 0.0; // 0.0 to 1.0
  private sostenutoEngaged = false;
  private unaCordaEngaged = false;

  constructor(callbacks: PedalControllerCallbacks) {
    this.callbacks = callbacks;
  }

  getSustain(): number {
    return this.sustainVal;
  }

  isSostenuto(): boolean {
    return this.sostenutoEngaged;
  }

  isUnaCorda(): boolean {
    return this.unaCordaEngaged;
  }

  setSustain(normalized: number): void {
    const clamped = Math.max(0.0, Math.min(1.0, normalized));
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

  setUnaCorda(engaged: boolean): void {
    this.unaCordaEngaged = engaged;
    const u16 = engaged ? 65535 : 0;
    this.callbacks.onPedalChange(PEDAL_UNA_CORDA, u16);
    this.emitState();
  }

  toggleUnaCorda(): boolean {
    this.setUnaCorda(!this.unaCordaEngaged);
    return this.unaCordaEngaged;
  }

  handleMidiCc(controller: number, value7: number): void {
    const val = Math.max(0, Math.min(127, value7));
    if (controller === 64) {
      this.setSustain(val / 127);
    } else if (controller === 66) {
      this.setSostenuto(val >= 64);
    } else if (controller === 67) {
      this.setUnaCorda(val >= 64);
    }
  }

  reset(): void {
    this.setSustain(0.0);
    this.setSostenuto(false);
    this.setUnaCorda(false);
  }

  private emitState(): void {
    this.callbacks.onStateUpdate?.({
      sustain: this.sustainVal,
      sostenuto: this.sostenutoEngaged,
      unaCorda: this.unaCordaEngaged,
    });
  }
}
