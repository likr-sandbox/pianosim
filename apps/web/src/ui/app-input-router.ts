/** Routes note, pedal, PC keyboard, Web MIDI, and A4 pulse inputs for AppShell. */

import {
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  u7ToU16,
  type PedalKind,
} from '../audio/abi';
import type { WasmAudioHost } from '../audio/wasm-host';
import { PcKeyboard } from '../input/pc-keyboard';
import { PedalController } from '../input/pedal-controller';
import { applyTouchCurve, DEFAULT_TOUCH_CURVE, type TouchCurveMode } from '../input/touch-curve';
import type { WebMidiManager } from '../midi/web-midi';
import type { KeyboardView } from './keyboard-view';

const A4_MIDI = 69;
const A4_TEST_DURATION_MS = 2000;

export interface InputRouterHooks {
  readonly ensureUnlocked: () => Promise<void>;
  readonly getKeyboardView: () => KeyboardView | undefined;
  readonly onPedalVisual: (sustain: number, sostenuto: boolean, unaCorda: number) => void;
  readonly onDevicesChanged: (devices: Array<{ id: string; name: string }>) => void;
}

export class AppInputRouter {
  readonly pedalController: PedalController;
  readonly pcKeyboard: PcKeyboard;
  private readonly host: WasmAudioHost;
  private readonly midiManager: WebMidiManager;
  private readonly hooks: InputRouterHooks;

  private velocity7 = 100;
  private touchCurve: TouchCurveMode = DEFAULT_TOUCH_CURVE;
  private a4PulseTimer: number | null = null;
  private a4PulseGen = 0;

  constructor(host: WasmAudioHost, midiManager: WebMidiManager, hooks: InputRouterHooks) {
    this.host = host;
    this.midiManager = midiManager;
    this.hooks = hooks;

    this.pedalController = new PedalController({
      onPedalChange: (kind: PedalKind, valU16: number) => {
        void this.hooks.ensureUnlocked();
        this.host.pedal(kind, valU16, 0);
      },
      onStateUpdate: (st) => {
        this.hooks.onPedalVisual(st.sustain, st.sostenuto, st.unaCorda);
      },
    });

    this.midiManager.setCallbacks({
      onNotePress: (midiKey) => this.hooks.getKeyboardView()?.setKeyActive(midiKey, true),
      onNoteRelease: (midiKey) => this.hooks.getKeyboardView()?.setKeyActive(midiKey, false),
      onPedalChange: (pedalKind, value7) => this.syncPedalFromExternal(pedalKind, value7),
      onDevicesChanged: (devices) => this.hooks.onDevicesChanged(devices),
      onMidiActivity: () => {
        void this.hooks.ensureUnlocked();
      },
    });

    this.pcKeyboard = new PcKeyboard({
      onPress: (midi) => this.acquireNote(midi),
      onRelease: (midi) => this.releaseNote(midi),
      onToggleSustain: () => this.pedalController.toggleSustain(),
    });
  }

  getBaseVelocity(): number {
    return this.velocity7;
  }

  setBaseVelocity(v: number): number {
    this.velocity7 = Number.isFinite(v) ? Math.max(1, Math.min(127, Math.trunc(v))) : 100;
    return this.velocity7;
  }

  setTouchCurve(mode: TouchCurveMode): void {
    this.touchCurve = mode;
    this.midiManager.setTouchCurve(mode);
  }

  acquireNote(midiKey: number, velocity7?: number): void {
    void this.hooks.ensureUnlocked();
    const rawVel = velocity7 ?? this.velocity7;
    const curvedVel = applyTouchCurve(rawVel, this.touchCurve);
    this.host.noteOn(midiKey, u7ToU16(curvedVel), 0);
    this.hooks.getKeyboardView()?.setKeyActive(midiKey, true);
  }

  releaseNote(midiKey: number): void {
    this.host.noteOff(midiKey, 65535, 0);
    this.hooks.getKeyboardView()?.setKeyActive(midiKey, false);
  }

  syncPedalFromExternal(pedalKind: number, value7: number): void {
    const cc =
      pedalKind === PEDAL_SUSTAIN
        ? 64
        : pedalKind === PEDAL_SOSTENUTO
          ? 66
          : pedalKind === PEDAL_UNA_CORDA
            ? 67
            : -1;
    if (cc >= 0) {
      this.pedalController.handleMidiCc(cc, value7, false);
    }
  }

  triggerA4Pulse(): void {
    this.clearA4Pulse();
    const gen = ++this.a4PulseGen;
    const unlockP = this.hooks.ensureUnlocked();
    this.acquireNote(A4_MIDI, 100);
    void unlockP.then(() => {
      if (this.a4PulseGen !== gen) return;
      this.a4PulseTimer = window.setTimeout(() => {
        if (this.a4PulseGen === gen) {
          this.releaseNote(A4_MIDI);
          this.a4PulseTimer = null;
        }
      }, A4_TEST_DURATION_MS);
    });
  }

  clearA4Pulse(): void {
    this.a4PulseGen++;
    if (this.a4PulseTimer !== null) {
      window.clearTimeout(this.a4PulseTimer);
      this.a4PulseTimer = null;
      this.releaseNote(A4_MIDI);
    }
  }

  resetAll(): void {
    this.clearA4Pulse();
    this.pedalController.reset();
  }

  destroy(): void {
    this.clearA4Pulse();
    this.pcKeyboard.destroy();
  }
}
