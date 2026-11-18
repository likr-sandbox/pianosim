/** Root application shell orchestrating audio host, MIDI player, input devices, and UI panels. */

import {
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  u7ToU16,
  type BodyId,
  type LidState,
  type ListenerPerspective,
  type PedalKind,
} from '../audio/abi';
import type { TelemetryClient } from '../audio/telemetry-client';
import type { WasmAudioHost } from '../audio/wasm-host';
import { PcKeyboard } from '../input/pc-keyboard';
import { PedalController } from '../input/pedal-controller';
import {
  applyTouchCurve,
  DEFAULT_TOUCH_CURVE,
  type TouchCurveMode,
} from '../input/touch-curve';
import { getDemoScore } from '../midi/demo-scores';
import { MidiPlayer } from '../midi/midi-player';
import { parseSmf } from '../midi/smf-parser';
import { WebMidiManager } from '../midi/web-midi';
import { PerfMeter } from '../viz/perf-meter';
import { SpectrumCanvas } from '../viz/spectrum-canvas';
import { KeyboardView } from './keyboard-view';
import { PedalBar } from './pedal-bar';
import { PresetBar } from './preset-bar';
import { STUDIO_SHELL_HTML } from './shell-template';
import { SpacePanel } from './space-panel';
import { TouchCurveSelector } from './touch-curve-selector';
import { TransportBar } from './transport-bar';
import { VoicingPanel, type VoicingState } from './voicing-panel';

const A4_MIDI = 69;
const A4_TEST_DURATION_MS = 2000;

export class AppShell {
  private readonly root: HTMLElement;
  private readonly host: WasmAudioHost;
  private readonly telemetry: TelemetryClient;
  private readonly midiManager: WebMidiManager;
  private readonly midiPlayer: MidiPlayer;
  private readonly pedalController: PedalController;

  private keyboardView!: KeyboardView;
  private pedalBar!: PedalBar;
  private touchCurveSelector!: TouchCurveSelector;
  private presetBar!: PresetBar;
  private voicingPanel!: VoicingPanel;
  private spacePanel!: SpacePanel;
  private transportBar!: TransportBar;
  private pcKeyboard!: PcKeyboard;
  private perfMeter!: PerfMeter;
  private spectrumCanvas!: SpectrumCanvas;

  private unlockBtn!: HTMLButtonElement;
  private a4Btn!: HTMLButtonElement;
  private panicBtn!: HTMLButtonElement;
  private velocitySlider!: HTMLInputElement;
  private velocityValEl!: HTMLElement;
  private errorBannerEl!: HTMLElement;

  private velocity7 = 100;
  private touchCurve: TouchCurveMode = DEFAULT_TOUCH_CURVE;
  private a4PulseTimer: number | null = null;
  private a4PulseGen = 0;
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(
    root: HTMLElement,
    host: WasmAudioHost,
    telemetry: TelemetryClient,
    midiManager?: WebMidiManager,
    midiPlayer?: MidiPlayer,
  ) {
    this.root = root;
    this.host = host;
    this.telemetry = telemetry;
    this.midiManager = midiManager ?? new WebMidiManager(host);
    this.midiPlayer = midiPlayer ?? new MidiPlayer(host);

    this.pedalController = new PedalController({
      onPedalChange: (kind: PedalKind, valU16: number) => {
        this.ensureUnlocked();
        this.host.pedal(kind, valU16, 0);
      },
      onStateUpdate: (st) => {
        this.pedalBar?.updateState(st.sustain, st.sostenuto, st.unaCorda);
      },
    });

    this.render();
    this.bindComponents();
    this.bindControls();

    // Load initial Bach Prelude demo score
    const initialScore = getDemoScore('bach-prelude');
    this.midiPlayer.load(initialScore);
  }

  showError(message: string): void {
    if (this.errorBannerEl) {
      this.errorBannerEl.hidden = false;
      this.errorBannerEl.textContent = `Error: ${message}`;
    }
  }

  clearError(): void {
    if (this.errorBannerEl) {
      this.errorBannerEl.hidden = true;
      this.errorBannerEl.textContent = '';
    }
  }

  destroy(): void {
    this.clearA4Pulse();
    this.keyboardView?.destroy();
    this.pedalBar?.destroy();
    this.touchCurveSelector?.destroy();
    this.presetBar?.destroy();
    this.voicingPanel?.destroy();
    this.spacePanel?.destroy();
    this.transportBar?.destroy();
    this.pcKeyboard?.destroy();
    this.perfMeter?.destroy();
    this.spectrumCanvas?.destroy();
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.root.innerHTML = '';
  }

  private render(): void {
    this.root.innerHTML = STUDIO_SHELL_HTML;

    this.unlockBtn = this.root.querySelector('#btn-unlock')!;
    this.a4Btn = this.root.querySelector('#btn-a4-trigger')!;
    this.panicBtn = this.root.querySelector('#btn-panic')!;
    this.velocitySlider = this.root.querySelector('#velocity-slider')!;
    this.velocityValEl = this.root.querySelector('#velocity-value')!;
    this.errorBannerEl = this.root.querySelector('#error-banner')!;
  }

  private updateA4ButtonLabel(a4FreqHz: number): void {
    this.a4Btn.textContent = `A4 = ${a4FreqHz.toFixed(1)} Hz`;
  }

  private bindComponents(): void {
    // Telemetry HUD PerfMeter
    const hudContainer = this.root.querySelector<HTMLElement>('#hud-mount')!;
    this.perfMeter = new PerfMeter(hudContainer, this.telemetry);

    // Spectrum Visualizer Canvas
    const canvasEl = this.root.querySelector<HTMLCanvasElement>('#spectrum-canvas')!;
    this.spectrumCanvas = new SpectrumCanvas(canvasEl, () => this.host.getAnalyser());

    // 88-Key Piano Keyboard
    const keyboardContainer = this.root.querySelector<HTMLElement>('#keyboard-mount')!;
    this.keyboardView = new KeyboardView(keyboardContainer, {
      onPress: (midi) => this.acquireNote(midi),
      onRelease: (midi) => this.releaseNote(midi),
    });

    // Touch Curve Selector
    const touchCurveContainer = this.root.querySelector<HTMLElement>('#touch-curve-mount')!;
    this.touchCurveSelector = new TouchCurveSelector(touchCurveContainer, (mode) => {
      this.touchCurve = mode;
      this.midiManager.setTouchCurve(mode);
    });

    // Three-pedal Bar
    const pedalContainer = this.root.querySelector<HTMLElement>('#pedal-mount')!;
    this.pedalBar = new PedalBar(pedalContainer, this.pedalController);

    // Voicing Panel
    const voicingContainer = this.root.querySelector<HTMLElement>('#voicing-mount')!;
    this.voicingPanel = new VoicingPanel(voicingContainer, (st: VoicingState) => {
      this.ensureUnlocked();
      this.presetBar?.setCustom();
      this.updateA4ButtonLabel(st.a4FreqHz);
      this.host.setVoicing(
        st.hammerHardness,
        st.unisonDetune,
        st.stretchTuning,
        st.stringLengthScale,
        st.a4FreqHz,
      );
    });

    // Space Panel
    const spaceContainer = this.root.querySelector<HTMLElement>('#space-mount')!;
    this.spacePanel = new SpacePanel(spaceContainer, {
      onSpaceChange: (persp: ListenerPerspective, lid: LidState, bodyId: BodyId) => {
        this.ensureUnlocked();
        this.presetBar?.setCustom();
        this.host.setSpace(persp, lid, bodyId);
      },
      onReverbChange: (decayS: number, mix: number) => {
        this.ensureUnlocked();
        this.presetBar?.setCustom();
        this.host.setReverb(decayS, mix);
      },
    });

    // Parameter Preset Bar
    const presetContainer = this.root.querySelector<HTMLElement>('#preset-mount')!;
    this.presetBar = new PresetBar(presetContainer, {
      onSelectPreset: (preset) => {
        this.ensureUnlocked();
        this.voicingPanel.setState(preset.voicing);
        this.updateA4ButtonLabel(preset.voicing.a4FreqHz);
        this.spacePanel.setState(preset.space);
        this.host.setVoicing(
          preset.voicing.hammerHardness,
          preset.voicing.unisonDetune,
          preset.voicing.stretchTuning,
          preset.voicing.stringLengthScale,
          preset.voicing.a4FreqHz,
        );
        this.host.setSpace(
          preset.space.perspective,
          preset.space.lidState,
          preset.space.bodyId,
        );
        this.host.setReverb(preset.space.reverbDecayS, preset.space.reverbMix);
      },
    });

    // Wire MIDI Player feedback callbacks for keyboard animation, pedals, and seek progress
    this.midiPlayer.setCallbacks({
      onNotePress: (midiKey) => {
        this.keyboardView?.setKeyActive(midiKey, true);
      },
      onNoteRelease: (midiKey) => {
        this.keyboardView?.setKeyActive(midiKey, false);
      },
      onPedalChange: (pedalKind, value7) => {
        this.syncPedalFromExternal(pedalKind, value7);
      },
      onProgress: (currentSec, totalSec) => {
        this.transportBar?.updateProgress(currentSec, totalSec);
      },
      onStateChange: (isPlaying) => {
        if (!isPlaying) {
          this.keyboardView?.clearActiveKeys();
        }
      },
    });

    // Transport Bar
    const transportContainer = this.root.querySelector<HTMLElement>('#transport-mount')!;
    this.transportBar = new TransportBar(
      transportContainer,
      this.midiPlayer,
      this.midiManager,
      {
        onSelectDemo: (demoId) => {
          this.clearError();
          const score = getDemoScore(demoId);
          this.midiPlayer.load(score);
        },
        onUploadMidi: (bytes) => {
          this.clearError();
          try {
            const score = parseSmf(bytes);
            this.midiPlayer.load(score);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            this.showError(msg);
          }
        },
      },
    );

    // Wire live Web MIDI callbacks bidirectionally to keyboard, pedals, devices, and auto-unlock
    this.midiManager.setCallbacks({
      onNotePress: (midiKey) => {
        this.keyboardView?.setKeyActive(midiKey, true);
      },
      onNoteRelease: (midiKey) => {
        this.keyboardView?.setKeyActive(midiKey, false);
      },
      onPedalChange: (pedalKind, value7) => {
        this.syncPedalFromExternal(pedalKind, value7);
      },
      onDevicesChanged: (devices) => {
        this.transportBar?.updateDevices(devices);
      },
      onMidiActivity: () => {
        void this.ensureUnlocked();
      },
    });
    this.transportBar.updateDevices(this.midiManager.getDevices());

    // PC Keyboard
    this.pcKeyboard = new PcKeyboard({
      onPress: (midi) => this.acquireNote(midi),
      onRelease: (midi) => this.releaseNote(midi),
      onToggleSustain: () => this.pedalController.toggleSustain(),
    });
  }

  private bindControls(): void {
    const onUnlockClick = async () => {
      await this.host.unlock();
      await this.midiManager.refresh();
      this.unlockBtn.classList.add('is-ready');
    };

    const onA4Click = () => {
      this.triggerA4Pulse();
    };

    const onPanicClick = () => {
      this.panicAll();
    };

    const onVelocityInput = () => {
      const v = parseInt(this.velocitySlider.value, 10);
      this.velocity7 = Number.isFinite(v) ? Math.max(1, Math.min(127, v)) : 100;
      this.velocityValEl.textContent = String(this.velocity7);
    };

    const onErrorClick = () => {
      this.clearError();
    };

    this.unlockBtn.addEventListener('click', onUnlockClick);
    this.a4Btn.addEventListener('click', onA4Click);
    this.panicBtn.addEventListener('click', onPanicClick);
    this.velocitySlider.addEventListener('input', onVelocityInput);
    this.errorBannerEl.addEventListener('click', onErrorClick);

    this.cleanupHandlers.push(() => {
      this.unlockBtn.removeEventListener('click', onUnlockClick);
      this.a4Btn.removeEventListener('click', onA4Click);
      this.panicBtn.removeEventListener('click', onPanicClick);
      this.velocitySlider.removeEventListener('input', onVelocityInput);
      this.errorBannerEl.removeEventListener('click', onErrorClick);
    });
  }

  private ensureUnlocked(): Promise<void> {
    if (this.host.isReady()) {
      return Promise.resolve();
    }
    return this.host
      .unlock()
      .then(async () => {
        await this.midiManager.refresh();
        this.unlockBtn.classList.add('is-ready');
      })
      .catch((err) => {
        const msg = err instanceof Error ? err.message : String(err);
        this.showError(msg);
      });
  }

  private syncPedalFromExternal(pedalKind: number, value7: number): void {
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

  private acquireNote(midiKey: number, velocity7?: number): void {
    void this.ensureUnlocked();
    const rawVel = velocity7 ?? this.velocity7;
    const curvedVel = applyTouchCurve(rawVel, this.touchCurve);
    const u16 = u7ToU16(curvedVel);
    this.host.noteOn(midiKey, u16, 0);
    this.keyboardView.setKeyActive(midiKey, true);
  }

  private releaseNote(midiKey: number): void {
    this.host.noteOff(midiKey, 65535, 0);
    this.keyboardView.setKeyActive(midiKey, false);
  }

  private triggerA4Pulse(): void {
    this.clearA4Pulse();
    const gen = ++this.a4PulseGen;
    const unlockP = this.ensureUnlocked();
    this.acquireNote(A4_MIDI, 100);
    void unlockP.then(() => {
      if (this.a4PulseGen !== gen) {
        return;
      }
      this.a4PulseTimer = window.setTimeout(() => {
        if (this.a4PulseGen === gen) {
          this.releaseNote(A4_MIDI);
          this.a4PulseTimer = null;
        }
      }, A4_TEST_DURATION_MS);
    });
  }

  private clearA4Pulse(): void {
    this.a4PulseGen++;
    if (this.a4PulseTimer !== null) {
      window.clearTimeout(this.a4PulseTimer);
      this.a4PulseTimer = null;
      this.releaseNote(A4_MIDI);
    }
  }

  private panicAll(): void {
    this.clearA4Pulse();
    this.midiPlayer.stop();
    this.host.allNotesOff();
    this.telemetry.clearClip();
    this.keyboardView.clearActiveKeys();
    this.pedalController.reset();
  }

  getKeyboardView(): KeyboardView {
    return this.keyboardView;
  }

  getPedalBar(): PedalBar {
    return this.pedalBar;
  }

  getTransportBar(): TransportBar {
    return this.transportBar;
  }

  getPerfMeter(): PerfMeter {
    return this.perfMeter;
  }

  getSpectrumCanvas(): SpectrumCanvas {
    return this.spectrumCanvas;
  }

  getHost(): WasmAudioHost {
    return this.host;
  }

  getTelemetry(): TelemetryClient {
    return this.telemetry;
  }
}
