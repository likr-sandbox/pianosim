/** Root application shell orchestrating audio host, MIDI player, input devices, and UI panels. */

import type { BodyId, LidState, ListenerPerspective } from '../audio/abi';
import type { TelemetryClient } from '../audio/telemetry-client';
import type { WasmAudioHost } from '../audio/wasm-host';
import { triggerWavDownload } from '../audio/wav-recorder';
import { fetchDemoScore } from '../midi/demo-scores';
import { MidiPlayer } from '../midi/midi-player';
import { parseSmf } from '../midi/smf-parser';
import { WebMidiManager } from '../midi/web-midi';
import { PerfMeter } from '../viz/perf-meter';
import { SpectrumCanvas } from '../viz/spectrum-canvas';
import { AppInputRouter } from './app-input-router';
import { KeyboardView } from './keyboard-view';
import { PedalBar } from './pedal-bar';
import { PerfPanel } from './perf-panel';
import { PresetBar } from './preset-bar';
import { RecorderControls } from './recorder-controls';
import { STUDIO_SHELL_HTML } from './shell-template';
import { SpacePanel } from './space-panel';
import { TouchCurveSelector } from './touch-curve-selector';
import { TransportBar } from './transport-bar';
import { VoicingPanel, type VoicingState } from './voicing-panel';

export class AppShell {
  private readonly root: HTMLElement;
  private readonly host: WasmAudioHost;
  private readonly telemetry: TelemetryClient;
  private readonly midiManager: WebMidiManager;
  private readonly midiPlayer: MidiPlayer;
  private inputRouter!: AppInputRouter;

  private keyboardView!: KeyboardView;
  private pedalBar!: PedalBar;
  private touchCurveSelector!: TouchCurveSelector;
  private presetBar!: PresetBar;
  private perfPanel!: PerfPanel;
  private voicingPanel!: VoicingPanel;
  private spacePanel!: SpacePanel;
  private transportBar!: TransportBar;
  private recorderControls!: RecorderControls;
  private perfMeter!: PerfMeter;
  private spectrumCanvas!: SpectrumCanvas;

  private unlockBtn!: HTMLButtonElement;
  private a4Btn!: HTMLButtonElement;
  private panicBtn!: HTMLButtonElement;
  private velocitySlider!: HTMLInputElement;
  private velocityValEl!: HTMLElement;
  private errorBannerEl!: HTMLElement;

  private demoLoadGen = 0;
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

    this.inputRouter = new AppInputRouter(this.host, this.midiManager, {
      ensureUnlocked: () => this.ensureUnlocked(),
      getKeyboardView: () => this.keyboardView,
      onPedalVisual: (sustain, sostenuto, unaCorda) => {
        this.pedalBar?.updateState(sustain, sostenuto, unaCorda);
      },
      onDevicesChanged: (devices) => {
        this.transportBar?.updateDevices(devices);
      },
    });

    this.render();
    this.bindComponents();
    this.bindControls();

    // Load initial Bach Prelude demo score
    void this.transportBar.triggerDemoLoad('bach-prelude');
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
    this.inputRouter.destroy();
    this.keyboardView?.destroy();
    this.pedalBar?.destroy();
    this.touchCurveSelector?.destroy();
    this.presetBar?.destroy();
    this.perfPanel?.destroy();
    this.voicingPanel?.destroy();
    this.spacePanel?.destroy();
    this.host.onRecordingUpdate(null);
    this.recorderControls?.destroy();
    this.transportBar?.destroy();
    this.perfMeter?.destroy();
    this.spectrumCanvas?.destroy();
    for (const off of this.cleanupHandlers) off();
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
    const hudContainer = this.root.querySelector<HTMLElement>('#hud-mount')!;
    this.perfMeter = new PerfMeter(hudContainer, this.telemetry);

    const canvasEl = this.root.querySelector<HTMLCanvasElement>('#spectrum-canvas')!;
    this.spectrumCanvas = new SpectrumCanvas(canvasEl, () => this.host.getAnalyser());

    const keyboardContainer = this.root.querySelector<HTMLElement>('#keyboard-mount')!;
    this.keyboardView = new KeyboardView(keyboardContainer, {
      onPress: (midi, vel) => this.inputRouter.acquireNote(midi, vel),
      onRelease: (midi) => this.inputRouter.releaseNote(midi),
      getBaseVelocity: () => this.inputRouter.getBaseVelocity(),
    });

    const touchCurveContainer = this.root.querySelector<HTMLElement>('#touch-curve-mount')!;
    this.touchCurveSelector = new TouchCurveSelector(touchCurveContainer, (mode) => {
      this.inputRouter.setTouchCurve(mode);
    });

    const pedalContainer = this.root.querySelector<HTMLElement>('#pedal-mount')!;
    this.pedalBar = new PedalBar(pedalContainer, this.inputRouter.pedalController);

    const voicingContainer = this.root.querySelector<HTMLElement>('#voicing-mount')!;
    this.voicingPanel = new VoicingPanel(voicingContainer, (st: VoicingState) => {
      void this.ensureUnlocked();
      this.presetBar?.setCustom();
      this.updateA4ButtonLabel(st.a4FreqHz);
      this.host.setVoicing(
        st.hammerHardness,
        st.unisonDetune,
        st.stretchTuning,
        st.stringLengthScale,
        st.a4FreqHz,
        st.temperamentId,
        st.condition,
      );
    });

    const spaceContainer = this.root.querySelector<HTMLElement>('#space-mount')!;
    this.spacePanel = new SpacePanel(spaceContainer, {
      onSpaceChange: (persp: ListenerPerspective, lid: LidState, bodyId: BodyId) => {
        void this.ensureUnlocked();
        this.presetBar?.setCustom();
        this.host.setSpace(persp, lid, bodyId);
      },
      onReverbChange: (decayS: number, mix: number) => {
        void this.ensureUnlocked();
        this.presetBar?.setCustom();
        this.host.setReverb(decayS, mix);
      },
    });

    const presetContainer = this.root.querySelector<HTMLElement>('#preset-mount')!;
    this.presetBar = new PresetBar(presetContainer, {
      onSelectPreset: (preset) => {
        void this.ensureUnlocked();
        this.voicingPanel.setState(preset.voicing);
        this.updateA4ButtonLabel(preset.voicing.a4FreqHz);
        this.spacePanel.setState(preset.space);
        this.host.setVoicing(
          preset.voicing.hammerHardness,
          preset.voicing.unisonDetune,
          preset.voicing.stretchTuning,
          preset.voicing.stringLengthScale,
          preset.voicing.a4FreqHz,
          preset.voicing.temperamentId,
          preset.voicing.condition,
        );
        this.host.setSpace(preset.space.perspective, preset.space.lidState, preset.space.bodyId);
        this.host.setReverb(preset.space.reverbDecayS, preset.space.reverbMix);
      },
    });

    const perfContainer = this.root.querySelector<HTMLElement>('#perf-mount')!;
    this.perfPanel = new PerfPanel(
      perfContainer,
      {
        onSelectPreset: (presetId) => {
          void this.ensureUnlocked();
          this.host.setPerfPreset(presetId);
        },
        onKnobChange: (key, value) => {
          void this.ensureUnlocked();
          this.host.setPerfKnob(key, value);
        },
      },
      this.host.getPerfPreset(),
      this.host.getPerfConfig(),
    );

    this.midiPlayer.setCallbacks({
      onNotePress: (midiKey) => this.keyboardView?.setKeyActive(midiKey, true),
      onNoteRelease: (midiKey) => this.keyboardView?.setKeyActive(midiKey, false),
      onPedalChange: (pedalKind, value7) => this.inputRouter.syncPedalFromExternal(pedalKind, value7),
      onProgress: (currentSec, totalSec) => this.transportBar?.updateProgress(currentSec, totalSec),
      onStateChange: (isPlaying) => {
        if (!isPlaying) this.keyboardView?.clearActiveKeys();
      },
    });

    const transportContainer = this.root.querySelector<HTMLElement>('#transport-mount')!;
    this.transportBar = new TransportBar(
      transportContainer,
      this.midiPlayer,
      this.midiManager,
      {
        onSelectDemo: (demoId) => this.loadDemo(demoId),
        onUploadMidi: (bytes) => {
          this.demoLoadGen++;
          this.clearError();
          try {
            const score = parseSmf(bytes);
            this.midiPlayer.load(score);
          } catch (err) {
            this.showError(err instanceof Error ? err.message : String(err));
          }
        },
      },
    );

    this.recorderControls = new RecorderControls(this.transportBar.getRecorderMount(), {
      onStartRecording: () => {
        void this.ensureUnlocked();
        this.host.startRecording();
      },
      onStopRecording: () => this.host.stopRecording(),
      onExportWav: (format) => {
        const blob = this.host.exportWavBlob(format);
        if (blob) triggerWavDownload(blob, format);
      },
    });
    this.host.onRecordingUpdate((frames, durationSec, recording) => {
      this.recorderControls.updateStatus(frames, durationSec, recording);
    });
    this.transportBar.updateDevices(this.midiManager.getDevices());
  }

  private bindControls(): void {
    const onUnlockClick = async () => {
      await this.host.unlock();
      await this.midiManager.refresh();
      this.unlockBtn.classList.add('is-ready');
    };
    const onA4Click = () => this.inputRouter.triggerA4Pulse();
    const onPanicClick = () => this.panicAll();
    const onVelocityInput = () => {
      const v = this.inputRouter.setBaseVelocity(parseInt(this.velocitySlider.value, 10));
      this.velocityValEl.textContent = String(v);
    };
    const onErrorClick = () => this.clearError();

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
    if (this.host.isReady()) return Promise.resolve();
    return this.host
      .unlock()
      .then(async () => {
        await this.midiManager.refresh();
        this.unlockBtn.classList.add('is-ready');
      })
      .catch((err) => {
        this.showError(err instanceof Error ? err.message : String(err));
      });
  }

  private async loadDemo(demoId: string): Promise<void> {
    const gen = ++this.demoLoadGen;
    this.clearError();
    this.midiPlayer.stop();
    try {
      const score = await fetchDemoScore(demoId);
      if (gen === this.demoLoadGen) this.midiPlayer.load(score);
    } catch (err) {
      if (gen === this.demoLoadGen) {
        this.showError(err instanceof Error ? err.message : String(err));
      }
    }
  }

  private panicAll(): void {
    this.inputRouter.resetAll();
    this.midiPlayer.stop();
    this.host.allNotesOff();
    this.telemetry.clearClip();
    this.keyboardView.clearActiveKeys();
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

  getPerfPanel(): PerfPanel {
    return this.perfPanel;
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
