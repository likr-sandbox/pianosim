/** Root application shell orchestrating audio host, MIDI player, input devices, and UI panels. */

import {
  u7ToU16,
  type LidState,
  type ListenerPerspective,
  type PedalKind,
} from '../audio/abi';
import type { TelemetryClient } from '../audio/telemetry-client';
import type { WasmAudioHost } from '../audio/wasm-host';
import { PcKeyboard } from '../input/pc-keyboard';
import { PedalController } from '../input/pedal-controller';
import { getDemoScore } from '../midi/demo-scores';
import { MidiPlayer } from '../midi/midi-player';
import { parseSmf } from '../midi/smf-parser';
import { WebMidiManager } from '../midi/web-midi';
import { KeyboardView } from './keyboard-view';
import { PedalBar } from './pedal-bar';
import { SpacePanel } from './space-panel';
import { TransportBar } from './transport-bar';
import { VoicingPanel, type VoicingState } from './voicing-panel';

const A4_MIDI = 69;
const A4_TEST_DURATION_MS = 750;

export class AppShell {
  private readonly root: HTMLElement;
  private readonly host: WasmAudioHost;
  private readonly telemetry: TelemetryClient;
  private readonly midiManager: WebMidiManager;
  private readonly midiPlayer: MidiPlayer;
  private readonly pedalController: PedalController;

  private keyboardView!: KeyboardView;
  private pedalBar!: PedalBar;
  private voicingPanel!: VoicingPanel;
  private spacePanel!: SpacePanel;
  private transportBar!: TransportBar;
  private pcKeyboard!: PcKeyboard;

  private unlockBtn!: HTMLButtonElement;
  private a4Btn!: HTMLButtonElement;
  private panicBtn!: HTMLButtonElement;
  private velocitySlider!: HTMLInputElement;
  private velocityValEl!: HTMLElement;
  private errorBannerEl!: HTMLElement;

  private velocity7 = 100;
  private a4PulseTimer: number | null = null;
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
    this.voicingPanel?.destroy();
    this.spacePanel?.destroy();
    this.transportBar?.destroy();
    this.pcKeyboard?.destroy();
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.root.innerHTML = '';
  }

  private render(): void {
    this.root.innerHTML = `
      <main class="studio-shell">
        <header class="studio-header">
          <div class="header-titles">
            <p class="eyebrow">Milestone 7 &middot; Pure-Rust Wasm SIMD128 Physical Engine</p>
            <h1 class="studio-title">PianoSim Web Studio</h1>
          </div>
          <div class="header-actions">
            <button id="btn-unlock" data-testid="unlock-audio" type="button" class="btn btn-primary">
              Unlock Audio
            </button>
            <button id="btn-a4-trigger" data-testid="trigger-a4" type="button" class="btn btn-accent">
              A4 = 440 Hz
            </button>
            <button id="btn-panic" data-testid="panic-btn" type="button" class="btn btn-ghost">
              All Notes Off
            </button>
          </div>
        </header>

        <p id="error-banner" data-testid="error-banner" class="error-banner" hidden></p>

        <!-- Real-time Diagnostics HUD & Spectrum Canvas Slot -->
        <section class="viz-section" aria-label="Audio Visualizer and Telemetry HUD">
          <div id="hud-mount" class="hud-slot"></div>
          <div class="canvas-card">
            <canvas
              id="spectrum-canvas"
              data-testid="spectrum-canvas"
              class="spectrum-canvas"
              width="600"
              height="120"
            ></canvas>
          </div>
        </section>

        <!-- Velocity & Performance Control Bar -->
        <section class="performance-controls" aria-label="Input Velocity and Pedals">
          <label class="velocity-control" for="velocity-slider">
            <span class="control-label">Strike Velocity</span>
            <input
              id="velocity-slider"
              data-testid="velocity-slider"
              type="range"
              min="1"
              max="127"
              value="100"
              class="velocity-slider"
            />
            <span id="velocity-value" class="val-mono">100</span>
          </label>
          <div id="pedal-mount" class="pedal-slot"></div>
        </section>

        <!-- Responsive 88-Key Piano Keyboard -->
        <section class="keyboard-stage" aria-label="88-Key Concert Grand Piano Keyboard">
          <div id="keyboard-mount" class="keyboard-mount"></div>
        </section>

        <!-- MIDI Transport & Scoring -->
        <div id="transport-mount" class="transport-slot"></div>

        <!-- Voicing & Space Panels -->
        <div class="studio-panels-row">
          <div id="voicing-mount" class="panel-slot"></div>
          <div id="space-mount" class="panel-slot"></div>
        </div>
      </main>
    `;

    this.unlockBtn = this.root.querySelector('#btn-unlock')!;
    this.a4Btn = this.root.querySelector('#btn-a4-trigger')!;
    this.panicBtn = this.root.querySelector('#btn-panic')!;
    this.velocitySlider = this.root.querySelector('#velocity-slider')!;
    this.velocityValEl = this.root.querySelector('#velocity-value')!;
    this.errorBannerEl = this.root.querySelector('#error-banner')!;
  }

  private bindComponents(): void {
    // 88-Key Piano Keyboard
    const keyboardContainer = this.root.querySelector<HTMLElement>('#keyboard-mount')!;
    this.keyboardView = new KeyboardView(keyboardContainer, {
      onPress: (midi) => this.acquireNote(midi),
      onRelease: (midi) => this.releaseNote(midi),
    });

    // Three-pedal Bar
    const pedalContainer = this.root.querySelector<HTMLElement>('#pedal-mount')!;
    this.pedalBar = new PedalBar(pedalContainer, this.pedalController);

    // Voicing Panel
    const voicingContainer = this.root.querySelector<HTMLElement>('#voicing-mount')!;
    this.voicingPanel = new VoicingPanel(voicingContainer, (st: VoicingState) => {
      this.host.setVoicing(
        st.hammerHardness,
        st.unisonDetune,
        st.stretchTuning,
        st.stringLengthScale,
      );
    });

    // Space Panel
    const spaceContainer = this.root.querySelector<HTMLElement>('#space-mount')!;
    this.spacePanel = new SpacePanel(spaceContainer, (persp: ListenerPerspective, lid: LidState) => {
      this.host.setSpace(persp, lid);
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

  private acquireNote(midiKey: number, velocity7?: number): void {
    const vel = velocity7 ?? this.velocity7;
    const u16 = u7ToU16(vel);
    this.host.noteOn(midiKey, u16, 0);
    this.keyboardView.setKeyActive(midiKey, true);
  }

  private releaseNote(midiKey: number): void {
    this.host.noteOff(midiKey, 0, 0);
    this.keyboardView.setKeyActive(midiKey, false);
  }

  private triggerA4Pulse(): void {
    this.clearA4Pulse();
    this.acquireNote(A4_MIDI, 100);
    this.a4PulseTimer = window.setTimeout(() => {
      this.releaseNote(A4_MIDI);
      this.a4PulseTimer = null;
    }, A4_TEST_DURATION_MS);
  }

  private clearA4Pulse(): void {
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

  getHost(): WasmAudioHost {
    return this.host;
  }

  getTelemetry(): TelemetryClient {
    return this.telemetry;
  }
}
