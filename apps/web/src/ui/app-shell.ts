import { PEDAL_SUSTAIN, u7ToU16, type PianoDiagnostics } from '../audio/abi';
import { formatPeakDb, type TelemetryClient } from '../audio/telemetry-client';
import type { WasmAudioHost } from '../audio/wasm-host';
import { TestKeyboardWidget } from './keyboard-widget';

const A4_MIDI = 69;
const A4_TEST_DURATION_MS = 750;

function formatEngineStatus(diag: PianoDiagnostics): string {
  switch (diag.status) {
    case 'locked':
      return 'Locked';
    case 'initializing':
      return 'Initializing...';
    case 'ready':
      return diag.contextState === 'running' ? 'Ready' : `Ready (${diag.contextState})`;
    case 'error':
      return `Error: ${diag.error ?? 'Unknown'}`;
  }
}

/** Responsive M1 studio shell with audio unlock, C4-C5 keyboard, sustain, and HUD. */
export class AppShell {
  private readonly host: WasmAudioHost;
  private readonly telemetry: TelemetryClient;
  private readonly root: HTMLElement;
  private readonly activeSources = new Map<number, number>();
  private readonly keyboard: TestKeyboardWidget;

  private unlockBtn!: HTMLButtonElement;
  private a4Btn!: HTMLButtonElement;
  private sustainBtn!: HTMLButtonElement;
  private panicBtn!: HTMLButtonElement;
  private velocityInput!: HTMLInputElement;
  private velocityValueEl!: HTMLElement;
  private statusEl!: HTMLElement;
  private sampleRateEl!: HTMLElement;
  private voicesEl!: HTMLElement;
  private peakDbEl!: HTMLElement;
  private callbackUsEl!: HTMLElement;
  private errorBannerEl!: HTMLElement;

  private sustainDown = false;
  private velocity7 = 100;
  private a4PulseActive = false;
  private a4PulseTimer: number | null = null;
  private unsubscribeTelemetry: (() => void) | null = null;
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(root: HTMLElement, host: WasmAudioHost, telemetry: TelemetryClient) {
    this.root = root;
    this.host = host;
    this.telemetry = telemetry;
    this.renderLayout();
    const keyboardContainer = this.requireEl<HTMLDivElement>('#octave-keyboard');
    this.keyboard = new TestKeyboardWidget(keyboardContainer, {
      onPress: (midi) => void this.acquireNote(midi),
      onRelease: (midi) => this.releaseNote(midi),
      onToggleSustain: () => void this.toggleSustain(),
      onBlurReset: () => this.releaseAllInputs(),
    });
    this.bindControls();
    this.unsubscribeTelemetry = this.telemetry.subscribe((diag) => this.updateHud(diag));
  }

  destroy(): void {
    this.clearA4Pulse();
    this.keyboard.destroy();
    if (this.unsubscribeTelemetry) {
      this.unsubscribeTelemetry();
      this.unsubscribeTelemetry = null;
    }
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
  }

  private renderLayout(): void {
    this.root.innerHTML = `
      <main class="studio-shell">
        <header class="studio-header">
          <div>
            <p class="eyebrow">M1 Web AudioWorklet + Pure-Rust Wasm Shell</p>
            <h1>PianoSim Studio</h1>
          </div>
          <div class="header-actions">
            <button id="btn-unlock" data-testid="unlock-audio" type="button" class="btn btn-primary">
              Unlock Audio
            </button>
            <button id="btn-a4-trigger" data-testid="trigger-a4" type="button" class="btn btn-accent">
              A4 = 440 Hz
            </button>
          </div>
        </header>

        <p id="error-banner" data-testid="error-banner" class="error-banner" hidden></p>

        <section class="hud-grid" data-testid="telemetry-hud" aria-label="Engine Telemetry">
          <div class="hud-card">
            <span class="hud-label">Engine</span>
            <strong id="hud-status" data-testid="hud-status" class="hud-value">Locked</strong>
          </div>
          <div class="hud-card">
            <span class="hud-label">Sample Rate</span>
            <strong id="hud-sample-rate" data-testid="hud-sample-rate" class="hud-value">-- Hz</strong>
          </div>
          <div class="hud-card">
            <span class="hud-label">Active Voices</span>
            <strong id="hud-voices" data-testid="hud-voices" class="hud-value">0</strong>
          </div>
          <div class="hud-card">
            <span class="hud-label">Peak Level</span>
            <strong id="hud-peak-db" data-testid="hud-peak-db" class="hud-value">-inf dB</strong>
          </div>
          <div class="hud-card">
            <span class="hud-label">Callback Time</span>
            <strong id="hud-callback-us" data-testid="hud-callback-us" class="hud-value">0.0 us</strong>
          </div>
        </section>

        <section class="control-strip" aria-label="Performance Controls">
          <div class="control-group">
            <button
              id="btn-sustain"
              data-testid="sustain-toggle"
              type="button"
              class="btn btn-toggle"
              aria-pressed="false"
            >
              Sustain Pedal (Space): OFF
            </button>
            <button id="btn-panic" data-testid="panic-btn" type="button" class="btn btn-ghost">
              All Notes Off
            </button>
          </div>
          <label class="velocity-control" for="velocity-slider">
            <span>Velocity</span>
            <input id="velocity-slider" data-testid="velocity-slider" type="range" min="1" max="127" value="100" />
            <span id="velocity-value" class="mono">100</span>
          </label>
        </section>

        <section class="keyboard-stage" aria-label="Test Keyboard C4 to C5">
          <div class="keyboard-meta">
            <span>C4 - C5 Test Octave</span>
            <span class="hint">Keys: A S D F G H J K (white) &middot; W E T Y U (black) &middot; Space (sustain)</span>
          </div>
          <div id="octave-keyboard" class="octave-keyboard" role="group" aria-label="Piano Keys C4 to C5"></div>
        </section>
      </main>
    `;

    this.unlockBtn = this.requireEl('#btn-unlock');
    this.a4Btn = this.requireEl('#btn-a4-trigger');
    this.sustainBtn = this.requireEl('#btn-sustain');
    this.panicBtn = this.requireEl('#btn-panic');
    this.velocityInput = this.requireEl('#velocity-slider');
    this.velocityValueEl = this.requireEl('#velocity-value');
    this.statusEl = this.requireEl('#hud-status');
    this.sampleRateEl = this.requireEl('#hud-sample-rate');
    this.voicesEl = this.requireEl('#hud-voices');
    this.peakDbEl = this.requireEl('#hud-peak-db');
    this.callbackUsEl = this.requireEl('#hud-callback-us');
    this.errorBannerEl = this.requireEl('#error-banner');
  }

  private bindControls(): void {
    const onUnlock = () => void this.host.unlock().catch(() => undefined);
    const onTriggerA4 = () => void this.triggerA4Pulse();
    const onSustainClick = () => void this.toggleSustain();
    const onPanic = () => this.releaseAllInputs();
    const onVelocity = () => {
      const next = Number(this.velocityInput.value);
      if (Number.isFinite(next)) {
        this.velocity7 = Math.max(1, Math.min(127, Math.trunc(next)));
        this.velocityValueEl.textContent = String(this.velocity7);
      }
    };

    this.unlockBtn.addEventListener('click', onUnlock);
    this.a4Btn.addEventListener('click', onTriggerA4);
    this.sustainBtn.addEventListener('click', onSustainClick);
    this.panicBtn.addEventListener('click', onPanic);
    this.velocityInput.addEventListener('input', onVelocity);

    this.cleanupHandlers.push(
      () => this.unlockBtn.removeEventListener('click', onUnlock),
      () => this.a4Btn.removeEventListener('click', onTriggerA4),
      () => this.sustainBtn.removeEventListener('click', onSustainClick),
      () => this.panicBtn.removeEventListener('click', onPanic),
      () => this.velocityInput.removeEventListener('input', onVelocity),
    );
  }

  private async ensureUnlocked(): Promise<boolean> {
    if (this.host.isReady()) {
      return true;
    }
    try {
      await this.host.unlock();
      return this.host.isReady();
    } catch {
      return false;
    }
  }

  private async acquireNote(midi: number): Promise<void> {
    const prevCount = this.activeSources.get(midi) ?? 0;
    this.activeSources.set(midi, prevCount + 1);
    this.keyboard.setKeyActive(midi, true);
    if (prevCount > 0) {
      return;
    }
    const unlocked = await this.ensureUnlocked();
    if (!unlocked) {
      return;
    }
    if ((this.activeSources.get(midi) ?? 0) > 0) {
      this.host.noteOn(midi, u7ToU16(this.velocity7), 0);
    }
  }

  private releaseNote(midi: number): void {
    const prevCount = this.activeSources.get(midi) ?? 0;
    if (prevCount <= 0) {
      return;
    }
    if (prevCount === 1) {
      this.activeSources.delete(midi);
      this.keyboard.setKeyActive(midi, false);
      if (this.host.isReady()) {
        this.host.noteOff(midi, 0, 0);
      }
    } else {
      this.activeSources.set(midi, prevCount - 1);
    }
  }

  private async triggerA4Pulse(): Promise<void> {
    const unlocked = await this.ensureUnlocked();
    if (!unlocked) {
      return;
    }
    this.clearA4Pulse();
    this.a4PulseActive = true;
    await this.acquireNote(A4_MIDI);
    this.a4PulseTimer = window.setTimeout(() => {
      this.clearA4Pulse();
    }, A4_TEST_DURATION_MS);
  }

  private clearA4Pulse(): void {
    if (this.a4PulseTimer !== null) {
      window.clearTimeout(this.a4PulseTimer);
      this.a4PulseTimer = null;
    }
    if (this.a4PulseActive) {
      this.a4PulseActive = false;
      this.releaseNote(A4_MIDI);
    }
  }

  private async toggleSustain(): Promise<void> {
    this.sustainDown = !this.sustainDown;
    this.sustainBtn.setAttribute('aria-pressed', String(this.sustainDown));
    this.sustainBtn.classList.toggle('is-active', this.sustainDown);
    this.sustainBtn.textContent = `Sustain Pedal (Space): ${this.sustainDown ? 'ON' : 'OFF'}`;
    if (await this.ensureUnlocked()) {
      this.host.pedal(PEDAL_SUSTAIN, this.sustainDown ? 65535 : 0, 0);
    }
  }

  private releaseAllInputs(): void {
    this.clearA4Pulse();
    this.keyboard.clearInputs();
    this.activeSources.clear();
    if (this.host.isReady()) {
      this.host.allNotesOff();
    }
  }

  private updateHud(diag: PianoDiagnostics): void {
    this.statusEl.textContent = formatEngineStatus(diag);
    this.sampleRateEl.textContent = diag.sampleRate > 0 ? `${diag.sampleRate} Hz` : '-- Hz';
    this.voicesEl.textContent = String(Math.round(diag.telemetry.activeVoices));
    this.peakDbEl.textContent = formatPeakDb(diag.telemetry.peakL, diag.telemetry.peakR);
    this.callbackUsEl.textContent = `${diag.telemetry.callbackUs.toFixed(1)} us`;

    this.unlockBtn.disabled = diag.status === 'initializing';
    this.unlockBtn.textContent =
      diag.status === 'ready'
        ? 'Audio Ready'
        : diag.status === 'initializing'
          ? 'Starting...'
          : 'Unlock Audio';
    this.unlockBtn.classList.toggle('is-ready', diag.status === 'ready');

    if (diag.error) {
      this.errorBannerEl.hidden = false;
      this.errorBannerEl.textContent = diag.error;
    } else {
      this.errorBannerEl.hidden = true;
      this.errorBannerEl.textContent = '';
    }
  }

  private requireEl<T extends Element>(selector: string): T {
    const el = this.root.querySelector<T>(selector);
    if (!el) {
      throw new Error(`Missing required UI element: ${selector}`);
    }
    return el;
  }
}
