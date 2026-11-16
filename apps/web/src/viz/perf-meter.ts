/** Real-time HUD displaying Peak Hold/RMS dBFS, dB SPL, -60..0 dBFS meter + CLIP, and honest p50/p99/max/overruns. */

import type { PianoDiagnostics } from '../audio/abi';
import { formatDbfs, formatPeakDb, type TelemetryClient } from '../audio/telemetry-client';

function dbToPercent(db: number): number {
  if (!Number.isFinite(db)) {
    return 0;
  }
  return Math.max(0, Math.min(100, ((db + 60.0) / 60.0) * 100.0));
}

export class PerfMeter {
  private readonly container: HTMLElement;
  private readonly telemetry: TelemetryClient;
  private unsubscribe: (() => void) | null = null;
  private onClipClick: (() => void) | null = null;

  private statusEl!: HTMLElement;
  private sampleRateEl!: HTMLElement;
  private voicesEl!: HTMLElement;
  private peakDbEl!: HTMLElement;
  private rmsDbEl!: HTMLElement;
  private splDbEl!: HTMLElement;
  private levelFillEl!: HTMLElement;
  private levelHoldEl!: HTMLElement;
  private clipBadgeEl!: HTMLButtonElement;
  private p50El!: HTMLElement;
  private p99El!: HTMLElement;
  private maxUsEl!: HTMLElement;
  private overrunsEl!: HTMLElement;
  private callbackUsEl!: HTMLElement;
  private alertEl!: HTMLElement;
  private alertTextEl!: HTMLElement;

  constructor(container: HTMLElement, telemetry: TelemetryClient) {
    this.container = container;
    this.telemetry = telemetry;
    this.render();
    this.bindElements();
    this.unsubscribe = this.telemetry.subscribe((diag) => this.update(diag));
  }

  destroy(): void {
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = null;
    }
    if (this.onClipClick && this.clipBadgeEl) {
      this.clipBadgeEl.removeEventListener('click', this.onClipClick);
      this.onClipClick = null;
    }
    this.container.innerHTML = '';
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="hud-grid" data-testid="telemetry-hud" aria-label="Engine Telemetry">
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
        <div class="hud-card hud-card-meter">
          <div class="hud-meter-header">
            <span class="hud-label">Output Level (-60..0 dBFS)</span>
            <button id="hud-clip" data-testid="hud-clip" type="button" class="hud-clip" title="Click to reset CLIP indicator">CLIP</button>
          </div>
          <div class="hud-value-row">
            <strong id="hud-peak-db" data-testid="hud-peak-db" class="hud-value">-inf dB</strong>
            <span class="hud-sub" id="hud-rms-db" data-testid="hud-rms-db">-inf dBFS</span>
            <span class="hud-sub" id="hud-spl-db" data-testid="hud-spl-db">-- dB SPL</span>
          </div>
          <div class="hud-level-track" data-testid="hud-level-bar" aria-hidden="true">
            <div id="hud-level-fill" data-testid="hud-level-fill" class="hud-level-fill" style="width: 0%"></div>
            <div id="hud-level-hold" class="hud-level-hold" style="left: 0%"></div>
          </div>
        </div>
        <div class="hud-card">
          <span class="hud-label">Callback Avg</span>
          <div class="hud-value-row">
            <strong id="hud-callback-us" data-testid="hud-callback-us" class="hud-value">0.0</strong>
            <span class="hud-unit">µs</span>
          </div>
        </div>
        <div class="hud-card">
          <span class="hud-label">p50 / p99 Load</span>
          <div class="hud-value-row">
            <strong id="hud-p50" data-testid="hud-p50" class="hud-value">0.0</strong>
            <span class="hud-unit">/</span>
            <strong id="hud-p99" data-testid="hud-p99" class="hud-value">0.0</strong>
            <span class="hud-unit">µs</span>
          </div>
        </div>
        <div class="hud-card">
          <span class="hud-label">True Max / D1 Overruns</span>
          <div class="hud-value-row">
            <strong id="hud-max-us" data-testid="hud-max-us" class="hud-value">0.0</strong>
            <span class="hud-unit">µs ·</span>
            <strong id="hud-overruns" data-testid="hud-overruns" class="hud-value">0</strong>
          </div>
        </div>
        <div class="hud-card hud-card-alert">
          <span class="hud-label">D1 Performance</span>
          <div id="hud-alert" data-testid="hud-alert" class="hud-alert ok" role="status">
            <span class="hud-alert-dot"></span>
            <span class="hud-alert-text">NOMINAL</span>
          </div>
        </div>
      </div>
    `;
  }

  private bindElements(): void {
    this.statusEl = this.container.querySelector('#hud-status')!;
    this.sampleRateEl = this.container.querySelector('#hud-sample-rate')!;
    this.voicesEl = this.container.querySelector('#hud-voices')!;
    this.peakDbEl = this.container.querySelector('#hud-peak-db')!;
    this.rmsDbEl = this.container.querySelector('#hud-rms-db')!;
    this.splDbEl = this.container.querySelector('#hud-spl-db')!;
    this.levelFillEl = this.container.querySelector('#hud-level-fill')!;
    this.levelHoldEl = this.container.querySelector('#hud-level-hold')!;
    this.clipBadgeEl = this.container.querySelector('#hud-clip')!;
    this.p50El = this.container.querySelector('#hud-p50')!;
    this.p99El = this.container.querySelector('#hud-p99')!;
    this.maxUsEl = this.container.querySelector('#hud-max-us')!;
    this.overrunsEl = this.container.querySelector('#hud-overruns')!;
    this.callbackUsEl = this.container.querySelector('#hud-callback-us')!;
    this.alertEl = this.container.querySelector('#hud-alert')!;
    this.alertTextEl = this.container.querySelector('.hud-alert-text')!;

    this.onClipClick = () => {
      this.telemetry.clearClip();
    };
    this.clipBadgeEl.addEventListener('click', this.onClipClick);
  }

  private update(diag: PianoDiagnostics): void {
    switch (diag.status) {
      case 'locked':
        this.statusEl.textContent = 'Locked';
        break;
      case 'initializing':
        this.statusEl.textContent = 'Initializing...';
        break;
      case 'ready':
        this.statusEl.textContent =
          diag.contextState === 'running' ? 'Ready' : `Ready (${diag.contextState})`;
        break;
      case 'error':
        this.statusEl.textContent = 'Error';
        break;
    }

    this.sampleRateEl.textContent = diag.sampleRate > 0 ? `${diag.sampleRate} Hz` : '-- Hz';
    this.voicesEl.textContent = String(Math.trunc(diag.telemetry.activeVoices));

    this.peakDbEl.textContent = formatPeakDb(diag.telemetry.peakL, diag.telemetry.peakR);
    this.rmsDbEl.textContent = `RMS ${formatDbfs(diag.rmsDbfs)}`;
    this.splDbEl.textContent = Number.isFinite(diag.splDb)
      ? `≈ ${diag.splDb.toFixed(1)} dB SPL`
      : '-- dB SPL';

    const rmsPct = dbToPercent(diag.rmsDbfs);
    const holdPct = dbToPercent(diag.peakHoldDbfs);
    this.levelFillEl.style.width = `${rmsPct.toFixed(1)}%`;
    this.levelHoldEl.style.left = `${holdPct.toFixed(1)}%`;
    this.levelHoldEl.style.opacity = holdPct > 0 ? '1' : '0';
    this.clipBadgeEl.classList.toggle('is-clipping', diag.clipLatched);

    this.callbackUsEl.textContent = diag.telemetry.callbackUs.toFixed(1);
    this.p50El.textContent = diag.p50Us.toFixed(1);
    this.p99El.textContent = diag.p99Us.toFixed(1);
    this.maxUsEl.textContent = diag.telemetry.maxCallbackUs.toFixed(1);
    this.overrunsEl.textContent = String(Math.trunc(diag.telemetry.overrunCount));

    const rate = diag.sampleRate > 0 ? diag.sampleRate : 48000;
    const blockBudgetUs = (128 / rate) * 1_000_000;
    const d1BudgetUs = blockBudgetUs * 0.7;
    const warningBudgetUs = blockBudgetUs * 0.5;
    const p99 = diag.p99Us;

    if (p99 > d1BudgetUs) {
      this.alertEl.className = 'hud-alert is-overloaded alert-critical';
      this.alertTextEl.textContent = 'OVERLOAD';
    } else if (p99 > warningBudgetUs || diag.telemetry.overrunCount > 0) {
      this.alertEl.className = 'hud-alert is-warning';
      this.alertTextEl.textContent = 'WARNING';
    } else {
      this.alertEl.className = 'hud-alert ok';
      this.alertTextEl.textContent = 'NOMINAL';
    }
  }
}
