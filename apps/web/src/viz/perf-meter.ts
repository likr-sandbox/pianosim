/** Real-time HUD displaying active voices, p50/p99 callback µs, and D1 overload alert. */

import type { PianoDiagnostics } from '../audio/abi';
import { formatPeakDb, type TelemetryClient } from '../audio/telemetry-client';

export class PerfMeter {
  private readonly container: HTMLElement;
  private readonly telemetry: TelemetryClient;
  private unsubscribe: (() => void) | null = null;

  private statusEl!: HTMLElement;
  private sampleRateEl!: HTMLElement;
  private voicesEl!: HTMLElement;
  private peakDbEl!: HTMLElement;
  private p50El!: HTMLElement;
  private p99El!: HTMLElement;
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
        <div class="hud-card">
          <span class="hud-label">Peak Level</span>
          <strong id="hud-peak-db" data-testid="hud-peak-db" class="hud-value">-inf dB</strong>
        </div>
        <div class="hud-card">
          <span class="hud-label">Callback Time</span>
          <div class="hud-value-row">
            <strong id="hud-callback-us" data-testid="hud-callback-us" class="hud-value">0.0</strong>
            <span class="hud-unit">µs</span>
          </div>
        </div>
        <div class="hud-card">
          <span class="hud-label">p50 Load</span>
          <div class="hud-value-row">
            <strong id="hud-p50" data-testid="hud-p50" class="hud-value">0.0</strong>
            <span class="hud-unit">µs</span>
          </div>
        </div>
        <div class="hud-card">
          <span class="hud-label">p99 Load</span>
          <div class="hud-value-row">
            <strong id="hud-p99" data-testid="hud-p99" class="hud-value">0.0</strong>
            <span class="hud-unit">µs</span>
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
    this.p50El = this.container.querySelector('#hud-p50')!;
    this.p99El = this.container.querySelector('#hud-p99')!;
    this.callbackUsEl = this.container.querySelector('#hud-callback-us')!;
    this.alertEl = this.container.querySelector('#hud-alert')!;
    this.alertTextEl = this.container.querySelector('.hud-alert-text')!;
  }

  private update(diag: PianoDiagnostics): void {
    // Engine status
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

    // Sample rate
    this.sampleRateEl.textContent = diag.sampleRate > 0 ? `${diag.sampleRate} Hz` : '-- Hz';

    // Active voices: plain integer string (e.g. "0", "24")
    this.voicesEl.textContent = String(Math.trunc(diag.telemetry.activeVoices));

    // Peak level
    this.peakDbEl.textContent = formatPeakDb(diag.telemetry.peakL, diag.telemetry.peakR);

    // Numeric strings without unit suffix so Number() and parseFloat() both succeed!
    this.callbackUsEl.textContent = diag.telemetry.callbackUs.toFixed(1);
    this.p50El.textContent = diag.p50Us.toFixed(1);
    this.p99El.textContent = diag.p99Us.toFixed(1);

    // D1 Performance Budget (< 1,866.7 µs @ 48 kHz, or 70% of 128-sample callback time)
    const rate = diag.sampleRate > 0 ? diag.sampleRate : 48000;
    const blockBudgetUs = (128 / rate) * 1_000_000;
    const d1BudgetUs = blockBudgetUs * 0.7; // ~1,866.7 µs @ 48 kHz
    const warningBudgetUs = blockBudgetUs * 0.5; // ~1,333.3 µs @ 48 kHz
    const p99 = diag.p99Us;

    if (p99 > d1BudgetUs) {
      this.alertEl.className = 'hud-alert is-overloaded alert-critical';
      this.alertTextEl.textContent = 'OVERLOAD';
    } else if (p99 > warningBudgetUs) {
      this.alertEl.className = 'hud-alert is-warning';
      this.alertTextEl.textContent = 'WARNING';
    } else {
      this.alertEl.className = 'hud-alert ok';
      this.alertTextEl.textContent = 'NOMINAL';
    }
  }
}
