/** Live audio recording & 24-bit TPDF / 32-bit float WAV export control strip. */

import type { WavFormat } from '../audio/abi';

export interface RecorderControlsCallbacks {
  onStartRecording(): void;
  onStopRecording(): Promise<void>;
  onExportWav(format: WavFormat): void;
}

export class RecorderControls {
  private readonly container: HTMLElement;
  private readonly callbacks: RecorderControlsCallbacks;

  private startBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private formatSelect!: HTMLSelectElement;
  private exportBtn!: HTMLButtonElement;
  private statusEl!: HTMLElement;

  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(container: HTMLElement, callbacks: RecorderControlsCallbacks) {
    this.container = container;
    this.callbacks = callbacks;
    this.render();
    this.bindEvents();
  }

  updateStatus(frames: number, durationSec: number, recording: boolean): void {
    this.startBtn.disabled = recording;
    this.stopBtn.disabled = !recording;
    this.exportBtn.disabled = recording || frames <= 0;
    this.startBtn.classList.toggle('is-recording', recording);
    if (recording) {
      this.statusEl.textContent = `REC ${durationSec.toFixed(1)}s`;
    } else if (frames > 0) {
      this.statusEl.textContent = `Ready ${durationSec.toFixed(1)}s`;
    } else {
      this.statusEl.textContent = 'Idle';
    }
  }

  destroy(): void {
    for (const off of this.cleanupHandlers) {
      off();
    }
    this.cleanupHandlers.length = 0;
    this.container.innerHTML = '';
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="recorder-controls" role="group" aria-label="Live WAV Recorder">
        <button
          id="btn-rec-start"
          data-testid="record-start"
          type="button"
          class="btn btn-record"
        >Record</button>
        <button
          id="btn-rec-stop"
          data-testid="record-stop"
          type="button"
          class="btn btn-secondary"
          disabled
        >Stop Rec</button>
        <select
          id="select-rec-format"
          data-testid="record-format"
          class="select-record-format"
          aria-label="WAV Export Format"
        >
          <option value="pcm24">24-bit WAV (TPDF)</option>
          <option value="f32">32-bit Float WAV</option>
        </select>
        <button
          id="btn-rec-export"
          data-testid="record-export"
          type="button"
          class="btn btn-accent"
          disabled
        >Export WAV</button>
        <span id="val-rec-status" data-testid="record-status" class="record-status val-mono">Idle</span>
      </div>
    `;

    this.startBtn = this.container.querySelector('#btn-rec-start')!;
    this.stopBtn = this.container.querySelector('#btn-rec-stop')!;
    this.formatSelect = this.container.querySelector('#select-rec-format')!;
    this.exportBtn = this.container.querySelector('#btn-rec-export')!;
    this.statusEl = this.container.querySelector('#val-rec-status')!;
  }

  private bindEvents(): void {
    const onStart = () => {
      this.callbacks.onStartRecording();
    };
    const onStop = () => {
      void this.callbacks.onStopRecording();
    };
    const onExport = () => {
      const format: WavFormat = this.formatSelect.value === 'f32' ? 'f32' : 'pcm24';
      this.callbacks.onExportWav(format);
    };

    this.startBtn.addEventListener('click', onStart);
    this.stopBtn.addEventListener('click', onStop);
    this.exportBtn.addEventListener('click', onExport);

    this.cleanupHandlers.push(() => {
      this.startBtn.removeEventListener('click', onStart);
      this.stopBtn.removeEventListener('click', onStop);
      this.exportBtn.removeEventListener('click', onExport);
    });
  }
}
