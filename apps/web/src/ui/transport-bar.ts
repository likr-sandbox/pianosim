/** Transport control bar for demo scores, SMF file upload, play/pause/stop, scrub seek, and tempo scaling. */

import { DEMO_SCORES } from '../midi/demo-scores';
import type { MidiPlayer } from '../midi/midi-player';
import type { WebMidiManager } from '../midi/web-midi';

export interface TransportBarCallbacks {
  onSelectDemo(demoId: string): Promise<void> | void;
  onUploadMidi(bytes: ArrayBuffer, fileName: string): void;
}

export class TransportBar {
  private readonly container: HTMLElement;
  private readonly player: MidiPlayer;
  private readonly midiManager: WebMidiManager;
  private readonly callbacks: TransportBarCallbacks;

  private demoSelect!: HTMLSelectElement;
  private playBtn!: HTMLButtonElement;
  private pauseBtn!: HTMLButtonElement;
  private stopBtn!: HTMLButtonElement;
  private seekBar!: HTMLInputElement;
  private tempoSlider!: HTMLInputElement;
  private tempoValEl!: HTMLElement;
  private fileInput!: HTMLInputElement;
  private deviceSelect!: HTMLSelectElement;
  private recorderSlot!: HTMLElement;

  private pendingLoad: Promise<void> = Promise.resolve();
  private readonly cleanupHandlers: Array<() => void> = [];

  constructor(
    container: HTMLElement,
    player: MidiPlayer,
    midiManager: WebMidiManager,
    callbacks: TransportBarCallbacks,
  ) {
    this.container = container;
    this.player = player;
    this.midiManager = midiManager;
    this.callbacks = callbacks;
    this.render();
    this.bindEvents();
  }

  updateProgress(currentSec: number, totalSec: number): void {
    if (this.seekBar) {
      this.seekBar.max = totalSec > 0 ? totalSec.toFixed(1) : '0';
      this.seekBar.value = currentSec.toFixed(1);
    }
  }

  updateDevices(devices: Array<{ id: string; name: string }>): void {
    if (!this.deviceSelect) return;
    const currentVal = this.deviceSelect.value;
    this.deviceSelect.innerHTML = '<option value="">All MIDI Inputs</option>';
    for (const d of devices) {
      const opt = document.createElement('option');
      opt.value = d.id;
      opt.textContent = d.name;
      this.deviceSelect.appendChild(opt);
    }
    this.deviceSelect.value = currentVal;
  }

  getRecorderMount(): HTMLElement {
    return this.recorderSlot;
  }

  triggerDemoLoad(demoId: string): Promise<void> {
    if (this.demoSelect) {
      this.demoSelect.value = demoId;
    }
    const loadPromise = Promise.resolve(this.callbacks.onSelectDemo(demoId));
    this.pendingLoad = loadPromise;
    return loadPromise;
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
      <section class="transport-bar" aria-label="MIDI Playback & Transport Controls">
        <div class="transport-row main-transport">
          <div class="transport-group demo-picker">
            <label for="select-demo" class="transport-label">Demo Score</label>
            <select id="select-demo" data-testid="demo-select" class="select-demo">
              ${DEMO_SCORES.map((s) => `<option value="${s.id}">${s.title}</option>`).join('\n              ')}
            </select>
          </div>

          <div class="transport-group playback-buttons">
            <button id="btn-play" data-testid="transport-play" type="button" class="btn btn-primary">
              Play
            </button>
            <button id="btn-pause" data-testid="transport-pause" type="button" class="btn btn-secondary">
              Pause
            </button>
            <button id="btn-stop" data-testid="transport-stop" type="button" class="btn btn-secondary">
              Stop
            </button>
          </div>

          <div class="transport-group tempo-control">
            <label for="slider-tempo" class="transport-label">
              <span>Tempo</span>
              <input
                id="slider-tempo"
                data-testid="tempo-slider"
                type="range"
                min="0.5"
                max="2.0"
                step="0.05"
                value="1.0"
                class="tempo-slider"
              />
              <span id="val-tempo" class="val-mono">1.0x</span>
            </label>
          </div>

          <div class="transport-group file-upload">
            <label for="input-midi-file" class="transport-label file-upload-label">
              <span>Upload SMF (.mid)</span>
              <input
                id="input-midi-file"
                data-testid="midi-file-input"
                type="file"
                accept=".mid,.midi"
                class="file-input"
              />
            </label>
          </div>

          <div class="transport-group midi-devices">
            <label for="midi-device-select" class="transport-label">Web MIDI</label>
            <select id="midi-device-select" data-testid="midi-device-select" class="select-device">
              <option value="">All MIDI Inputs</option>
            </select>
          </div>

          <div id="recorder-mount" class="transport-group recorder-group"></div>
        </div>

        <div class="transport-row seek-row">
          <label for="progress-bar" class="seek-label">
            <span>Playback Position</span>
            <input
              id="progress-bar"
              data-testid="transport-seek"
              type="range"
              min="0"
              max="32"
              step="0.1"
              value="0"
              class="transport-seek"
            />
          </label>
        </div>
      </section>
    `;

    this.demoSelect = this.container.querySelector('#select-demo')!;
    this.playBtn = this.container.querySelector('#btn-play')!;
    this.pauseBtn = this.container.querySelector('#btn-pause')!;
    this.stopBtn = this.container.querySelector('#btn-stop')!;
    this.seekBar = this.container.querySelector('#progress-bar')!;
    this.tempoSlider = this.container.querySelector('#slider-tempo')!;
    this.tempoValEl = this.container.querySelector('#val-tempo')!;
    this.fileInput = this.container.querySelector('#input-midi-file')!;
    this.deviceSelect = this.container.querySelector('#midi-device-select')!;
    this.recorderSlot = this.container.querySelector('#recorder-mount')!;
  }

  private async awaitPendingLoad(): Promise<void> {
    let current = this.pendingLoad;
    await current;
    while (current !== this.pendingLoad) {
      current = this.pendingLoad;
      await current;
    }
  }

  private bindEvents(): void {
    const onDemoChange = () => {
      this.pendingLoad = Promise.resolve(this.callbacks.onSelectDemo(this.demoSelect.value));
    };

    const onPlayClick = async () => {
      await this.awaitPendingLoad();
      await this.player.play();
    };

    const onPauseClick = () => {
      this.player.pause();
    };

    const onStopClick = () => {
      this.player.stop();
    };

    const onSeekInput = async () => {
      const targetSec = parseFloat(this.seekBar.value);
      await this.awaitPendingLoad();
      this.player.seek(targetSec);
    };

    const onTempoInput = () => {
      const factor = parseFloat(this.tempoSlider.value);
      this.tempoValEl.textContent = `${factor.toFixed(2)}x`;
      this.player.setTempo(factor);
    };

    const onFileChange = () => {
      const file = this.fileInput.files?.[0];
      if (!file) return;
      this.pendingLoad = Promise.resolve();
      const reader = new FileReader();
      reader.onload = () => {
        if (reader.result instanceof ArrayBuffer) {
          this.callbacks.onUploadMidi(reader.result, file.name);
        }
      };
      reader.readAsArrayBuffer(file);
    };

    const onDeviceChange = () => {
      this.midiManager.selectInput(this.deviceSelect.value);
    };

    this.demoSelect.addEventListener('change', onDemoChange);
    this.playBtn.addEventListener('click', onPlayClick);
    this.pauseBtn.addEventListener('click', onPauseClick);
    this.stopBtn.addEventListener('click', onStopClick);
    this.seekBar.addEventListener('input', onSeekInput);
    this.tempoSlider.addEventListener('input', onTempoInput);
    this.fileInput.addEventListener('change', onFileChange);
    this.deviceSelect.addEventListener('change', onDeviceChange);

    this.cleanupHandlers.push(() => {
      this.demoSelect.removeEventListener('change', onDemoChange);
      this.playBtn.removeEventListener('click', onPlayClick);
      this.pauseBtn.removeEventListener('click', onPauseClick);
      this.stopBtn.removeEventListener('click', onStopClick);
      this.seekBar.removeEventListener('input', onSeekInput);
      this.tempoSlider.removeEventListener('input', onTempoInput);
      this.fileInput.removeEventListener('change', onFileChange);
      this.deviceSelect.removeEventListener('change', onDeviceChange);
    });
  }
}
