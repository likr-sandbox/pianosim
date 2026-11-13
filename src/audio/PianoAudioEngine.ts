/**
 * PianoAudioEngine.ts
 * Browser Web Audio API engine managing AudioContext lifecycle, AudioWorklet initialization,
 * autoplay resume, and fallback to ScriptProcessorNode when AudioWorklet is unavailable.
 */

import { PianoSynthesizer } from './PianoSynthesizer';
import {
  createPianoWorkletBlobUrl,
  WorkletInboundMessage,
  WorkletOutboundMessage,
} from './worklet/piano-worklet-processor';

export class PianoAudioEngine {
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;

  // Fallback engine if AudioWorklet is unsupported (e.g. legacy browsers/iframes)
  private fallbackSynth: PianoSynthesizer | null = null;
  private fallbackScriptNode: ScriptProcessorNode | null = null;
  private isFallbackMode: boolean = false;

  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;
  private activeVoices: number = 0;
  private sustainPedalDown: boolean = false;

  constructor() {}

  public init(): Promise<void> {
    if (this.initialized) return Promise.resolve();
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      // Headless / Node.js environment safeguard
      if (typeof window === 'undefined') {
        return;
      }

      // Initialize Web Audio context with standard cross-browser constructor
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;

      if (!AudioCtx) {
        console.warn('Web Audio API is not supported in this environment');
        return;
      }

      this.audioContext = new AudioCtx({ latencyHint: 'interactive' });

      // Attempt standard AudioWorklet initialization via in-memory Blob URL
      if (typeof this.audioContext.audioWorklet !== 'undefined') {
        try {
          const workletUrl = createPianoWorkletBlobUrl();
          await this.audioContext.audioWorklet.addModule(workletUrl);
          URL.revokeObjectURL(workletUrl);

          this.workletNode = new AudioWorkletNode(this.audioContext, 'piano-worklet-processor', {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [2],
          });

          this.workletNode.port.onmessage = (event: MessageEvent<WorkletOutboundMessage>) => {
            this.handleWorkletMessage(event.data);
          };

          this.workletNode.connect(this.audioContext.destination);
          this.initialized = true;
          this.isFallbackMode = false;
          return;
        } catch (err) {
          console.warn('AudioWorklet module initialization failed, switching to ScriptProcessor fallback:', err);
        }
      }

      // Fallback: ScriptProcessorNode running PianoSynthesizer on main thread
      this.initFallback();
      this.initialized = true;
    })();

    return this.initPromise;
  }

  private initFallback(): void {
    if (!this.audioContext) return;
    this.isFallbackMode = true;
    this.fallbackSynth = new PianoSynthesizer(this.audioContext.sampleRate);
    const bufferSize = 512;
    this.fallbackScriptNode = this.audioContext.createScriptProcessor(bufferSize, 0, 2);

    this.fallbackScriptNode.onaudioprocess = (e: AudioProcessingEvent) => {
      const left = e.outputBuffer.getChannelData(0);
      const right = e.outputBuffer.getChannelData(1);
      if (this.fallbackSynth) {
        this.fallbackSynth.render(left, right, bufferSize);
      }
    };

    this.fallbackScriptNode.connect(this.audioContext.destination);
  }

  public async resume(): Promise<void> {
    if (!this.audioContext) {
      await this.init();
    }
    if (this.audioContext && this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  public noteOn(midiNote: number, velocity: number = 80): void {
    if (velocity === 0) {
      this.noteOff(midiNote);
      return;
    }
    if (!this.initialized) {
      this.init().then(() => this.noteOn(midiNote, velocity));
      return;
    }
    this.resume();

    if (this.isFallbackMode && this.fallbackSynth) {
      this.fallbackSynth.noteOn({ midiNote, velocity });
      return;
    }

    if (this.workletNode) {
      const msg: WorkletInboundMessage = {
        type: 'noteOn',
        midiNote,
        velocity,
      };
      this.workletNode.port.postMessage(msg);
    }
  }

  public noteOff(midiNote: number): void {
    if (!this.initialized) return;

    if (this.isFallbackMode && this.fallbackSynth) {
      this.fallbackSynth.noteOff({ midiNote });
      return;
    }

    if (this.workletNode) {
      const msg: WorkletInboundMessage = {
        type: 'noteOff',
        midiNote,
      };
      this.workletNode.port.postMessage(msg);
    }
  }

  public setSustainPedal(down: boolean): void {
    this.sustainPedalDown = down;
    if (!this.initialized) return;

    if (this.isFallbackMode && this.fallbackSynth) {
      this.fallbackSynth.setSustainPedal(down);
      return;
    }

    if (this.workletNode) {
      const msg: WorkletInboundMessage = {
        type: 'setSustainPedal',
        down,
      };
      this.workletNode.port.postMessage(msg);
    }
  }

  public allNotesOff(): void {
    if (!this.initialized) return;

    if (this.isFallbackMode && this.fallbackSynth) {
      this.fallbackSynth.allNotesOff();
      return;
    }

    if (this.workletNode) {
      const msg: WorkletInboundMessage = { type: 'allNotesOff' };
      this.workletNode.port.postMessage(msg);
    }
  }

  private handleWorkletMessage(msg: WorkletOutboundMessage): void {
    if (msg.type === 'voiceCount') {
      this.activeVoices = msg.count;
    }
  }

  public isSustainPedalDown(): boolean {
    return this.sustainPedalDown;
  }

  public isInitialized(): boolean {
    return this.initialized;
  }

  public getSampleRate(): number {
    return this.audioContext?.sampleRate ?? 44100;
  }

  public getActiveVoiceCount(): number {
    if (this.isFallbackMode && this.fallbackSynth) {
      return this.fallbackSynth.getActiveVoiceCount();
    }
    return this.activeVoices;
  }
}
