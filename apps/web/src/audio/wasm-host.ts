import {
  WORKLET_PROCESSOR_NAME,
  type PedalKind,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
} from './abi';
import { TelemetryClient } from './telemetry-client';
import workletCode from './worklet-processor.ts?worklet';

const WASM_MAGIC = [0x00, 0x61, 0x73, 0x6d] as const;
const INIT_TIMEOUT_MS = 10_000;
const ANALYSER_FFT_SIZE = 16_384;

function assertValidWasmBytes(bytes: ArrayBuffer): void {
  if (bytes.byteLength < 8) {
    throw new Error('Fetched piano_wasm.wasm is empty or truncated.');
  }
  const header = new Uint8Array(bytes, 0, 4);
  for (let i = 0; i < 4; i++) {
    if (header[i] !== WASM_MAGIC[i]) {
      throw new Error(
        'Invalid Wasm binary magic header (expected compiled public/piano_wasm.wasm).',
      );
    }
  }
}

async function fetchWasmBinary(): Promise<ArrayBuffer> {
  const url = `${import.meta.env.BASE_URL}piano_wasm.wasm`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  assertValidWasmBytes(bytes);
  return bytes;
}

/** Manages AudioContext lifecycle, Wasm Worklet initialization, and passive diagnostic tap. */
export class WasmAudioHost {
  private readonly telemetry: TelemetryClient;
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private analyser: AnalyserNode | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(telemetry: TelemetryClient) {
    this.telemetry = telemetry;
  }

  isReady(): boolean {
    return this.workletNode !== null && this.telemetry.getDiagnostics().status === 'ready';
  }

  async unlock(): Promise<void> {
    if (this.isReady() && this.audioContext) {
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
        this.telemetry.syncContextState();
      }
      return;
    }
    if (this.initPromise) {
      return this.initPromise;
    }
    this.initPromise = this.initializePipeline().finally(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  noteOn(midiKey: number, velocityU16: number, sampleOffset = 0): void {
    this.postToWorklet({ type: 'noteOn', midiKey, velocityU16, sampleOffset });
  }

  noteOff(midiKey: number, releaseVelU16 = 0, sampleOffset = 0): void {
    this.postToWorklet({ type: 'noteOff', midiKey, releaseVelU16, sampleOffset });
  }

  pedal(pedalKind: PedalKind, valueU16: number, sampleOffset = 0): void {
    this.postToWorklet({ type: 'pedal', pedalKind, valueU16, sampleOffset });
  }

  allNotesOff(): void {
    this.postToWorklet({ type: 'allNotesOff' });
  }

  async dispose(): Promise<void> {
    if (this.workletNode) {
      this.postToWorklet({ type: 'destroy' });
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    if (this.analyser) {
      this.analyser.disconnect();
      this.analyser = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
    this.telemetry.reset();
  }

  private postToWorklet(msg: WorkletInboundMessage): void {
    if (this.workletNode) {
      this.workletNode.port.postMessage(msg);
    }
  }

  private async initializePipeline(): Promise<void> {
    this.telemetry.setStatus('initializing');
    let audioContext: AudioContext | null = null;
    let workletNode: AudioWorkletNode | null = null;
    let analyser: AnalyserNode | null = null;

    try {
      const AudioCtx =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('Web Audio API is not supported in this browser.');
      }

      audioContext = new AudioCtx({ latencyHint: 'interactive' });
      audioContext.onstatechange = () => {
        this.telemetry.syncContextState();
      };
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      if (typeof audioContext.audioWorklet === 'undefined') {
        throw new Error('AudioWorklet is unavailable in this execution context.');
      }

      const blobUrl = URL.createObjectURL(
        new Blob([workletCode], { type: 'application/javascript' }),
      );
      try {
        await audioContext.audioWorklet.addModule(blobUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }

      const wasmBytes = await fetchWasmBinary();

      workletNode = new AudioWorkletNode(audioContext, WORKLET_PROCESSOR_NAME, {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });

      analyser = audioContext.createAnalyser();
      analyser.fftSize = ANALYSER_FFT_SIZE;
      analyser.smoothingTimeConstant = 0;

      // Passive diagnostic tap directly inline before AudioContext destination.
      workletNode.connect(analyser);
      analyser.connect(audioContext.destination);

      this.audioContext = audioContext;
      this.workletNode = workletNode;
      this.analyser = analyser;
      this.telemetry.setAudioGraph(audioContext, analyser, workletNode);

      await this.handshakeWorklet(workletNode, wasmBytes);
      this.telemetry.syncContextState();
      this.telemetry.setStatus('ready');
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (workletNode) {
        workletNode.disconnect();
      }
      if (analyser) {
        analyser.disconnect();
      }
      if (audioContext) {
        await audioContext.close().catch(() => undefined);
      }
      this.audioContext = null;
      this.workletNode = null;
      this.analyser = null;
      this.telemetry.setAudioGraph(null, null, null);
      this.telemetry.setStatus('error', message);
      throw err;
    }
  }

  private handshakeWorklet(node: AudioWorkletNode, wasmBytes: ArrayBuffer): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = window.setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error('Timed out waiting for Wasm AudioWorklet initialization.'));
        }
      }, INIT_TIMEOUT_MS);

      node.onprocessorerror = (event) => {
        const msg = `AudioWorklet processor error: ${String(event)}`;
        this.telemetry.setStatus('error', msg);
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          reject(new Error(msg));
        }
      };

      node.port.onmessage = (event: MessageEvent<WorkletOutboundMessage>) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          if (!settled) {
            settled = true;
            window.clearTimeout(timer);
            resolve();
          }
          return;
        }
        if (msg.type === 'error') {
          this.telemetry.setStatus('error', msg.message);
          if (!settled) {
            settled = true;
            window.clearTimeout(timer);
            reject(new Error(msg.message));
          }
          return;
        }
        if (msg.type === 'telemetry') {
          this.telemetry.updateTelemetry({
            activeVoices: msg.activeVoices,
            peakL: msg.peakL,
            peakR: msg.peakR,
            rmsMono: msg.rmsMono,
            callbackUs: msg.callbackUs,
            maxCallbackUs: msg.maxCallbackUs,
            blockCount: msg.blockCount,
          });
        }
      };

      node.port.postMessage({ type: 'init', wasmBytes } satisfies WorkletInboundMessage, [
        wasmBytes,
      ]);
    });
  }
}
