import {
  HAMMER_HARDNESS_DEFAULT,
  LID_OPEN,
  PERSPECTIVE_PLAYER,
  STRING_SCALE_DEFAULT,
  STRETCH_TUNING_DEFAULT,
  UNISON_DETUNE_DEFAULT,
  WEB_AUDIO_SAMPLE_RATE,
  WORKLET_PROCESSOR_NAME,
  type LidState,
  type ListenerPerspective,
  type PedalKind,
  type PrepInboundMessage,
  type PrepOutboundMessage,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
} from './abi';
import prepWorkerCode from './prep-worker.ts?blobworker';
import { TelemetryClient } from './telemetry-client';
import { compileWasmModule } from './wasm-loader';
import workletCode from './worklet-processor.ts?worklet';

const INIT_TIMEOUT_MS = 10_000;
const ANALYSER_FFT_SIZE = 2048;
const MAX_PENDING_BEFORE_READY = 256;

/** Manages 48 kHz AudioContext lifecycle, Wasm Worklet, and off-thread Prep Web Worker. */
export class WasmAudioHost {
  private readonly telemetry: TelemetryClient;
  private audioContext: AudioContext | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private prepWorker: Worker | null = null;
  private analyser: AnalyserNode | null = null;
  private initPromise: Promise<void> | null = null;
  private pipelineReady = false;
  private readonly pendingBeforeReady: Array<() => void> = [];

  private lastHammer = HAMMER_HARDNESS_DEFAULT;
  private lastTuning = {
    unison: UNISON_DETUNE_DEFAULT,
    stretch: STRETCH_TUNING_DEFAULT,
    scale: STRING_SCALE_DEFAULT,
  };
  private voicingSeq = 0;
  private voicingInFlight = false;
  private pendingVoicingReq: Extract<PrepInboundMessage, { type: 'prepVoicing' }> | null = null;

  private lastSpace: { persp: ListenerPerspective; lid: LidState } = {
    persp: PERSPECTIVE_PLAYER,
    lid: LID_OPEN,
  };
  private spaceSeq = 0;
  private spaceInFlight = false;
  private pendingSpaceReq: Extract<PrepInboundMessage, { type: 'prepSpace' }> | null = null;

  constructor(telemetry: TelemetryClient) {
    this.telemetry = telemetry;
  }

  isReady(): boolean {
    return (
      this.pipelineReady &&
      this.workletNode !== null &&
      this.telemetry.getDiagnostics().status === 'ready'
    );
  }

  getAudioContext(): AudioContext | null {
    return this.audioContext;
  }

  getAnalyser(): AnalyserNode | null {
    return this.analyser;
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

  private enqueueOrRun(fn: () => void): void {
    if (this.pipelineReady && this.workletNode) {
      fn();
    } else if (this.pendingBeforeReady.length < MAX_PENDING_BEFORE_READY) {
      this.pendingBeforeReady.push(fn);
    }
  }

  noteOn(midiKey: number, velocityU16: number, sampleOffset = 0): void {
    this.enqueueOrRun(() => {
      this.postToWorklet({ type: 'noteOn', midiKey, velocityU16, sampleOffset });
    });
  }

  noteOff(midiKey: number, releaseVelU16 = 0, sampleOffset = 0): void {
    this.enqueueOrRun(() => {
      this.postToWorklet({ type: 'noteOff', midiKey, releaseVelU16, sampleOffset });
    });
  }

  pedal(pedalKind: PedalKind, valueU16: number, sampleOffset = 0): void {
    this.enqueueOrRun(() => {
      this.postToWorklet({ type: 'pedal', pedalKind, valueU16, sampleOffset });
    });
  }

  allNotesOff(): void {
    if (!this.pipelineReady) {
      this.pendingBeforeReady.length = 0;
      return;
    }
    this.postToWorklet({ type: 'allNotesOff' });
  }

  setVoicing(
    hammerHardness: number,
    unisonDetune: number,
    stretchTuning: number,
    stringLengthScale: number,
  ): void {
    this.enqueueOrRun(() => {
      if (hammerHardness !== this.lastHammer) {
        this.lastHammer = hammerHardness;
        this.postToWorklet({ type: 'setHammer', hammerHardness });
      }
      const tuningChanged =
        unisonDetune !== this.lastTuning.unison ||
        stretchTuning !== this.lastTuning.stretch ||
        stringLengthScale !== this.lastTuning.scale;
      if (!tuningChanged || !this.prepWorker) {
        return;
      }
      this.lastTuning = {
        unison: unisonDetune,
        stretch: stretchTuning,
        scale: stringLengthScale,
      };
      const req: Extract<PrepInboundMessage, { type: 'prepVoicing' }> = {
        type: 'prepVoicing',
        hammerHardness,
        unisonDetune,
        stretchTuning,
        stringLengthScale,
        seq: ++this.voicingSeq,
      };
      if (this.voicingInFlight) {
        this.pendingVoicingReq = req;
      } else {
        this.voicingInFlight = true;
        this.prepWorker.postMessage(req);
      }
    });
  }

  setSpace(perspective: ListenerPerspective, lidState: LidState): void {
    this.enqueueOrRun(() => {
      if (
        perspective === this.lastSpace.persp &&
        lidState === this.lastSpace.lid
      ) {
        return;
      }
      this.lastSpace = { persp: perspective, lid: lidState };
      if (!this.prepWorker) {
        this.postToWorklet({ type: 'setSpace', perspective, lidState });
        return;
      }
      const req: Extract<PrepInboundMessage, { type: 'prepSpace' }> = {
        type: 'prepSpace',
        perspective,
        lidState,
        seq: ++this.spaceSeq,
      };
      if (this.spaceInFlight) {
        this.pendingSpaceReq = req;
      } else {
        this.spaceInFlight = true;
        this.prepWorker.postMessage(req);
      }
    });
  }

  async dispose(): Promise<void> {
    this.pipelineReady = false;
    this.pendingBeforeReady.length = 0;
    this.pendingVoicingReq = null;
    this.pendingSpaceReq = null;
    this.voicingInFlight = false;
    this.spaceInFlight = false;
    if (this.prepWorker) {
      this.prepWorker.postMessage({ type: 'destroy' } satisfies PrepInboundMessage);
      this.prepWorker.terminate();
      this.prepWorker = null;
    }
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
    let prepWorker: Worker | null = null;
    let analyser: AnalyserNode | null = null;

    try {
      const AudioCtx =
        window.AudioContext ??
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AudioCtx) {
        throw new Error('Web Audio API is not supported in this browser.');
      }

      audioContext = new AudioCtx({
        latencyHint: 'interactive',
        sampleRate: WEB_AUDIO_SAMPLE_RATE,
      });
      audioContext.onstatechange = () => {
        this.telemetry.syncContextState();
      };
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      if (typeof audioContext.audioWorklet === 'undefined') {
        throw new Error('AudioWorklet is unavailable in this execution context.');
      }

      const workletBlobUrl = URL.createObjectURL(
        new Blob([workletCode], { type: 'application/javascript' }),
      );
      const workerBlobUrl = URL.createObjectURL(
        new Blob([prepWorkerCode], { type: 'application/javascript' }),
      );
      let wasmModule: WebAssembly.Module;
      try {
        const [, compiled] = await Promise.all([
          audioContext.audioWorklet.addModule(workletBlobUrl),
          compileWasmModule(),
        ]);
        wasmModule = compiled;
        prepWorker = new Worker(workerBlobUrl, { type: 'module' });
      } finally {
        URL.revokeObjectURL(workletBlobUrl);
        URL.revokeObjectURL(workerBlobUrl);
      }

      workletNode = new AudioWorkletNode(audioContext, WORKLET_PROCESSOR_NAME, {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });

      analyser = audioContext.createAnalyser();
      analyser.fftSize = ANALYSER_FFT_SIZE;
      analyser.smoothingTimeConstant = 0;
      workletNode.connect(analyser);
      analyser.connect(audioContext.destination);

      this.audioContext = audioContext;
      this.workletNode = workletNode;
      this.prepWorker = prepWorker;
      this.analyser = analyser;
      this.telemetry.setAudioGraph(audioContext, analyser, workletNode);

      const channel = new MessageChannel();
      await Promise.all([
        this.handshakeWorklet(workletNode, wasmModule, channel.port2),
        this.handshakePrepWorker(prepWorker, wasmModule, audioContext.sampleRate, channel.port1),
      ]);

      this.pipelineReady = true;
      this.telemetry.syncContextState();
      this.telemetry.setStatus('ready');

      const queued = this.pendingBeforeReady.splice(0);
      for (const run of queued) {
        run();
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.pipelineReady = false;
      this.pendingBeforeReady.length = 0;
      if (prepWorker) {
        prepWorker.terminate();
      }
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
      this.prepWorker = null;
      this.analyser = null;
      this.telemetry.setAudioGraph(null, null, null);
      this.telemetry.setStatus('error', message);
      throw err;
    }
  }

  private handshakeWorklet(
    node: AudioWorkletNode,
    wasmModule: WebAssembly.Module,
    prepPort: MessagePort,
  ): Promise<void> {
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
            p50Us: msg.p50Us,
            p99Us: msg.p99Us,
            overrunCount: msg.overrunCount,
            clipped: msg.clipped,
            blockCount: msg.blockCount,
          });
        }
      };

      node.port.postMessage(
        { type: 'init', wasmModule, prepPort } satisfies WorkletInboundMessage,
        [prepPort],
      );
    });
  }

  private handshakePrepWorker(
    worker: Worker,
    wasmModule: WebAssembly.Module,
    sampleRate: number,
    workletPort: MessagePort,
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = window.setTimeout(() => {
        if (!settled) {
          settled = true;
          reject(new Error('Timed out waiting for Prep Web Worker initialization.'));
        }
      }, INIT_TIMEOUT_MS);

      worker.onerror = (ev) => {
        const msg = `Prep Worker error: ${ev.message}`;
        if (!settled) {
          settled = true;
          window.clearTimeout(timer);
          reject(new Error(msg));
        }
      };

      worker.onmessage = (event: MessageEvent<PrepOutboundMessage>) => {
        const msg = event.data;
        if (msg.type === 'ready') {
          if (!settled) {
            settled = true;
            window.clearTimeout(timer);
            resolve();
          }
          return;
        }
        if (msg.type === 'spaceDone') {
          if (this.pendingSpaceReq && this.prepWorker) {
            const next = this.pendingSpaceReq;
            this.pendingSpaceReq = null;
            this.prepWorker.postMessage(next);
          } else {
            this.spaceInFlight = false;
          }
          return;
        }
        if (msg.type === 'voicingDone') {
          if (this.pendingVoicingReq && this.prepWorker) {
            const next = this.pendingVoicingReq;
            this.pendingVoicingReq = null;
            this.prepWorker.postMessage(next);
          } else {
            this.voicingInFlight = false;
          }
          return;
        }
        if (msg.type === 'error') {
          if (!settled) {
            settled = true;
            window.clearTimeout(timer);
            reject(new Error(msg.message));
          }
        }
      };

      worker.postMessage(
        { type: 'init', wasmModule, sampleRate, workletPort } satisfies PrepInboundMessage,
        [workletPort],
      );
    });
  }
}
