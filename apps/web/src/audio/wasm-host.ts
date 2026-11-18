import {
  A4_FREQ_DEFAULT,
  BODY_CONCERT_GRAND,
  CONDITION_DEFAULT,
  HAMMER_HARDNESS_DEFAULT,
  LID_OPEN,
  PERSPECTIVE_PLAYER,
  REVERB_DECAY_DEFAULT,
  REVERB_DECAY_MAX,
  REVERB_DECAY_MIN,
  REVERB_MIX_DEFAULT,
  REVERB_MIX_MAX,
  REVERB_MIX_MIN,
  STRING_SCALE_DEFAULT,
  STRETCH_TUNING_DEFAULT,
  TEMPERAMENT_EQUAL,
  UNISON_DETUNE_DEFAULT,
  WEB_AUDIO_SAMPLE_RATE,
  type BodyId,
  type LidState,
  type ListenerPerspective,
  type PedalKind,
  type PrepInboundMessage,
  type TemperamentId,
  type WavFormat,
  type WorkletInboundMessage,
} from './abi';
import { bootstrapAudioPipeline } from './pipeline-bootstrap';
import { TelemetryClient } from './telemetry-client';
import { WavRecorder } from './wav-recorder';

const MAX_PENDING_BEFORE_READY = 256;

const clampReverb = (v: number, lo: number, hi: number, fallback: number): number =>
  Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback;

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
    a4: A4_FREQ_DEFAULT,
    temperamentId: TEMPERAMENT_EQUAL as TemperamentId,
    condition: CONDITION_DEFAULT,
  };
  private voicingSeq = 0;
  private voicingInFlight = false;
  private pendingVoicingReq: Extract<PrepInboundMessage, { type: 'prepVoicing' }> | null = null;

  private readonly wavRecorder = new WavRecorder();
  private stopRecordingResolver: (() => void) | null = null;
  private recUpdateListener: ((frames: number, durationSec: number, recording: boolean) => void) | null =
    null;

  private lastSpace: { body: BodyId; persp: ListenerPerspective; lid: LidState } = {
    body: BODY_CONCERT_GRAND,
    persp: PERSPECTIVE_PLAYER,
    lid: LID_OPEN,
  };
  private spaceSeq = 0;
  private spaceInFlight = false;
  private pendingSpaceReq: Extract<PrepInboundMessage, { type: 'prepSpace' }> | null = null;

  private lastReverb = {
    decayS: REVERB_DECAY_DEFAULT,
    mix: REVERB_MIX_DEFAULT,
  };

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
    a4FreqHz = A4_FREQ_DEFAULT,
    temperamentId: TemperamentId = TEMPERAMENT_EQUAL,
    condition = CONDITION_DEFAULT,
  ): void {
    this.enqueueOrRun(() => {
      if (hammerHardness !== this.lastHammer) {
        this.lastHammer = hammerHardness;
        this.postToWorklet({ type: 'setHammer', hammerHardness });
      }
      const tuningChanged =
        unisonDetune !== this.lastTuning.unison ||
        stretchTuning !== this.lastTuning.stretch ||
        stringLengthScale !== this.lastTuning.scale ||
        a4FreqHz !== this.lastTuning.a4 ||
        temperamentId !== this.lastTuning.temperamentId ||
        condition !== this.lastTuning.condition;
      if (!tuningChanged) {
        return;
      }
      this.lastTuning = {
        unison: unisonDetune,
        stretch: stretchTuning,
        scale: stringLengthScale,
        a4: a4FreqHz,
        temperamentId,
        condition,
      };
      if (!this.prepWorker) {
        this.postToWorklet({
          type: 'setVoicing',
          hammerHardness,
          unisonDetune,
          stretchTuning,
          stringLengthScale,
          a4FreqHz,
          temperamentId,
          condition,
        });
        return;
      }
      const req: Extract<PrepInboundMessage, { type: 'prepVoicing' }> = {
        type: 'prepVoicing',
        hammerHardness,
        unisonDetune,
        stretchTuning,
        stringLengthScale,
        a4FreqHz,
        temperamentId,
        condition,
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

  onRecordingUpdate(
    cb: ((frames: number, durationSec: number, recording: boolean) => void) | null,
  ): void {
    this.recUpdateListener = cb;
  }

  startRecording(): void {
    const sr = this.audioContext?.sampleRate ?? WEB_AUDIO_SAMPLE_RATE;
    this.wavRecorder.start(sr);
    this.enqueueOrRun(() => {
      this.postToWorklet({ type: 'setRecording', active: true });
    });
    this.notifyRecUpdate();
  }

  stopRecording(): Promise<void> {
    if (!this.wavRecorder.isRecording()) {
      return Promise.resolve();
    }
    this.wavRecorder.stop();
    if (!this.pipelineReady || !this.workletNode) {
      this.notifyRecUpdate();
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        this.stopRecordingResolver = null;
        this.notifyRecUpdate();
        resolve();
      };
      this.stopRecordingResolver = finish;
      this.postToWorklet({ type: 'setRecording', active: false });
      window.setTimeout(finish, 250);
    });
  }

  isRecording(): boolean {
    return this.wavRecorder.isRecording();
  }

  getRecordingFrameCount(): number {
    return this.wavRecorder.getFrameCount();
  }

  getRecordingDurationSeconds(): number {
    return this.wavRecorder.getDurationSeconds();
  }

  exportWavBlob(format: WavFormat): Blob | null {
    return this.wavRecorder.createBlob(format);
  }

  private notifyRecUpdate(): void {
    this.recUpdateListener?.(
      this.wavRecorder.getFrameCount(),
      this.wavRecorder.getDurationSeconds(),
      this.wavRecorder.isRecording(),
    );
  }

  setSpace(
    perspective: ListenerPerspective,
    lidState: LidState,
    bodyId?: BodyId,
  ): void {
    this.enqueueOrRun(() => {
      const targetBody = bodyId ?? this.lastSpace.body;
      if (
        targetBody === this.lastSpace.body &&
        perspective === this.lastSpace.persp &&
        lidState === this.lastSpace.lid
      ) {
        return;
      }
      this.lastSpace = { body: targetBody, persp: perspective, lid: lidState };
      if (!this.prepWorker) {
        this.postToWorklet({ type: 'setSpace', bodyId: targetBody, perspective, lidState });
        return;
      }
      const req: Extract<PrepInboundMessage, { type: 'prepSpace' }> = {
        type: 'prepSpace',
        bodyId: targetBody,
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

  setReverb(decayS: number, mix: number): void {
    const clampedDecay = clampReverb(
      decayS,
      REVERB_DECAY_MIN,
      REVERB_DECAY_MAX,
      REVERB_DECAY_DEFAULT,
    );
    const clampedMix = clampReverb(mix, REVERB_MIX_MIN, REVERB_MIX_MAX, REVERB_MIX_DEFAULT);
    this.enqueueOrRun(() => {
      if (clampedDecay === this.lastReverb.decayS && clampedMix === this.lastReverb.mix) {
        return;
      }
      this.lastReverb = { decayS: clampedDecay, mix: clampedMix };
      this.postToWorklet({ type: 'setReverb', decayS: clampedDecay, mix: clampedMix });
    });
  }

  async dispose(): Promise<void> {
    this.pipelineReady = false;
    this.pendingBeforeReady.length = 0;
    this.pendingVoicingReq = null;
    this.pendingSpaceReq = null;
    this.voicingInFlight = false;
    this.spaceInFlight = false;
    this.stopRecordingResolver?.();
    this.stopRecordingResolver = null;
    this.wavRecorder.clear();
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
    try {
      const handles = await bootstrapAudioPipeline({
        telemetry: this.telemetry,
        onSpaceDone: () => {
          if (this.pendingSpaceReq && this.prepWorker) {
            const next = this.pendingSpaceReq;
            this.pendingSpaceReq = null;
            this.prepWorker.postMessage(next);
          } else {
            this.spaceInFlight = false;
          }
        },
        onVoicingDone: () => {
          if (this.pendingVoicingReq && this.prepWorker) {
            const next = this.pendingVoicingReq;
            this.pendingVoicingReq = null;
            this.prepWorker.postMessage(next);
          } else {
            this.voicingInFlight = false;
          }
        },
        onRecChunk: (samples, frames) => {
          const view = new Float32Array(samples, 0, frames * 2);
          this.wavRecorder.pushInterleavedChunk(view);
          if (this.workletNode) {
            this.workletNode.port.postMessage(
              { type: 'recReturn', samples } satisfies WorkletInboundMessage,
              [samples],
            );
          }
          this.notifyRecUpdate();
        },
        onRecStopped: () => {
          this.stopRecordingResolver?.();
        },
      });
      this.audioContext = handles.audioContext;
      this.workletNode = handles.workletNode;
      this.prepWorker = handles.prepWorker;
      this.analyser = handles.analyser;
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
      this.audioContext = null;
      this.workletNode = null;
      this.prepWorker = null;
      this.analyser = null;
      this.telemetry.setStatus('error', message);
      throw err;
    }
  }
}
