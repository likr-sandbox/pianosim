import {
  A4_FREQ_DEFAULT,
  A4_FREQ_MAX,
  A4_FREQ_MIN,
  BODY_CONCERT_GRAND,
  CONDITION_DEFAULT,
  CONDITION_MAX,
  CONDITION_MIN,
  DEFAULT_LIVE_PERF_PRESET,
  PERF_PRESET_CONFIGS,
  REVERB_DECAY_DEFAULT,
  REVERB_DECAY_MAX,
  REVERB_DECAY_MIN,
  REVERB_MIX_DEFAULT,
  REVERB_MIX_MAX,
  REVERB_MIX_MIN,
  type BodyId,
  type CachedWorkletState,
  type LidState,
  type ListenerPerspective,
  type PerfPresetId,
  type PerfProfileConfig,
  type PianoWasmExports,
  type PrepToWorkletMessage,
  type TemperamentId,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
} from './abi';
import { BlobStreamer } from './blob-streamer';
import { WorkletRecorderPool } from './worklet-recorder';
import { hasPerfNow, workletNowMs, WorkletTelemetryTracker } from './worklet-telemetry';
import {
  applyRuntimePerfKnobs,
  buildEngineContext,
  defaultCachedWorkletState,
  restoreCachedEngineState,
  runActive12Warmup,
  validateWasmExports,
} from './worklet-wasm-init';

declare abstract class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: AudioWorkletNodeOptions);
  abstract process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  processorCtor: new (options?: AudioWorkletNodeOptions) => AudioWorkletProcessor,
): void;

declare const sampleRate: number;

const PROCESSOR_NAME = 'piano-wasm-processor';
const MAX_FRAMES = 512;
const MAX_PENDING_MESSAGES = 256;

type Mutable<T> = { -readonly [K in keyof T]: T[K] };

const clamp = (v: number, lo: number, hi: number, fallback: number): number =>
  Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback;

function zeroOutputs(outL: Float32Array, outR: Float32Array | null): void {
  outL.fill(0);
  if (outR) outR.fill(0);
}

class PianoWasmProcessor extends AudioWorkletProcessor {
  private rawExports: WebAssembly.Exports | null = null;
  private exports: PianoWasmExports | null = null;
  private ctx = 0;
  private memBuf: ArrayBuffer | null = null;
  private memF32: Float32Array | null = null;
  private outLPtr = 0;
  private outRPtr = 0;
  private telemetryPtr = 0;
  private initializing = false;
  private destroyed = false;
  private prepPort: MessagePort | null = null;
  private readonly blobStreamer = new BlobStreamer();
  private reportedSpaceSeq = 0;
  private reportedVoicingSeq = 0;
  private activePerfConfig: PerfProfileConfig = PERF_PRESET_CONFIGS[DEFAULT_LIVE_PERF_PRESET];
  private readonly cachedState: Mutable<CachedWorkletState> = defaultCachedWorkletState();
  private readonly recorder = new WorkletRecorderPool();
  private readonly telemTracker = new WorkletTelemetryTracker();
  private readonly postTransferable = (
    msg: WorkletOutboundMessage,
    transfer?: Transferable[],
  ): void => {
    if (transfer && transfer.length > 0) this.port.postMessage(msg, transfer);
    else this.port.postMessage(msg);
  };
  private readonly syncMemForBlob = (): Float32Array =>
    this.syncMemoryView(this.cachedChunk || 128);
  private pending: WorkletInboundMessage[] = [];

  private outLView: Float32Array | null = null;
  private outRView: Float32Array | null = null;
  private cachedChunk = 0;

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent<WorkletInboundMessage>) => {
      this.onInboundMessage(event.data);
    };
  }

  private readonly postOutbound = (msg: WorkletOutboundMessage): void => {
    this.port.postMessage(msg);
  };

  private onInboundMessage(msg: WorkletInboundMessage): void {
    if (this.destroyed) return;
    if (msg.type === 'init') {
      if (msg.prepPort) this.attachPrepPort(msg.prepPort);
      void this.initWasm(msg);
      return;
    }
    if (!this.exports || this.ctx === 0) {
      if (msg.type === 'destroy') {
        this.destroyed = true;
        this.pending.length = 0;
      } else if (this.initializing && this.pending.length < MAX_PENDING_MESSAGES) {
        this.pending.push(msg);
      }
      return;
    }
    this.dispatchCommand(msg);
  }

  private attachPrepPort(port: MessagePort): void {
    if (this.prepPort) this.prepPort.close();
    this.prepPort = port;
    port.onmessage = (event: MessageEvent<PrepToWorkletMessage>) => {
      if (this.destroyed || !this.exports || this.ctx === 0) return;
      const d = event.data;
      if (d.type === 'spaceBlob') {
        this.cachedState.bodyId = (d.bodyId ?? BODY_CONCERT_GRAND) as BodyId;
        this.cachedState.perspective = d.perspective;
        this.cachedState.lidState = d.lidState;
      } else if (d.type === 'voicingBlob') {
        this.cachedState.hammerHardness = d.hammerHardness;
        this.cachedState.unisonDetune = d.unisonDetune;
        this.cachedState.stretchTuning = d.stretchTuning;
        this.cachedState.stringLengthScale = d.stringLengthScale;
        this.cachedState.a4FreqHz = d.a4FreqHz;
        this.cachedState.temperamentId = d.temperamentId;
        this.cachedState.condition = d.condition;
      }
      this.blobStreamer.onPrepBlob(d);
    };
  }

  private bindContextPointers(exports: PianoWasmExports, ctx: number): void {
    const outLPtr = exports.piano_out_l_ptr(ctx) >>> 0;
    const outRPtr = exports.piano_out_r_ptr(ctx) >>> 0;
    const telemetryPtr = exports.piano_telemetry_ptr(ctx) >>> 0;
    if (outLPtr === 0 || outRPtr === 0 || telemetryPtr === 0) {
      exports.piano_destroy(ctx);
      throw new Error('piano_*_ptr returned a null buffer pointer');
    }
    if (outLPtr === outRPtr) {
      exports.piano_destroy(ctx);
      throw new Error('piano_out_l_ptr and piano_out_r_ptr must not alias');
    }
    this.ctx = ctx;
    this.outLPtr = outLPtr;
    this.outRPtr = outRPtr;
    this.telemetryPtr = telemetryPtr;
    this.memBuf = null;
    this.memF32 = null;
    this.syncMemoryView(128);
  }

  private async initWasm(
    msg: Extract<WorkletInboundMessage, { type: 'init' }>,
  ): Promise<void> {
    if (this.initializing || this.exports !== null || this.destroyed) return;
    this.initializing = true;
    try {
      let instance: WebAssembly.Instance;
      if (msg.wasmModule) {
        instance = await WebAssembly.instantiate(msg.wasmModule, {});
      } else if (msg.wasmBytes) {
        instance = (await WebAssembly.instantiate(msg.wasmBytes, {})).instance;
      } else {
        throw new Error('AudioWorklet init requires wasmModule or wasmBytes');
      }
      const exports = validateWasmExports(instance.exports);
      const preset: PerfPresetId = msg.perfPreset ?? DEFAULT_LIVE_PERF_PRESET;
      this.activePerfConfig = msg.perfConfig ?? PERF_PRESET_CONFIGS[preset];

      const ctx = buildEngineContext(exports, sampleRate, this.activePerfConfig);
      this.rawExports = instance.exports;
      this.exports = exports;
      this.bindContextPointers(exports, ctx);
      this.initializing = false;

      runActive12Warmup(
        instance.exports,
        exports,
        ctx,
        sampleRate,
        this.syncMemForBlob,
        this.activePerfConfig,
      );

      for (const queued of this.pending) {
        if (this.destroyed) break;
        this.dispatchCommand(queued);
      }
      this.pending.length = 0;
      if (!this.destroyed) this.postOutbound({ type: 'ready', sampleRate });
    } catch (err) {
      this.initializing = false;
      this.pending.length = 0;
      this.postOutbound({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  }

  private syncMemoryView(chunk = 128): Float32Array {
    const exports = this.exports!;
    const buf = exports.memory.buffer;
    if (buf !== this.memBuf || this.memF32 === null || this.cachedChunk !== chunk) {
      this.memBuf = buf;
      this.memF32 = new Float32Array(buf);
      this.outLPtr = exports.piano_out_l_ptr(this.ctx) >>> 0;
      this.outRPtr = exports.piano_out_r_ptr(this.ctx) >>> 0;
      this.telemetryPtr = exports.piano_telemetry_ptr(this.ctx) >>> 0;
      const lBase = this.outLPtr >>> 2;
      const rBase = this.outRPtr >>> 2;
      this.outLView = this.memF32.subarray(lBase, lBase + chunk);
      this.outRView = this.memF32.subarray(rBase, rBase + chunk);
      this.cachedChunk = chunk;
    }
    return this.memF32;
  }

  private dispatchCommand(msg: WorkletInboundMessage): void {
    const exports = this.exports;
    const ctx = this.ctx;
    if (!exports || ctx === 0) return;

    switch (msg.type) {
      case 'noteOn':
        exports.piano_note_on(ctx, msg.midiKey & 0x7f, msg.velocityU16 & 0xffff, msg.sampleOffset >>> 0);
        break;
      case 'noteOff':
        exports.piano_note_off(ctx, msg.midiKey & 0x7f, msg.releaseVelU16 & 0xffff, msg.sampleOffset >>> 0);
        break;
      case 'pedal':
        exports.piano_pedal(ctx, msg.pedalKind & 0xff, msg.valueU16 & 0xffff, msg.sampleOffset >>> 0);
        break;
      case 'allNotesOff':
        exports.piano_all_notes_off(ctx);
        break;
      case 'setHammer': {
        const h = clamp(msg.hammerHardness, 0.0, 1.0, 0.5);
        this.cachedState.hammerHardness = h;
        exports.piano_set_hammer_hardness(ctx, h);
        break;
      }
      case 'setVoicing': {
        const s = this.cachedState;
        s.hammerHardness = clamp(msg.hammerHardness, 0.0, 1.0, 0.5);
        s.unisonDetune = clamp(msg.unisonDetune, 0.0, 4.0, 1.0);
        s.stretchTuning = clamp(msg.stretchTuning, 0.0, 2.0, 1.0);
        s.stringLengthScale = clamp(msg.stringLengthScale, 0.5, 2.0, 1.0);
        s.a4FreqHz = clamp(msg.a4FreqHz, A4_FREQ_MIN, A4_FREQ_MAX, A4_FREQ_DEFAULT);
        s.temperamentId = ((msg.temperamentId >>> 0) & 0x3) as TemperamentId;
        s.condition = clamp(msg.condition, CONDITION_MIN, CONDITION_MAX, CONDITION_DEFAULT);
        exports.piano_set_voicing_full(
          ctx, s.hammerHardness, s.unisonDetune, s.stretchTuning,
          s.stringLengthScale, s.a4FreqHz, s.temperamentId, s.condition,
        );
        break;
      }
      case 'setConcertPitch': {
        const a4 = clamp(msg.concertPitchHz, A4_FREQ_MIN, A4_FREQ_MAX, A4_FREQ_DEFAULT);
        this.cachedState.a4FreqHz = a4;
        exports.piano_set_concert_pitch(ctx, a4);
        break;
      }
      case 'setTemperament': {
        const t = ((msg.temperamentId >>> 0) & 0x3) as TemperamentId;
        this.cachedState.temperamentId = t;
        exports.piano_set_temperament(ctx, t);
        break;
      }
      case 'setCondition': {
        const c = clamp(msg.condition, CONDITION_MIN, CONDITION_MAX, CONDITION_DEFAULT);
        this.cachedState.condition = c;
        exports.piano_set_condition(ctx, c);
        break;
      }
      case 'setRecording':
        this.recorder.setActive(msg.active, this.postTransferable);
        break;
      case 'recReturn':
        this.recorder.recycleBuffer(msg.samples);
        break;
      case 'setSpace': {
        const bodyId = ((msg.bodyId ?? BODY_CONCERT_GRAND) & 0x3) as BodyId;
        const persp = ((msg.perspective >>> 0) & 0x3) as ListenerPerspective;
        const lid = ((msg.lidState >>> 0) & 0x3) as LidState;
        this.cachedState.bodyId = bodyId;
        this.cachedState.perspective = persp;
        this.cachedState.lidState = lid;
        exports.piano_set_body_space(ctx, bodyId, persp, lid);
        break;
      }
      case 'setReverb': {
        const decayS = clamp(msg.decayS, REVERB_DECAY_MIN, REVERB_DECAY_MAX, REVERB_DECAY_DEFAULT);
        const mix = clamp(msg.mix, REVERB_MIX_MIN, REVERB_MIX_MAX, REVERB_MIX_DEFAULT);
        this.cachedState.reverbDecayS = decayS;
        this.cachedState.reverbMix = mix;
        exports.piano_set_reverb(ctx, decayS, mix);
        break;
      }
      case 'setPerfKnob':
        this.activePerfConfig = msg.perfConfig;
        exports.piano_set_perf_knob(ctx, msg.knobId, msg.value);
        this.postOutbound({
          type: 'perfApplied', perfSeq: msg.seq, presetId: msg.presetId, perfConfig: msg.perfConfig,
        });
        break;
      case 'setPerfProfile':
        this.handleSetPerfProfile(msg);
        break;
      case 'destroy':
        exports.piano_destroy(ctx);
        this.ctx = 0;
        this.rawExports = null;
        this.exports = null;
        this.memBuf = null;
        this.memF32 = null;
        this.outLView = null;
        this.outRView = null;
        this.cachedChunk = 0;
        this.blobStreamer.reset();
        this.reportedSpaceSeq = 0;
        this.reportedVoicingSeq = 0;
        this.recorder.reset();
        if (this.prepPort) {
          this.prepPort.close();
          this.prepPort = null;
        }
        this.destroyed = true;
        break;
      case 'init':
        break;
    }
  }

  private handleSetPerfProfile(
    msg: Extract<WorkletInboundMessage, { type: 'setPerfProfile' }>,
  ): void {
    const exports = this.exports;
    if (!exports || this.ctx === 0) return;
    this.activePerfConfig = msg.perfConfig;

    if (!msg.rebuildRequired) {
      applyRuntimePerfKnobs(exports, this.ctx, msg.perfConfig);
      this.postOutbound({
        type: 'perfApplied', perfSeq: msg.seq, presetId: msg.presetId, perfConfig: msg.perfConfig,
      });
      return;
    }

    if (msg.cachedState) {
      const s = msg.cachedState;
      this.cachedState.bodyId = s.bodyId;
      this.cachedState.perspective = s.perspective;
      this.cachedState.lidState = s.lidState;
      this.cachedState.reverbDecayS = s.reverbDecayS;
      this.cachedState.reverbMix = s.reverbMix;
      this.cachedState.hammerHardness = s.hammerHardness;
      this.cachedState.unisonDetune = s.unisonDetune;
      this.cachedState.stretchTuning = s.stretchTuning;
      this.cachedState.stringLengthScale = s.stringLengthScale;
      this.cachedState.a4FreqHz = s.a4FreqHz;
      this.cachedState.temperamentId = s.temperamentId;
      this.cachedState.condition = s.condition;
    }
    this.blobStreamer.reset(msg.minSpaceSeq ?? 0, msg.minVoicingSeq ?? 0);
    exports.piano_destroy(this.ctx);
    const nextCtx = buildEngineContext(exports, sampleRate, msg.perfConfig);
    this.bindContextPointers(exports, nextCtx);
    if (this.rawExports) {
      runActive12Warmup(
        this.rawExports, exports, nextCtx, sampleRate, this.syncMemForBlob, msg.perfConfig,
      );
    }
    restoreCachedEngineState(exports, nextCtx, this.cachedState);
    this.postOutbound({
      type: 'perfApplied', perfSeq: msg.seq, presetId: msg.presetId, perfConfig: msg.perfConfig,
    });
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    if (this.destroyed) return false;
    const outputGroup = outputs[0];
    if (!outputGroup || outputGroup.length === 0) return true;
    const outL = outputGroup[0];
    const outR = outputGroup.length > 1 ? outputGroup[1] : null;
    if (!this.exports || this.ctx === 0) {
      zeroOutputs(outL, outR);
      return true;
    }

    const exports = this.exports;
    const ctx = this.ctx;
    const totalFrames = outL.length;
    const perfAvailable = hasPerfNow();
    const t0 = workletNowMs();
    this.telemTracker.recordProcessStart(t0);

    this.blobStreamer.stepStreamingBlobs(exports, ctx, this.syncMemForBlob);
    const spSeq = this.blobStreamer.getCommittedSpaceSeq();
    const vcSeq = this.blobStreamer.getCommittedVoicingSeq();
    if (spSeq !== this.reportedSpaceSeq || vcSeq !== this.reportedVoicingSeq) {
      this.reportedSpaceSeq = spSeq;
      this.reportedVoicingSeq = vcSeq;
      this.postOutbound({ type: 'prepApplied', spaceSeq: spSeq, voicingSeq: vcSeq });
    }
    let offset = 0;
    let mem = this.memF32;
    while (offset < totalFrames) {
      const remaining = totalFrames - offset;
      const chunk = remaining > MAX_FRAMES ? MAX_FRAMES : remaining;
      exports.piano_render(ctx, chunk);
      mem = this.syncMemoryView(chunk);
      if (this.outLView) outL.set(this.outLView, offset);
      if (outR && this.outRView) outR.set(this.outRView, offset);
      offset += chunk;
    }
    const dtUs = Math.max(0, (workletNowMs() - t0) * 1000.0);
    if (this.recorder.isActive()) {
      this.recorder.captureBlock(outL, outR, totalFrames, this.postTransferable);
    }
    const activeVoices = mem ? mem[this.telemetryPtr >>> 2] >>> 0 : 0;
    this.telemTracker.recordBlock(
      outL, outR, totalFrames, sampleRate, dtUs, perfAvailable,
      activeVoices, mem !== null, this.postOutbound,
    );
    return true;
  }
}

registerProcessor(PROCESSOR_NAME, PianoWasmProcessor);
