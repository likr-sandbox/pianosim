import type {
  PianoWasmExports,
  PrepToWorkletMessage,
  WorkletInboundMessage,
  WorkletOutboundMessage,
} from './abi';

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
const TELEMETRY_STRIDE_BLOCKS = 32;
const MAX_PENDING_MESSAGES = 256;
const BLOB_STREAM_CHUNK = 8_192;
const ACTIVE12_WARMUP_KEYS = [21, 28, 36, 43, 48, 55, 60, 64, 67, 76, 88, 100];

const REQUIRED_EXPORTS: ReadonlyArray<keyof PianoWasmExports> = [
  'memory', 'piano_init', 'piano_destroy', 'piano_note_on', 'piano_note_off', 'piano_pedal',
  'piano_all_notes_off', 'piano_set_voicing', 'piano_set_hammer_hardness', 'piano_set_space',
  'piano_staging_ptr', 'piano_staging_offset_ptr', 'piano_stage_space_chunk',
  'piano_commit_space', 'piano_is_space_crossfading', 'piano_apply_space_blob',
  'piano_apply_voicing_blob', 'piano_render', 'piano_out_l_ptr', 'piano_out_r_ptr',
  'piano_telemetry_ptr',
];

const clamp = (v: number, lo: number, hi: number, fallback: number): number =>
  Number.isFinite(v) ? Math.max(lo, Math.min(hi, v)) : fallback;

const hasPerf = (): boolean => {
  const p = typeof performance !== 'undefined' ? performance : globalThis?.performance;
  return typeof p?.now === 'function';
};
const nowMs = (): number => (hasPerf() ? performance.now() : Date.now());

function validateWasmExports(raw: WebAssembly.Exports): PianoWasmExports {
  for (const name of REQUIRED_EXPORTS) {
    const val = raw[name];
    if (name === 'memory' ? !(val instanceof WebAssembly.Memory) : typeof val !== 'function') {
      throw new Error(`Missing or invalid Wasm export: ${name}`);
    }
  }
  return raw as PianoWasmExports;
}

function zeroOutputs(outL: Float32Array, outR: Float32Array | null): void {
  outL.fill(0);
  if (outR) outR.fill(0);
}

class PianoWasmProcessor extends AudioWorkletProcessor {
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
  private lastSpaceSeq = -1;
  private lastVoicingSeq = -1;
  private pendingSpace: { persp: number; lid: number; src: Float32Array; off: number } | null = null;
  private pendingVoicing: {
    hammer: number; detune: number; stretch: number; scale: number; a4: number; src: Float32Array; off: number;
  } | null = null;
  private pending: WorkletInboundMessage[] = [];

  private blockCount = 0;
  private strideBlockCount = 0;
  private windowSumUs = 0;
  private windowMaxUs = 0;
  private windowPeakL = 0;
  private windowPeakR = 0;
  private windowSqSum = 0;
  private windowSamples = 0;
  private windowClipped = false;
  private totalOverruns = 0;

  private outLView: Float32Array | null = null;
  private outRView: Float32Array | null = null;
  private cachedChunk = 0;

  constructor() {
    super();
    this.port.onmessage = (event: MessageEvent<WorkletInboundMessage>) => {
      this.onInboundMessage(event.data);
    };
  }

  private postOutbound(msg: WorkletOutboundMessage): void {
    this.port.postMessage(msg);
  }

  private onInboundMessage(msg: WorkletInboundMessage): void {
    if (this.destroyed) {
      return;
    }
    if (msg.type === 'init') {
      if (msg.prepPort) {
        this.attachPrepPort(msg.prepPort);
      }
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
    if (this.prepPort) {
      this.prepPort.close();
    }
    this.prepPort = port;
    port.onmessage = (event: MessageEvent<PrepToWorkletMessage>) => {
      this.onPrepBlob(event.data);
    };
  }

  private onPrepBlob(msg: PrepToWorkletMessage): void {
    if (this.destroyed || !this.exports || this.ctx === 0) return;
    if (msg.type === 'spaceBlob') {
      if (msg.seq < this.lastSpaceSeq) return;
      this.lastSpaceSeq = msg.seq;
      this.pendingSpace = {
        persp: msg.perspective >>> 0,
        lid: msg.lidState >>> 0,
        src: new Float32Array(msg.blob),
        off: 0,
      };
      return;
    }
    if (msg.type === 'voicingBlob') {
      if (msg.seq < this.lastVoicingSeq) return;
      this.lastVoicingSeq = msg.seq;
      this.pendingVoicing = {
        hammer: msg.hammerHardness,
        detune: msg.unisonDetune,
        stretch: msg.stretchTuning,
        scale: msg.stringLengthScale,
        a4: msg.a4FreqHz,
        src: new Float32Array(msg.blob),
        off: 0,
      };
    }
  }

  private stepStreamingBlobs(exports: PianoWasmExports, ctx: number): void {
    if (this.pendingSpace) {
      if (exports.piano_is_space_crossfading(ctx) !== 0) return;
      const b = this.pendingSpace;
      const n = Math.min(BLOB_STREAM_CHUNK, b.src.length - b.off);
      const ptr = exports.piano_staging_ptr(ctx, n) >>> 0;
      if (ptr === 0) {
        this.pendingSpace = null;
        return;
      }
      const mem = this.syncMemoryView(this.cachedChunk || 128);
      mem.set(b.src.subarray(b.off, b.off + n), ptr >>> 2);
      if (exports.piano_stage_space_chunk(ctx, b.persp, b.lid, b.off, n, b.src.length) !== 0) {
        this.pendingSpace = null;
        return;
      }
      b.off += n;
      if (b.off >= b.src.length) {
        exports.piano_commit_space(ctx, b.persp, b.lid);
        this.pendingSpace = null;
      }
      return;
    }
    if (this.pendingVoicing) {
      const b = this.pendingVoicing;
      const n = Math.min(BLOB_STREAM_CHUNK, b.src.length - b.off);
      const ptr = exports.piano_staging_offset_ptr(ctx, b.off, n) >>> 0;
      if (ptr === 0) {
        this.pendingVoicing = null;
        return;
      }
      const mem = this.syncMemoryView(this.cachedChunk || 128);
      mem.set(b.src.subarray(b.off, b.off + n), ptr >>> 2);
      b.off += n;
      if (b.off >= b.src.length) {
        exports.piano_apply_voicing_blob(ctx, b.hammer, b.detune, b.stretch, b.scale, b.a4, b.src.length);
        this.pendingVoicing = null;
      }
    }
  }

  private async initWasm(
    msg: Extract<WorkletInboundMessage, { type: 'init' }>,
  ): Promise<void> {
    if (this.initializing || this.exports !== null || this.destroyed) {
      return;
    }
    this.initializing = true;
    try {
      let instance: WebAssembly.Instance;
      if (msg.wasmModule) {
        instance = await WebAssembly.instantiate(msg.wasmModule, {});
      } else if (msg.wasmBytes) {
        const res = await WebAssembly.instantiate(msg.wasmBytes, {});
        instance = res.instance;
      } else {
        throw new Error('AudioWorklet init requires wasmModule or wasmBytes');
      }
      const exports = validateWasmExports(instance.exports);
      const ctx = exports.piano_init(sampleRate) >>> 0;
      if (ctx === 0) {
        throw new Error('piano_init returned a null context pointer');
      }
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

      this.exports = exports;
      this.ctx = ctx;
      this.outLPtr = outLPtr;
      this.outRPtr = outRPtr;
      this.telemetryPtr = telemetryPtr;
      this.syncMemoryView(128);
      this.initializing = false;

      // Active-12 full-bucket JIT warmup covering open Sustain, SpaceCrossfade, hammer
      // dirty mask, sustained dry A4, and Prep staging/voicing installers before ready.
      exports.piano_pedal(ctx, 0, 40_000, 0);
      for (const midi of ACTIVE12_WARMUP_KEYS) exports.piano_note_on(ctx, midi, 42_000, 0);
      for (let i = 0; i < 120; i++) exports.piano_render(ctx, 128);
      exports.piano_set_space(ctx, 1, 1);
      for (let i = 0; i < 4; i++) exports.piano_render(ctx, 128);
      exports.piano_set_space(ctx, 0, 2);
      exports.piano_set_hammer_hardness(ctx, 0.65);
      for (let i = 0; i < 32; i++) exports.piano_render(ctx, 128);
      exports.piano_set_hammer_hardness(ctx, 0.5);
      exports.piano_pedal(ctx, 0, 0, 0);
      exports.piano_all_notes_off(ctx);
      for (let i = 0; i < 16; i++) exports.piano_render(ctx, 128);
      exports.piano_note_on(ctx, 69, 48_000, 0);
      for (let i = 0; i < 160; i++) exports.piano_render(ctx, 128);
      exports.piano_all_notes_off(ctx);
      for (let i = 0; i < 8; i++) exports.piano_render(ctx, 128);
      this.warmPrepInstallers(instance.exports, exports, ctx);

      for (const queued of this.pending) {
        if (this.destroyed) break;
        this.dispatchCommand(queued);
      }
      this.pending.length = 0;
      if (!this.destroyed) this.postOutbound({ type: 'ready', sampleRate });
    } catch (err) {
      this.initializing = false;
      this.pending.length = 0;
      const message = err instanceof Error ? err.message : String(err);
      this.postOutbound({ type: 'error', message });
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
    if (!exports || ctx === 0) {
      return;
    }
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
      case 'setHammer':
        exports.piano_set_hammer_hardness(ctx, clamp(msg.hammerHardness, 0.0, 1.0, 0.5));
        break;
      case 'setVoicing':
        exports.piano_set_voicing(
          ctx,
          clamp(msg.hammerHardness, 0.0, 1.0, 0.5),
          clamp(msg.unisonDetune, 0.0, 4.0, 1.0),
          clamp(msg.stretchTuning, 0.0, 2.0, 1.0),
          clamp(msg.stringLengthScale, 0.5, 2.0, 1.0),
          clamp(msg.a4FreqHz, 415.0, 466.0, 440.0),
        );
        break;
      case 'setSpace':
        exports.piano_set_space(ctx, (msg.perspective >>> 0) & 0x3, (msg.lidState >>> 0) & 0x3);
        break;
      case 'destroy':
        exports.piano_destroy(ctx);
        this.ctx = 0;
        this.exports = null;
        this.memBuf = null;
        this.memF32 = null;
        this.outLView = null;
        this.outRView = null;
        this.cachedChunk = 0;
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

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    if (this.destroyed) {
      return false;
    }
    const outputGroup = outputs[0];
    if (!outputGroup || outputGroup.length === 0) {
      return true;
    }
    const outL = outputGroup[0];
    const outR = outputGroup.length > 1 ? outputGroup[1] : null;
    if (!this.exports || this.ctx === 0) {
      zeroOutputs(outL, outR);
      return true;
    }

    const exports = this.exports;
    const ctx = this.ctx;
    const totalFrames = outL.length;
    const t0 = nowMs();
    this.stepStreamingBlobs(exports, ctx);
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
    const dtUs = Math.max(0, (nowMs() - t0) * 1000.0);
    const d1BudgetUs = (totalFrames / sampleRate) * 700_000.0;
    if (hasPerf() ? dtUs > d1BudgetUs : dtUs > (totalFrames / sampleRate) * 1_000_000.0) {
      this.totalOverruns += 1;
    }
    this.windowSumUs += dtUs;
    if (dtUs > this.windowMaxUs) this.windowMaxUs = dtUs;

    let blockPeakL = 0;
    let blockPeakR = 0;
    let blockSqSum = 0;
    for (let i = 0; i < totalFrames; i++) {
      const l = outL[i];
      const r = outR ? outR[i] : l;
      const al = Math.abs(l);
      const ar = Math.abs(r);
      if (al > blockPeakL) blockPeakL = al;
      if (ar > blockPeakR) blockPeakR = ar;
      const m = 0.5 * (l + r);
      blockSqSum += m * m;
    }
    if (blockPeakL > this.windowPeakL) this.windowPeakL = blockPeakL;
    if (blockPeakR > this.windowPeakR) this.windowPeakR = blockPeakR;
    if (blockPeakL >= 0.999 || blockPeakR >= 0.999) this.windowClipped = true;
    this.windowSqSum += blockSqSum;
    this.windowSamples += totalFrames;
    this.strideBlockCount += 1;
    this.blockCount += 1;

    if (this.strideBlockCount >= TELEMETRY_STRIDE_BLOCKS && mem) {
      if (!hasPerf() && this.windowSumUs / this.strideBlockCount > d1BudgetUs) {
        this.totalOverruns += 1;
      }
      this.postOutbound({
        type: 'telemetry',
        activeVoices: mem[this.telemetryPtr >>> 2] >>> 0,
        peakL: this.windowPeakL,
        peakR: this.windowPeakR,
        rmsMono: this.windowSamples > 0 ? Math.sqrt(this.windowSqSum / this.windowSamples) : 0,
        callbackUs: this.windowSumUs / this.strideBlockCount,
        maxCallbackUs: this.windowMaxUs,
        p50Us: 0,
        p99Us: 0,
        overrunCount: this.totalOverruns,
        clipped: this.windowClipped,
        blockCount: this.blockCount,
      });
      this.strideBlockCount = 0;
      this.windowSumUs = 0;
      this.windowMaxUs = 0;
      this.windowPeakL = 0;
      this.windowPeakR = 0;
      this.windowSqSum = 0;
      this.windowSamples = 0;
      this.windowClipped = false;
    }
    return true;
  }

  private warmPrepInstallers(raw: WebAssembly.Exports, exports: PianoWasmExports, ctx: number): void {
    const p = raw as Record<string, ((...a: number[]) => number) | undefined>;
    if (!p.piano_prep_init || !p.piano_prep_voicing || !p.piano_prep_space || !p.piano_prep_out_ptr) return;
    const prep = p.piano_prep_init(sampleRate) >>> 0;
    if (!prep) return;
    try {
      for (const [h, d] of [[0.501, 1.01], [0.5, 1.0]]) {
        const vcLen = p.piano_prep_voicing(prep, h, d, 1.0, 1.0, 440.0) >>> 0;
        const src = p.piano_prep_out_ptr(prep) >>> 0;
        const mem = this.syncMemoryView(128);
        const dst = exports.piano_staging_offset_ptr(ctx, 0, vcLen) >>> 0;
        if (dst) mem.set(mem.subarray(src >>> 2, (src >>> 2) + vcLen), dst >>> 2);
        exports.piano_apply_voicing_blob(ctx, h, d, 1.0, 1.0, 440.0, vcLen);
      }
      const spLen = p.piano_prep_space(prep, 0, 2) >>> 0;
      const spSrc = p.piano_prep_out_ptr(prep) >>> 0;
      const m2 = this.syncMemoryView(128);
      for (let off = 0; off < spLen; off += BLOB_STREAM_CHUNK) {
        const n = Math.min(BLOB_STREAM_CHUNK, spLen - off);
        const dst = exports.piano_staging_ptr(ctx, n) >>> 0;
        if (dst) m2.set(m2.subarray((spSrc >>> 2) + off, (spSrc >>> 2) + off + n), dst >>> 2);
        exports.piano_stage_space_chunk(ctx, 0, 2, off, n, spLen);
      }
      exports.piano_commit_space(ctx, 0, 2);
      exports.piano_is_space_crossfading(ctx);
      for (let i = 0; i < 4; i++) exports.piano_render(ctx, 128);
    } finally { p.piano_prep_destroy?.(prep); }
  }
}

registerProcessor(PROCESSOR_NAME, PianoWasmProcessor);
