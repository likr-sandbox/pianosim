import {
  BODY_CONCERT_GRAND,
  encodeKnobValue,
  PERF_KNOB_SPECS,
  type PerfProfileConfig,
  type PianoWasmExports,
  type PrepToWorkletMessage,
} from './abi';

export const BLOB_STREAM_CHUNK = 2_048;

type StreamerWasmExports = PianoWasmExports & {
  piano_space_chunk_ptr?: (ctx: number) => number;
  piano_is_voicing_applying?: (ctx: number) => number;
};

interface PendingSpaceBlob {
  active: boolean;
  seq: number;
  body: number;
  persp: number;
  lid: number;
  src: Float32Array | null;
  off: number;
}

interface PendingVoicingBlob {
  active: boolean;
  seq: number;
  hammer: number;
  detune: number;
  stretch: number;
  scale: number;
  a4: number;
  temperament: number;
  condition: number;
  src: Float32Array | null;
  off: number;
}

function createSpaceSlot(): PendingSpaceBlob {
  return { active: false, seq: 0, body: BODY_CONCERT_GRAND, persp: 0, lid: 0, src: null, off: 0 };
}

function clearSpaceSlot(slot: PendingSpaceBlob): void {
  slot.active = false;
  slot.seq = 0;
  slot.body = BODY_CONCERT_GRAND;
  slot.persp = 0;
  slot.lid = 0;
  slot.src = null;
  slot.off = 0;
}

function createVoicingSlot(): PendingVoicingBlob {
  return {
    active: false,
    seq: 0,
    hammer: 0.5,
    detune: 1.0,
    stretch: 1.0,
    scale: 1.0,
    a4: 440.0,
    temperament: 0,
    condition: 0.0,
    src: null,
    off: 0,
  };
}

function clearVoicingSlot(slot: PendingVoicingBlob): void {
  slot.active = false;
  slot.seq = 0;
  slot.hammer = 0.5;
  slot.detune = 1.0;
  slot.stretch = 1.0;
  slot.scale = 1.0;
  slot.a4 = 440.0;
  slot.temperament = 0;
  slot.condition = 0.0;
  slot.src = null;
  slot.off = 0;
}

/**
 * Serializes off-thread Prep Worker space and voicing blobs across the Wasm
 * staging buffers without per-chunk or per-message descriptor allocation.
 */
export class BlobStreamer {
  private minSpaceSeq = 0;
  private minVoicingSeq = 0;
  private lastSpaceSeq = -1;
  private lastVoicingSeq = -1;
  private committedSpaceSeq = 0;
  private committedVoicingSeq = 0;
  private activeSpace: PendingSpaceBlob = createSpaceSlot();
  private queuedSpace: PendingSpaceBlob = createSpaceSlot();
  private activeVoicing: PendingVoicingBlob = createVoicingSlot();
  private queuedVoicing: PendingVoicingBlob = createVoicingSlot();

  reset(minSpaceSeq = 0, minVoicingSeq = 0): void {
    this.minSpaceSeq = minSpaceSeq;
    this.minVoicingSeq = minVoicingSeq;
    this.lastSpaceSeq = minSpaceSeq;
    this.lastVoicingSeq = minVoicingSeq;
    if (minSpaceSeq === 0 && minVoicingSeq === 0) {
      this.committedSpaceSeq = 0;
      this.committedVoicingSeq = 0;
    } else {
      this.committedSpaceSeq = Math.max(this.committedSpaceSeq, minSpaceSeq);
      this.committedVoicingSeq = Math.max(this.committedVoicingSeq, minVoicingSeq);
    }
    clearSpaceSlot(this.activeSpace);
    clearSpaceSlot(this.queuedSpace);
    clearVoicingSlot(this.activeVoicing);
    clearVoicingSlot(this.queuedVoicing);
  }

  getCommittedSpaceSeq(): number {
    return this.committedSpaceSeq;
  }

  getCommittedVoicingSeq(): number {
    return this.committedVoicingSeq;
  }

  onPrepBlob(msg: PrepToWorkletMessage): void {
    if (msg.type === 'spaceBlob') {
      if (msg.seq <= this.minSpaceSeq || msg.seq < this.lastSpaceSeq) return;
      this.lastSpaceSeq = msg.seq;
      const inFlight = this.activeSpace.active && this.activeSpace.off > 0;
      const target = inFlight ? this.queuedSpace : this.activeSpace;
      target.active = true;
      target.seq = msg.seq;
      target.body = (msg.bodyId ?? BODY_CONCERT_GRAND) >>> 0;
      target.persp = msg.perspective >>> 0;
      target.lid = msg.lidState >>> 0;
      target.src = new Float32Array(msg.blob);
      target.off = 0;
      if (!inFlight) clearSpaceSlot(this.queuedSpace);
      return;
    }
    if (msg.type === 'voicingBlob') {
      if (msg.seq <= this.minVoicingSeq || msg.seq < this.lastVoicingSeq) return;
      this.lastVoicingSeq = msg.seq;
      const inFlight = this.activeVoicing.active && this.activeVoicing.off > 0;
      const target = inFlight ? this.queuedVoicing : this.activeVoicing;
      target.active = true;
      target.seq = msg.seq;
      target.hammer = msg.hammerHardness;
      target.detune = msg.unisonDetune;
      target.stretch = msg.stretchTuning;
      target.scale = msg.stringLengthScale;
      target.a4 = msg.a4FreqHz;
      target.temperament = msg.temperamentId >>> 0;
      target.condition = msg.condition;
      target.src = new Float32Array(msg.blob);
      target.off = 0;
      if (!inFlight) clearVoicingSlot(this.queuedVoicing);
    }
  }

  stepStreamingBlobs(
    exports: PianoWasmExports,
    ctx: number,
    syncMemoryView: () => Float32Array,
  ): void {
    // Serialize access to WasmContext::staging: while one blob is mid-transfer (off > 0),
    // never let the other blob overwrite staging[0..n].
    if (this.activeVoicing.active && this.activeVoicing.off > 0) {
      this.stepVoicing(exports, ctx, syncMemoryView);
      return;
    }
    if (this.activeSpace.active && this.activeSpace.off > 0) {
      this.stepSpace(exports, ctx, syncMemoryView);
      return;
    }
    if (this.activeSpace.active && exports.piano_is_space_crossfading(ctx) === 0) {
      this.stepSpace(exports, ctx, syncMemoryView);
      return;
    }
    if (this.activeVoicing.active) {
      this.stepVoicing(exports, ctx, syncMemoryView);
    }
  }

  private finishSpace(seq: number): void {
    if (seq > this.committedSpaceSeq) {
      this.committedSpaceSeq = seq;
    }
    const finished = this.activeSpace;
    clearSpaceSlot(finished);
    if (this.queuedSpace.active) {
      this.activeSpace = this.queuedSpace;
      this.queuedSpace = finished;
    }
  }

  private finishVoicing(seq: number): void {
    if (seq > this.committedVoicingSeq) {
      this.committedVoicingSeq = seq;
    }
    const finished = this.activeVoicing;
    clearVoicingSlot(finished);
    if (this.queuedVoicing.active) {
      this.activeVoicing = this.queuedVoicing;
      this.queuedVoicing = finished;
    }
  }

  private stepSpace(
    exports: PianoWasmExports,
    ctx: number,
    syncMemoryView: () => Float32Array,
  ): void {
    const b = this.activeSpace;
    if (!b.active || !b.src) return;
    if (b.off === 0 && exports.piano_is_space_crossfading(ctx) !== 0) return;
    const sx = exports as StreamerWasmExports;
    const src = b.src;
    const off = b.off;
    const n = Math.min(BLOB_STREAM_CHUNK, src.length - off);
    const ptr = (sx.piano_space_chunk_ptr ? sx.piano_space_chunk_ptr(ctx) : sx.piano_staging_ptr(ctx, n)) >>> 0;
    if (ptr === 0) {
      this.finishSpace(b.seq);
      return;
    }
    const mem = syncMemoryView();
    const dst = ptr >>> 2;
    for (let i = 0; i < n; i++) mem[dst + i] = src[off + i];
    if (
      exports.piano_stage_body_space_chunk(ctx, b.body, b.persp, b.lid, off, n, src.length) !== 0
    ) {
      this.finishSpace(b.seq);
      return;
    }
    b.off = off + n;
    if (b.off >= src.length) {
      exports.piano_commit_body_space(ctx, b.body, b.persp, b.lid);
      this.finishSpace(b.seq);
    }
  }

  private stepVoicing(
    exports: PianoWasmExports,
    ctx: number,
    syncMemoryView: () => Float32Array,
  ): void {
    const b = this.activeVoicing;
    if (!b.active || !b.src) return;
    const sx = exports as StreamerWasmExports;
    if (b.off === 0 && (sx.piano_is_voicing_applying?.(ctx) ?? 0) !== 0) return;
    const src = b.src;
    const off = b.off;
    const n = Math.min(BLOB_STREAM_CHUNK, src.length - off);
    const ptr = exports.piano_staging_offset_ptr(ctx, off, n) >>> 0;
    if (ptr === 0) {
      this.finishVoicing(b.seq);
      return;
    }
    const mem = syncMemoryView();
    const dst = ptr >>> 2;
    for (let i = 0; i < n; i++) mem[dst + i] = src[off + i];
    b.off = off + n;
    if (b.off >= src.length) {
      exports.piano_apply_voicing_blob_ex(
        ctx,
        b.hammer,
        b.detune,
        b.stretch,
        b.scale,
        b.a4,
        b.temperament,
        b.condition,
        src.length,
      );
      this.finishVoicing(b.seq);
    }
  }
}

export function warmPrepInstallers(
  raw: WebAssembly.Exports,
  exports: PianoWasmExports,
  ctx: number,
  sr: number,
  syncMemoryView: () => Float32Array,
  perfConfig?: PerfProfileConfig,
): void {
  const p = raw as Record<string, ((...a: number[]) => number) | undefined>;
  if (
    !p.piano_prep_init ||
    !p.piano_prep_voicing_ex ||
    !p.piano_prep_space_for_body ||
    !p.piano_prep_out_ptr
  ) {
    return;
  }
  const prep = p.piano_prep_init(sr) >>> 0;
  if (!prep) return;
  try {
    if (
      perfConfig &&
      p.piano_perf_builder_new &&
      p.piano_perf_builder_set &&
      p.piano_perf_builder_destroy &&
      p.piano_prep_set_perf_builder
    ) {
      const builder = p.piano_perf_builder_new(0) >>> 0;
      if (builder) {
        try {
          for (const spec of PERF_KNOB_SPECS) {
            p.piano_perf_builder_set(builder, spec.knobId, encodeKnobValue(spec.key, perfConfig));
          }
          p.piano_prep_set_perf_builder(prep, builder);
        } finally {
          p.piano_perf_builder_destroy(builder);
        }
      }
    }
    const sx = exports as StreamerWasmExports;
    for (const [h, d] of [[0.501, 1.01], [0.5, 1.0]]) {
      const vcLen = p.piano_prep_voicing_ex(prep, h, d, 1.0, 1.0, 440.0, 0, 0.0) >>> 0;
      const src = p.piano_prep_out_ptr(prep) >>> 0;
      const mem = syncMemoryView();
      const dst = exports.piano_staging_offset_ptr(ctx, 0, vcLen) >>> 0;
      if (dst) {
        const sIdx = src >>> 2;
        const dIdx = dst >>> 2;
        for (let i = 0; i < vcLen; i++) mem[dIdx + i] = mem[sIdx + i];
      }
      exports.piano_apply_voicing_blob_ex(ctx, h, d, 1.0, 1.0, 440.0, 0, 0.0, vcLen);
    }
    const spLen = p.piano_prep_space_for_body(prep, 0, 0, 2) >>> 0;
    const spSrc = p.piano_prep_out_ptr(prep) >>> 0;
    const m2 = syncMemoryView();
    for (let off = 0; off < spLen; off += BLOB_STREAM_CHUNK) {
      const n = Math.min(BLOB_STREAM_CHUNK, spLen - off);
      const dst = (sx.piano_space_chunk_ptr ? sx.piano_space_chunk_ptr(ctx) : sx.piano_staging_ptr(ctx, n)) >>> 0;
      if (dst) {
        const sIdx = (spSrc >>> 2) + off;
        const dIdx = dst >>> 2;
        for (let i = 0; i < n; i++) m2[dIdx + i] = m2[sIdx + i];
      }
      exports.piano_stage_body_space_chunk(ctx, 0, 0, 2, off, n, spLen);
    }
    exports.piano_commit_body_space(ctx, 0, 0, 2);
    exports.piano_is_space_crossfading(ctx);
    for (let i = 0; i < 4; i++) exports.piano_render(ctx, 128);
  } finally {
    p.piano_prep_destroy?.(prep);
  }
}
