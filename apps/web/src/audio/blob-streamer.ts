import {
  BODY_CONCERT_GRAND,
  type PianoWasmExports,
  type PrepToWorkletMessage,
} from './abi';

export const BLOB_STREAM_CHUNK = 8_192;

interface PendingSpaceBlob {
  body: number;
  persp: number;
  lid: number;
  src: Float32Array;
  off: number;
}

interface PendingVoicingBlob {
  hammer: number;
  detune: number;
  stretch: number;
  scale: number;
  a4: number;
  src: Float32Array;
  off: number;
}

/**
 * Serializes off-thread Prep Worker space and voicing blobs across the single
 * `WasmContext::staging` buffer without allocation inside the audio loop.
 */
export class BlobStreamer {
  private lastSpaceSeq = -1;
  private lastVoicingSeq = -1;
  private pendingSpace: PendingSpaceBlob | null = null;
  private queuedSpace: PendingSpaceBlob | null = null;
  private pendingVoicing: PendingVoicingBlob | null = null;
  private queuedVoicing: PendingVoicingBlob | null = null;

  reset(): void {
    this.lastSpaceSeq = -1;
    this.lastVoicingSeq = -1;
    this.pendingSpace = null;
    this.queuedSpace = null;
    this.pendingVoicing = null;
    this.queuedVoicing = null;
  }

  onPrepBlob(msg: PrepToWorkletMessage): void {
    if (msg.type === 'spaceBlob') {
      if (msg.seq < this.lastSpaceSeq) return;
      this.lastSpaceSeq = msg.seq;
      const entry: PendingSpaceBlob = {
        body: (msg.bodyId ?? BODY_CONCERT_GRAND) >>> 0,
        persp: msg.perspective >>> 0,
        lid: msg.lidState >>> 0,
        src: new Float32Array(msg.blob),
        off: 0,
      };
      if (this.pendingSpace && this.pendingSpace.off > 0) {
        this.queuedSpace = entry;
      } else {
        this.pendingSpace = entry;
        this.queuedSpace = null;
      }
      return;
    }
    if (msg.type === 'voicingBlob') {
      if (msg.seq < this.lastVoicingSeq) return;
      this.lastVoicingSeq = msg.seq;
      const entry: PendingVoicingBlob = {
        hammer: msg.hammerHardness,
        detune: msg.unisonDetune,
        stretch: msg.stretchTuning,
        scale: msg.stringLengthScale,
        a4: msg.a4FreqHz,
        src: new Float32Array(msg.blob),
        off: 0,
      };
      if (this.pendingVoicing && this.pendingVoicing.off > 0) {
        this.queuedVoicing = entry;
      } else {
        this.pendingVoicing = entry;
        this.queuedVoicing = null;
      }
    }
  }

  stepStreamingBlobs(
    exports: PianoWasmExports,
    ctx: number,
    syncMemoryView: () => Float32Array,
  ): void {
    // Serialize access to WasmContext::staging: while one blob is mid-transfer (off > 0),
    // never let the other blob overwrite staging[0..n].
    if (this.pendingVoicing && this.pendingVoicing.off > 0) {
      this.stepVoicing(exports, ctx, syncMemoryView);
      return;
    }
    if (this.pendingSpace && this.pendingSpace.off > 0) {
      this.stepSpace(exports, ctx, syncMemoryView);
      return;
    }
    if (this.pendingSpace && exports.piano_is_space_crossfading(ctx) === 0) {
      this.stepSpace(exports, ctx, syncMemoryView);
      return;
    }
    if (this.pendingVoicing) {
      this.stepVoicing(exports, ctx, syncMemoryView);
    }
  }

  private finishSpace(): void {
    this.pendingSpace = this.queuedSpace;
    this.queuedSpace = null;
  }

  private finishVoicing(): void {
    this.pendingVoicing = this.queuedVoicing;
    this.queuedVoicing = null;
  }

  private stepSpace(
    exports: PianoWasmExports,
    ctx: number,
    syncMemoryView: () => Float32Array,
  ): void {
    const b = this.pendingSpace;
    if (!b) return;
    if (b.off === 0 && exports.piano_is_space_crossfading(ctx) !== 0) return;
    const n = Math.min(BLOB_STREAM_CHUNK, b.src.length - b.off);
    const ptr = exports.piano_staging_ptr(ctx, n) >>> 0;
    if (ptr === 0) {
      this.finishSpace();
      return;
    }
    const mem = syncMemoryView();
    mem.set(b.src.subarray(b.off, b.off + n), ptr >>> 2);
    if (
      exports.piano_stage_body_space_chunk(ctx, b.body, b.persp, b.lid, b.off, n, b.src.length) !==
      0
    ) {
      this.finishSpace();
      return;
    }
    b.off += n;
    if (b.off >= b.src.length) {
      exports.piano_commit_body_space(ctx, b.body, b.persp, b.lid);
      this.finishSpace();
    }
  }

  private stepVoicing(
    exports: PianoWasmExports,
    ctx: number,
    syncMemoryView: () => Float32Array,
  ): void {
    const b = this.pendingVoicing;
    if (!b) return;
    const n = Math.min(BLOB_STREAM_CHUNK, b.src.length - b.off);
    const ptr = exports.piano_staging_offset_ptr(ctx, b.off, n) >>> 0;
    if (ptr === 0) {
      this.finishVoicing();
      return;
    }
    const mem = syncMemoryView();
    mem.set(b.src.subarray(b.off, b.off + n), ptr >>> 2);
    b.off += n;
    if (b.off >= b.src.length) {
      exports.piano_apply_voicing_blob(
        ctx,
        b.hammer,
        b.detune,
        b.stretch,
        b.scale,
        b.a4,
        b.src.length,
      );
      this.finishVoicing();
    }
  }
}

export function warmPrepInstallers(
  raw: WebAssembly.Exports,
  exports: PianoWasmExports,
  ctx: number,
  sr: number,
  syncMemoryView: () => Float32Array,
): void {
  const p = raw as Record<string, ((...a: number[]) => number) | undefined>;
  if (
    !p.piano_prep_init ||
    !p.piano_prep_voicing ||
    !p.piano_prep_space_for_body ||
    !p.piano_prep_out_ptr
  ) {
    return;
  }
  const prep = p.piano_prep_init(sr) >>> 0;
  if (!prep) return;
  try {
    for (const [h, d] of [[0.501, 1.01], [0.5, 1.0]]) {
      const vcLen = p.piano_prep_voicing(prep, h, d, 1.0, 1.0, 440.0) >>> 0;
      const src = p.piano_prep_out_ptr(prep) >>> 0;
      const mem = syncMemoryView();
      const dst = exports.piano_staging_offset_ptr(ctx, 0, vcLen) >>> 0;
      if (dst) mem.set(mem.subarray(src >>> 2, (src >>> 2) + vcLen), dst >>> 2);
      exports.piano_apply_voicing_blob(ctx, h, d, 1.0, 1.0, 440.0, vcLen);
    }
    const spLen = p.piano_prep_space_for_body(prep, 0, 0, 2) >>> 0;
    const spSrc = p.piano_prep_out_ptr(prep) >>> 0;
    const m2 = syncMemoryView();
    for (let off = 0; off < spLen; off += BLOB_STREAM_CHUNK) {
      const n = Math.min(BLOB_STREAM_CHUNK, spLen - off);
      const dst = exports.piano_staging_ptr(ctx, n) >>> 0;
      if (dst) m2.set(m2.subarray((spSrc >>> 2) + off, (spSrc >>> 2) + off + n), dst >>> 2);
      exports.piano_stage_body_space_chunk(ctx, 0, 0, 2, off, n, spLen);
    }
    exports.piano_commit_body_space(ctx, 0, 0, 2);
    exports.piano_is_space_crossfading(ctx);
    for (let i = 0; i < 4; i++) exports.piano_render(ctx, 128);
  } finally {
    p.piano_prep_destroy?.(prep);
  }
}
