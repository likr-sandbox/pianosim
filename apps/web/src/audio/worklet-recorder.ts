import type { WorkletOutboundMessage } from './abi';

const REC_POOL_SIZE = 16;
const REC_CHUNK_FRAMES = 2_048;

export type WorkletPostFn = (msg: WorkletOutboundMessage, transfer?: Transferable[]) => void;

/**
 * Pre-allocated ping-pong Transferable buffer pool for AudioWorklet recording.
 * Performs ZERO heap allocations inside `captureBlock()` on the audio render thread.
 */
export class WorkletRecorderPool {
  private readonly slots: Array<Float32Array | null> = new Array(REC_POOL_SIZE);
  private activeBuf: Float32Array | null = null;
  private activeFrames = 0;
  private recording = false;

  constructor() {
    for (let i = 0; i < REC_POOL_SIZE; i++) {
      this.slots[i] = new Float32Array(REC_CHUNK_FRAMES * 2);
    }
  }

  isActive(): boolean {
    return this.recording;
  }

  setActive(active: boolean, post: WorkletPostFn): void {
    if (active === this.recording) {
      if (!active) {
        post({ type: 'recStopped' });
      }
      return;
    }
    if (active) {
      this.recording = true;
      if (!this.activeBuf) {
        this.activeBuf = this.acquireSlot();
      }
      this.activeFrames = 0;
      return;
    }
    this.flushActive(post);
    this.recording = false;
    post({ type: 'recStopped' });
  }

  recycleBuffer(samples: ArrayBuffer): void {
    if (samples.byteLength !== REC_CHUNK_FRAMES * 2 * 4) {
      return;
    }
    const view = new Float32Array(samples);
    for (let i = 0; i < REC_POOL_SIZE; i++) {
      if (this.slots[i] === null) {
        this.slots[i] = view;
        return;
      }
    }
  }

  captureBlock(
    outL: Float32Array,
    outR: Float32Array | null,
    frames: number,
    post: WorkletPostFn,
  ): void {
    if (!this.recording) {
      return;
    }
    let srcOff = 0;
    while (srcOff < frames) {
      if (!this.activeBuf) {
        this.activeBuf = this.acquireSlot();
        this.activeFrames = 0;
        if (!this.activeBuf) {
          return;
        }
      }
      const space = REC_CHUNK_FRAMES - this.activeFrames;
      const take = frames - srcOff < space ? frames - srcOff : space;
      let dstIdx = this.activeFrames * 2;
      if (outR) {
        for (let i = 0; i < take; i++) {
          const s = srcOff + i;
          this.activeBuf[dstIdx++] = outL[s];
          this.activeBuf[dstIdx++] = outR[s];
        }
      } else {
        for (let i = 0; i < take; i++) {
          const v = outL[srcOff + i];
          this.activeBuf[dstIdx++] = v;
          this.activeBuf[dstIdx++] = v;
        }
      }
      this.activeFrames += take;
      srcOff += take;
      if (this.activeFrames >= REC_CHUNK_FRAMES) {
        this.flushActive(post);
      }
    }
  }

  reset(): void {
    this.recording = false;
    this.activeFrames = 0;
    this.activeBuf = null;
  }

  private acquireSlot(): Float32Array | null {
    for (let i = 0; i < REC_POOL_SIZE; i++) {
      const slot = this.slots[i];
      if (slot !== null) {
        this.slots[i] = null;
        return slot;
      }
    }
    return null;
  }

  private flushActive(post: WorkletPostFn): void {
    if (!this.activeBuf || this.activeFrames <= 0) {
      this.activeFrames = 0;
      return;
    }
    const buf = this.activeBuf;
    const frames = this.activeFrames;
    this.activeBuf = null;
    this.activeFrames = 0;
    post({ type: 'recChunk', samples: buf.buffer as ArrayBuffer, frames }, [buf.buffer]);
  }
}
