/** Dedicated Web Worker rendering full-fidelity MIDI scores offline via piano_wasm.wasm. */

import {
  cc7ToU16,
  encodeKnobValue,
  PEDAL_SOSTENUTO,
  PEDAL_SUSTAIN,
  PEDAL_UNA_CORDA,
  PERF_KNOB_SPECS,
  u7ToU16,
  WEB_AUDIO_SAMPLE_RATE,
  type PerfProfileConfig,
  type PianoPrepWasmExports,
  type PianoWasmExports,
} from './abi';
import type { OfflineRenderRequest, OfflineWorkerInbound, OfflineWorkerOutbound } from './offline-abi';
import { buildEngineContext, validateWasmExports } from './worklet-wasm-init';
import type { MidiPlaybackEvent } from '../midi/smf-parser';

const BLOCK_FRAMES = 128;
const YIELD_EVERY_BLOCKS = 256;
const MAX_EVENTS_PER_BLOCK = 512;
const SETTLE_BLOCKS = 16;

let cachedModule: WebAssembly.Module | null = null;
let latestJobId = 0;
let cancelledJobId = 0;
let renderQueue: Promise<void> = Promise.resolve();

const workerScope = self as unknown as {
  postMessage(message: OfflineWorkerOutbound, transfer?: Transferable[]): void;
};

function post(msg: OfflineWorkerOutbound, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) {
    workerScope.postMessage(msg, transfer);
  } else {
    workerScope.postMessage(msg);
  }
}

function applySyncVoicingBlob(
  raw: WebAssembly.Exports,
  exports: PianoWasmExports,
  ctx: number,
  sampleRate: number,
  req: OfflineRenderRequest,
): void {
  const prepExports = raw as unknown as Partial<PianoPrepWasmExports>;
  if (
    typeof prepExports.piano_prep_init !== 'function' ||
    typeof prepExports.piano_prep_voicing_ex !== 'function' ||
    typeof prepExports.piano_prep_out_ptr !== 'function' ||
    typeof prepExports.piano_prep_destroy !== 'function'
  ) {
    return;
  }
  const prep = prepExports.piano_prep_init(sampleRate) >>> 0;
  if (prep === 0) return;
  try {
    applyPrepPerfProfile(prepExports, prep, req.perfConfig);
    const v = req.voicing;
    const lenF32 =
      prepExports.piano_prep_voicing_ex(
        prep,
        v.hammerHardness,
        v.unisonDetune,
        v.stretchTuning,
        v.stringLengthScale,
        v.a4FreqHz,
        v.temperamentId >>> 0,
        v.condition,
      ) >>> 0;
    if (lenF32 === 0) return;
    const srcPtr = prepExports.piano_prep_out_ptr(prep) >>> 0;
    const dstPtr = exports.piano_staging_offset_ptr(ctx, 0, lenF32) >>> 0;
    if (srcPtr !== 0 && dstPtr !== 0) {
      const memF32 = new Float32Array(exports.memory.buffer);
      const srcBase = srcPtr >>> 2;
      memF32.set(memF32.subarray(srcBase, srcBase + lenF32), dstPtr >>> 2);
      exports.piano_apply_voicing_blob_ex(
        ctx,
        v.hammerHardness,
        v.unisonDetune,
        v.stretchTuning,
        v.stringLengthScale,
        v.a4FreqHz,
        v.temperamentId >>> 0,
        v.condition,
        lenF32,
      );
    }
  } finally {
    prepExports.piano_prep_destroy(prep);
  }
}

function applyPrepPerfProfile(
  prepExports: Partial<PianoPrepWasmExports>,
  prep: number,
  perfConfig: PerfProfileConfig,
): void {
  if (
    typeof prepExports.piano_perf_builder_new !== 'function' ||
    typeof prepExports.piano_perf_builder_set !== 'function' ||
    typeof prepExports.piano_perf_builder_destroy !== 'function' ||
    typeof prepExports.piano_prep_set_perf_builder !== 'function'
  ) {
    return;
  }
  const builder = prepExports.piano_perf_builder_new(0) >>> 0;
  if (builder === 0) return;
  try {
    for (const spec of PERF_KNOB_SPECS) {
      prepExports.piano_perf_builder_set(builder, spec.knobId, encodeKnobValue(spec.key, perfConfig));
    }
    prepExports.piano_prep_set_perf_builder(prep, builder);
  } finally {
    prepExports.piano_perf_builder_destroy(builder);
  }
}

function dispatchEvent(
  exports: PianoWasmExports,
  ctx: number,
  ev: MidiPlaybackEvent,
  offset: number,
): void {
  if (ev.type === 'noteOn') {
    exports.piano_note_on(ctx, ev.midiKey & 0x7f, u7ToU16(ev.velocity), offset);
  } else if (ev.type === 'noteOff') {
    exports.piano_note_off(ctx, ev.midiKey & 0x7f, u7ToU16(ev.velocity), offset);
  } else if (ev.type === 'cc') {
    if (ev.controller === 64) {
      exports.piano_pedal(ctx, PEDAL_SUSTAIN, cc7ToU16(ev.value), offset);
    } else if (ev.controller === 66) {
      exports.piano_pedal(ctx, PEDAL_SOSTENUTO, ev.value >= 64 ? 65535 : 0, offset);
    } else if (ev.controller === 67) {
      exports.piano_pedal(ctx, PEDAL_UNA_CORDA, cc7ToU16(ev.value), offset);
    }
  }
}

function renderChunkWithQueueGuard(
  exports: PianoWasmExports,
  ctx: number,
  events: ReadonlyArray<MidiPlaybackEvent>,
  eventSamples: Int32Array,
  evStart: number,
  evEnd: number,
  blockStart: number,
  frames: number,
  left: Float32Array,
  right: Float32Array,
): void {
  let subStart = blockStart;
  const blockEnd = blockStart + frames;
  let cursor = evStart;

  while (subStart < blockEnd) {
    const remaining = evEnd - cursor;
    let chunkFrames = blockEnd - subStart;
    let batchEnd = evEnd;

    if (remaining > MAX_EVENTS_PER_BLOCK) {
      const splitIdx = cursor + MAX_EVENTS_PER_BLOCK;
      const splitSample = eventSamples[splitIdx];
      if (splitSample > subStart) {
        chunkFrames = Math.min(chunkFrames, splitSample - subStart);
        batchEnd = splitIdx;
      } else if (chunkFrames > 1) {
        chunkFrames = 1;
        batchEnd = splitIdx;
      } else {
        // >512 events on the final single sample: dispatch pedal events first, then notes up to 512.
        let sent = 0;
        for (let i = cursor; i < evEnd && sent < MAX_EVENTS_PER_BLOCK; i++) {
          if (events[i].type === 'cc') {
            dispatchEvent(exports, ctx, events[i], 0);
            sent++;
          }
        }
        for (let i = cursor; i < evEnd && sent < MAX_EVENTS_PER_BLOCK; i++) {
          if (events[i].type !== 'cc') {
            dispatchEvent(exports, ctx, events[i], 0);
            sent++;
          }
        }
        cursor = evEnd;
        batchEnd = evEnd;
      }
    }

    while (cursor < batchEnd && eventSamples[cursor] < subStart + chunkFrames) {
      const offset = Math.max(0, Math.min(chunkFrames - 1, eventSamples[cursor] - subStart));
      dispatchEvent(exports, ctx, events[cursor], offset);
      cursor++;
    }

    exports.piano_render(ctx, chunkFrames);
    const outLPtr = exports.piano_out_l_ptr(ctx) >>> 0;
    const outRPtr = exports.piano_out_r_ptr(ctx) >>> 0;
    const memF32 = new Float32Array(exports.memory.buffer);
    const lBase = outLPtr >>> 2;
    const rBase = outRPtr >>> 2;
    left.set(memF32.subarray(lBase, lBase + chunkFrames), subStart);
    right.set(memF32.subarray(rBase, rBase + chunkFrames), subStart);
    subStart += chunkFrames;
  }
}

async function resolveWasmModule(
  msg: Extract<OfflineWorkerInbound, { type: 'render' }>,
): Promise<WebAssembly.Module> {
  if (msg.wasmModule) {
    cachedModule = msg.wasmModule;
    return msg.wasmModule;
  }
  if (msg.wasmBytes) {
    const compiled = await WebAssembly.compile(msg.wasmBytes);
    cachedModule = compiled;
    return compiled;
  }
  if (cachedModule) return cachedModule;
  throw new Error('OfflineWorker render requires wasmModule or wasmBytes.');
}

async function executeRender(msg: Extract<OfflineWorkerInbound, { type: 'render' }>): Promise<void> {
  const req = msg.request;
  const { jobId } = req;
  if (cancelledJobId === jobId || latestJobId !== jobId) {
    post({ type: 'cancelled', jobId });
    return;
  }

  const wasmModule = await resolveWasmModule(msg);
  const instance = await WebAssembly.instantiate(wasmModule, {});
  const exports = validateWasmExports(instance.exports);
  const ctx = buildEngineContext(exports, WEB_AUDIO_SAMPLE_RATE, req.perfConfig);

  try {
    const { space, voicing, events } = req;
    const spRc = exports.piano_set_body_space(
      ctx,
      space.bodyId >>> 0,
      space.perspective >>> 0,
      space.lidState >>> 0,
    );
    if (spRc !== 0) throw new Error(`piano_set_body_space failed (${spRc})`);
    exports.piano_set_reverb(ctx, space.reverbDecayS, space.reverbMix);

    exports.piano_set_voicing_full(
      ctx,
      voicing.hammerHardness,
      voicing.unisonDetune,
      voicing.stretchTuning,
      voicing.stringLengthScale,
      voicing.a4FreqHz,
      voicing.temperamentId >>> 0,
      voicing.condition,
    );
    applySyncVoicingBlob(instance.exports, exports, ctx, WEB_AUDIO_SAMPLE_RATE, req);
    exports.piano_set_hammer_hardness(ctx, voicing.hammerHardness);
    for (let i = 0; i < SETTLE_BLOCKS; i++) {
      exports.piano_render(ctx, BLOCK_FRAMES);
    }

    const tailSeconds = 2.0 + Math.min(3.2, Math.max(0, space.reverbDecayS));
    const totalSec = Math.max(0, req.durationSeconds) + tailSeconds;
    const totalFrames = Math.max(BLOCK_FRAMES, Math.ceil(totalSec * WEB_AUDIO_SAMPLE_RATE));

    const left = new Float32Array(totalFrames);
    const right = new Float32Array(totalFrames);

    const eventSamples = new Int32Array(events.length);
    for (let i = 0; i < events.length; i++) {
      eventSamples[i] = Math.max(
        0,
        Math.min(totalFrames - 1, Math.round(events[i].timeSeconds * WEB_AUDIO_SAMPLE_RATE)),
      );
    }

    let evCursor = 0;
    let blockIndex = 0;
    post({ type: 'progress', jobId, progress: 0, renderedSec: 0, totalSec });

    for (let blockStart = 0; blockStart < totalFrames; blockStart += BLOCK_FRAMES) {
      const frames = Math.min(BLOCK_FRAMES, totalFrames - blockStart);
      const blockEnd = blockStart + frames;
      const evStart = evCursor;
      while (evCursor < events.length && eventSamples[evCursor] < blockEnd) {
        evCursor++;
      }

      renderChunkWithQueueGuard(
        exports,
        ctx,
        events,
        eventSamples,
        evStart,
        evCursor,
        blockStart,
        frames,
        left,
        right,
      );

      blockIndex++;
      if (blockIndex % YIELD_EVERY_BLOCKS === 0) {
        const renderedFrames = blockStart + frames;
        post({
          type: 'progress',
          jobId,
          progress: renderedFrames / totalFrames,
          renderedSec: renderedFrames / WEB_AUDIO_SAMPLE_RATE,
          totalSec,
        });
        await new Promise((r) => setTimeout(r, 0));
        if (cancelledJobId === jobId || latestJobId !== jobId) {
          post({ type: 'cancelled', jobId });
          return;
        }
      }
    }

    post({ type: 'progress', jobId, progress: 1, renderedSec: totalSec, totalSec });
    post(
      {
        type: 'done',
        jobId,
        sampleRate: WEB_AUDIO_SAMPLE_RATE,
        totalFrames,
        left: left.buffer,
        right: right.buffer,
      },
      [left.buffer, right.buffer],
    );
  } finally {
    exports.piano_destroy(ctx);
  }
}

self.onmessage = (event: MessageEvent<OfflineWorkerInbound>) => {
  const msg = event.data;
  if (msg.type === 'cancel') {
    cancelledJobId = msg.jobId;
    return;
  }
  if (msg.type === 'render') {
    latestJobId = msg.request.jobId;
    renderQueue = renderQueue
      .then(() => executeRender(msg))
      .catch((err) => {
        const message = err instanceof Error ? err.message : String(err);
        post({ type: 'error', jobId: msg.request.jobId, message });
      });
  }
};
