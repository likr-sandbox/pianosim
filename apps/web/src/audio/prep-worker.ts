/** Dedicated off-thread Prep Web Worker running spatial IR/NUPOLS & tuning precomputation. */

import {
  BODY_CONCERT_GRAND,
  type PianoPrepWasmExports,
  type PrepInboundMessage,
  type PrepOutboundMessage,
  type PrepToWorkletMessage,
} from './abi';

const REQUIRED_PREP_EXPORTS: ReadonlyArray<keyof PianoPrepWasmExports> = [
  'memory',
  'piano_prep_init',
  'piano_prep_destroy',
  'piano_prep_space',
  'piano_prep_space_for_body',
  'piano_prep_voicing',
  'piano_prep_out_ptr',
];

function validatePrepExports(raw: WebAssembly.Exports): PianoPrepWasmExports {
  for (const name of REQUIRED_PREP_EXPORTS) {
    const val = raw[name];
    if (name === 'memory') {
      if (!(val instanceof WebAssembly.Memory)) {
        throw new Error(`PrepWorker missing WebAssembly.Memory export: ${name}`);
      }
    } else if (typeof val !== 'function') {
      throw new Error(`PrepWorker missing required export: ${name}`);
    }
  }
  return raw as PianoPrepWasmExports;
}

let exportsRef: PianoPrepWasmExports | null = null;
let prepPtr = 0;
let workletPort: MessagePort | null = null;
let destroyed = false;

function postStatus(msg: PrepOutboundMessage): void {
  self.postMessage(msg);
}

async function initPrepWasm(
  msg: Extract<PrepInboundMessage, { type: 'init' }>,
): Promise<void> {
  if (destroyed || exportsRef !== null) {
    return;
  }
  try {
    let instance: WebAssembly.Instance;
    if (msg.wasmModule) {
      instance = await WebAssembly.instantiate(msg.wasmModule, {});
    } else if (msg.wasmBytes) {
      const res = await WebAssembly.instantiate(msg.wasmBytes, {});
      instance = res.instance;
    } else {
      throw new Error('PrepWorker init requires compiled wasmModule or wasmBytes.');
    }
    const exports = validatePrepExports(instance.exports);
    const ptr = exports.piano_prep_init(msg.sampleRate) >>> 0;
    if (ptr === 0) {
      throw new Error('piano_prep_init returned null pointer.');
    }
    exportsRef = exports;
    prepPtr = ptr;
    workletPort = msg.workletPort;
    postStatus({ type: 'ready' });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    postStatus({ type: 'error', message });
  }
}

function copyPreparedSlice(lenF32: number): ArrayBuffer | null {
  if (!exportsRef || prepPtr === 0 || lenF32 <= 0) {
    return null;
  }
  const ptr = exportsRef.piano_prep_out_ptr(prepPtr) >>> 0;
  if (ptr === 0) {
    return null;
  }
  const f32View = new Float32Array(exportsRef.memory.buffer, ptr, lenF32);
  const owned = new Float32Array(lenF32);
  owned.set(f32View);
  return owned.buffer;
}

self.onmessage = (event: MessageEvent<PrepInboundMessage>) => {
  const msg = event.data;
  if (destroyed) {
    return;
  }
  switch (msg.type) {
    case 'init':
      void initPrepWasm(msg);
      break;
    case 'prepSpace': {
      if (!exportsRef || prepPtr === 0 || !workletPort) {
        postStatus({ type: 'spaceDone', seq: msg.seq });
        return;
      }
      const bodyId = msg.bodyId ?? BODY_CONCERT_GRAND;
      const len =
        exportsRef.piano_prep_space_for_body(
          prepPtr,
          bodyId >>> 0,
          msg.perspective >>> 0,
          msg.lidState >>> 0,
        ) >>> 0;
      const buf = copyPreparedSlice(len);
      if (buf) {
        const transferMsg: PrepToWorkletMessage = {
          type: 'spaceBlob',
          bodyId,
          perspective: msg.perspective,
          lidState: msg.lidState,
          seq: msg.seq,
          blob: buf,
        };
        workletPort.postMessage(transferMsg, [buf]);
      }
      postStatus({ type: 'spaceDone', seq: msg.seq });
      break;
    }
    case 'prepVoicing': {
      if (!exportsRef || prepPtr === 0 || !workletPort) {
        postStatus({ type: 'voicingDone', seq: msg.seq });
        return;
      }
      const len =
        exportsRef.piano_prep_voicing(
          prepPtr,
          msg.hammerHardness,
          msg.unisonDetune,
          msg.stretchTuning,
          msg.stringLengthScale,
          msg.a4FreqHz,
        ) >>> 0;
      const buf = copyPreparedSlice(len);
      if (buf) {
        const transferMsg: PrepToWorkletMessage = {
          type: 'voicingBlob',
          hammerHardness: msg.hammerHardness,
          unisonDetune: msg.unisonDetune,
          stretchTuning: msg.stretchTuning,
          stringLengthScale: msg.stringLengthScale,
          a4FreqHz: msg.a4FreqHz,
          seq: msg.seq,
          blob: buf,
        };
        workletPort.postMessage(transferMsg, [buf]);
      }
      postStatus({ type: 'voicingDone', seq: msg.seq });
      break;
    }
    case 'destroy':
      destroyed = true;
      if (exportsRef && prepPtr !== 0) {
        exportsRef.piano_prep_destroy(prepPtr);
      }
      exportsRef = null;
      prepPtr = 0;
      if (workletPort) {
        workletPort.close();
        workletPort = null;
      }
      self.close();
      break;
  }
};
