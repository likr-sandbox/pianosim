import type {
  PianoWasmExports,
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
const TELEMETRY_STRIDE_BLOCKS = 16;
const MAX_PENDING_MESSAGES = 256;

const REQUIRED_EXPORTS: ReadonlyArray<keyof PianoWasmExports> = [
  'memory',
  'piano_init',
  'piano_destroy',
  'piano_note_on',
  'piano_note_off',
  'piano_pedal',
  'piano_all_notes_off',
  'piano_render',
  'piano_out_l_ptr',
  'piano_out_r_ptr',
  'piano_telemetry_ptr',
];

function nowMs(): number {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function validateWasmExports(raw: WebAssembly.Exports): PianoWasmExports {
  for (const name of REQUIRED_EXPORTS) {
    const val = raw[name];
    if (name === 'memory') {
      if (!(val instanceof WebAssembly.Memory)) {
        throw new Error(`Missing or invalid WebAssembly.Memory export: ${name}`);
      }
    } else if (typeof val !== 'function') {
      throw new Error(`Missing required C-ABI function export: ${name}`);
    }
  }
  return raw as PianoWasmExports;
}

function zeroOutputs(outL: Float32Array, outR: Float32Array | null): void {
  outL.fill(0);
  if (outR) {
    outR.fill(0);
  }
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
  private pending: WorkletInboundMessage[] = [];
  private blockCount = 0;
  private lastCallbackUs = 0;
  private maxCallbackUs = 0;

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
      void this.initWasm(msg.wasmBytes);
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

  private async initWasm(wasmBytes: ArrayBuffer): Promise<void> {
    if (this.initializing || this.exports !== null || this.destroyed) {
      return;
    }
    this.initializing = true;
    try {
      const { instance } = await WebAssembly.instantiate(wasmBytes, {});
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
      this.syncMemoryView();
      this.initializing = false;

      for (const queued of this.pending) {
        if (this.destroyed) {
          break;
        }
        this.dispatchCommand(queued);
      }
      this.pending.length = 0;

      if (!this.destroyed) {
        this.postOutbound({ type: 'ready', sampleRate });
      }
    } catch (err) {
      this.initializing = false;
      this.pending.length = 0;
      const message = err instanceof Error ? err.message : String(err);
      this.postOutbound({ type: 'error', message });
    }
  }

  private syncMemoryView(): Float32Array {
    const exports = this.exports!;
    const buf = exports.memory.buffer;
    if (buf !== this.memBuf || this.memF32 === null) {
      this.memBuf = buf;
      this.memF32 = new Float32Array(buf);
      this.outLPtr = exports.piano_out_l_ptr(this.ctx) >>> 0;
      this.outRPtr = exports.piano_out_r_ptr(this.ctx) >>> 0;
      this.telemetryPtr = exports.piano_telemetry_ptr(this.ctx) >>> 0;
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
      case 'destroy':
        exports.piano_destroy(ctx);
        this.ctx = 0;
        this.exports = null;
        this.memBuf = null;
        this.memF32 = null;
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

    let offset = 0;
    while (offset < totalFrames) {
      const remaining = totalFrames - offset;
      const chunk = remaining > MAX_FRAMES ? MAX_FRAMES : remaining;
      exports.piano_render(ctx, chunk);

      // Re-acquire memory view if Wasm linear memory grew during render.
      const mem = this.syncMemoryView();
      const lBase = this.outLPtr >>> 2;
      const rBase = this.outRPtr >>> 2;

      if (outR) {
        for (let i = 0; i < chunk; i++) {
          outL[offset + i] = mem[lBase + i];
          outR[offset + i] = mem[rBase + i];
        }
      } else {
        for (let i = 0; i < chunk; i++) {
          outL[offset + i] = mem[lBase + i];
        }
      }
      offset += chunk;
    }

    const elapsedUs = Math.max(0, (nowMs() - t0) * 1000);
    this.lastCallbackUs = elapsedUs;
    if (elapsedUs > this.maxCallbackUs) {
      this.maxCallbackUs = elapsedUs;
    }

    this.blockCount += 1;
    if (this.blockCount % TELEMETRY_STRIDE_BLOCKS === 0) {
      const mem = this.syncMemoryView();
      const tBase = this.telemetryPtr >>> 2;
      this.postOutbound({
        type: 'telemetry',
        activeVoices: mem[tBase],
        peakL: mem[tBase + 1],
        peakR: mem[tBase + 2],
        rmsMono: mem[tBase + 3],
        callbackUs: this.lastCallbackUs,
        maxCallbackUs: this.maxCallbackUs,
        blockCount: this.blockCount,
      });
      this.maxCallbackUs = 0;
    }

    return true;
  }
}

registerProcessor(PROCESSOR_NAME, PianoWasmProcessor);
