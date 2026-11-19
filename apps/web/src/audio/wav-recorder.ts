/** Off-audio-thread stereo recording accumulator and RIFF/WAVE encoder (24-bit TPDF & 32-bit float). */

import { WEB_AUDIO_SAMPLE_RATE, type WavFormat } from './abi';

const WAVE_FORMAT_PCM = 1;
const WAVE_FORMAT_IEEE_FLOAT = 3;
const PCM24_MAX = 8_388_607.0;
const PCM24_MIN = -8_388_608.0;
const INV_2POW32 = 1.0 / 4_294_967_296.0;

/** Deterministic Mulberry32 PRNG for reproducible TPDF dither in (-1.0, +1.0) LSBs. */
class TpdfPrng {
  private stateA = 0x5049_414e;
  private stateB = 0x4f32_3442;

  private nextUniform(which: 0 | 1): number {
    let x = which === 0 ? (this.stateA = (this.stateA + 0x6d2b_79f5) | 0) : (this.stateB = (this.stateB + 0x6d2b_79f5) | 0);
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) * INV_2POW32;
  }

  nextTpdfLsb(): number {
    return this.nextUniform(0) - this.nextUniform(1);
  }
}

function writeAscii(view: Uint8Array, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) {
    view[offset + i] = text.charCodeAt(i);
  }
}

function writeWavHeader(
  bytes: Uint8Array,
  sampleRate: number,
  frames: number,
  audioFormat: number,
  bitsPerSample: number,
): void {
  const numChannels = 2;
  const blockAlign = numChannels * (bitsPerSample >>> 3);
  const dataBytes = frames * blockAlign;
  const byteRate = sampleRate * blockAlign;
  const riffSize = 36 + dataBytes;

  writeAscii(bytes, 0, 'RIFF');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  dv.setUint32(4, riffSize, true);
  writeAscii(bytes, 8, 'WAVE');
  writeAscii(bytes, 12, 'fmt ');
  dv.setUint32(16, 16, true);
  dv.setUint16(20, audioFormat, true);
  dv.setUint16(22, numChannels, true);
  dv.setUint32(24, sampleRate, true);
  dv.setUint32(28, byteRate, true);
  dv.setUint16(32, blockAlign, true);
  dv.setUint16(34, bitsPerSample, true);
  writeAscii(bytes, 36, 'data');
  dv.setUint32(40, dataBytes, true);
}

/** Encodes interleaved stereo f32 chunks into a genuine RIFF/WAVE byte array. */
export function encodeStereoWav(
  chunks: ReadonlyArray<Float32Array>,
  totalFrames: number,
  sampleRate: number,
  format: WavFormat,
): Uint8Array {
  const sr = Number.isFinite(sampleRate) && sampleRate > 0 ? Math.round(sampleRate) : WEB_AUDIO_SAMPLE_RATE;
  const frames = Math.max(0, Math.trunc(totalFrames));

  if (format === 'f32') {
    const blockAlign = 8;
    const out = new Uint8Array(44 + frames * blockAlign);
    writeWavHeader(out, sr, frames, WAVE_FORMAT_IEEE_FLOAT, 32);
    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
    let byteOff = 44;
    let writtenFrames = 0;
    for (const chunk of chunks) {
      const chunkFrames = Math.min(frames - writtenFrames, chunk.length >>> 1);
      for (let i = 0; i < chunkFrames; i++) {
        const l = chunk[i * 2];
        const r = chunk[i * 2 + 1];
        dv.setFloat32(byteOff, Number.isFinite(l) ? l : 0.0, true);
        dv.setFloat32(byteOff + 4, Number.isFinite(r) ? r : 0.0, true);
        byteOff += 8;
      }
      writtenFrames += chunkFrames;
      if (writtenFrames >= frames) break;
    }
    return out;
  }

  // 24-bit PCM with deterministic TPDF dither
  const blockAlign = 6;
  const out = new Uint8Array(44 + frames * blockAlign);
  writeWavHeader(out, sr, frames, WAVE_FORMAT_PCM, 24);
  const rng = new TpdfPrng();
  let byteOff = 44;
  let writtenFrames = 0;

  for (const chunk of chunks) {
    const chunkFrames = Math.min(frames - writtenFrames, chunk.length >>> 1);
    for (let i = 0; i < chunkFrames; i++) {
      const rawL = chunk[i * 2];
      const rawR = chunk[i * 2 + 1];
      const clL = Number.isFinite(rawL) ? Math.max(-1.0, Math.min(1.0, rawL)) : 0.0;
      const clR = Number.isFinite(rawR) ? Math.max(-1.0, Math.min(1.0, rawR)) : 0.0;
      const qL = Math.max(PCM24_MIN, Math.min(PCM24_MAX, Math.round(clL * PCM24_MAX + rng.nextTpdfLsb()))) | 0;
      const qR = Math.max(PCM24_MIN, Math.min(PCM24_MAX, Math.round(clR * PCM24_MAX + rng.nextTpdfLsb()))) | 0;

      out[byteOff] = qL & 0xff;
      out[byteOff + 1] = (qL >> 8) & 0xff;
      out[byteOff + 2] = (qL >> 16) & 0xff;
      out[byteOff + 3] = qR & 0xff;
      out[byteOff + 4] = (qR >> 8) & 0xff;
      out[byteOff + 5] = (qR >> 16) & 0xff;
      byteOff += 6;
    }
    writtenFrames += chunkFrames;
    if (writtenFrames >= frames) break;
  }
  return out;
}

/** Encodes planar stereo f32 buffers (left/right) into a genuine RIFF/WAVE byte array. */
export function encodePlanarStereoWav(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
  format: WavFormat,
): Uint8Array {
  const sr = Number.isFinite(sampleRate) && sampleRate > 0 ? Math.round(sampleRate) : WEB_AUDIO_SAMPLE_RATE;
  const frames = Math.max(0, Math.min(left.length, right.length));

  if (format === 'f32') {
    const out = new Uint8Array(44 + frames * 8);
    writeWavHeader(out, sr, frames, WAVE_FORMAT_IEEE_FLOAT, 32);
    const dv = new DataView(out.buffer, out.byteOffset, out.byteLength);
    let byteOff = 44;
    for (let i = 0; i < frames; i++) {
      const l = left[i];
      const r = right[i];
      dv.setFloat32(byteOff, Number.isFinite(l) ? l : 0.0, true);
      dv.setFloat32(byteOff + 4, Number.isFinite(r) ? r : 0.0, true);
      byteOff += 8;
    }
    return out;
  }

  const out = new Uint8Array(44 + frames * 6);
  writeWavHeader(out, sr, frames, WAVE_FORMAT_PCM, 24);
  const rng = new TpdfPrng();
  let byteOff = 44;
  for (let i = 0; i < frames; i++) {
    const rawL = left[i];
    const rawR = right[i];
    const clL = Number.isFinite(rawL) ? Math.max(-1.0, Math.min(1.0, rawL)) : 0.0;
    const clR = Number.isFinite(rawR) ? Math.max(-1.0, Math.min(1.0, rawR)) : 0.0;
    const qL = Math.max(PCM24_MIN, Math.min(PCM24_MAX, Math.round(clL * PCM24_MAX + rng.nextTpdfLsb()))) | 0;
    const qR = Math.max(PCM24_MIN, Math.min(PCM24_MAX, Math.round(clR * PCM24_MAX + rng.nextTpdfLsb()))) | 0;

    out[byteOff] = qL & 0xff;
    out[byteOff + 1] = (qL >> 8) & 0xff;
    out[byteOff + 2] = (qL >> 16) & 0xff;
    out[byteOff + 3] = qR & 0xff;
    out[byteOff + 4] = (qR >> 8) & 0xff;
    out[byteOff + 5] = (qR >> 16) & 0xff;
    byteOff += 6;
  }
  return out;
}

export function createPlanarWavBlob(
  left: Float32Array,
  right: Float32Array,
  sampleRate: number,
  format: WavFormat,
): Blob | null {
  if (Math.min(left.length, right.length) <= 0) {
    return null;
  }
  const bytes = encodePlanarStereoWav(left, right, sampleRate, format);
  return new Blob([bytes.buffer as ArrayBuffer], { type: 'audio/wav' });
}

/** Off-audio-thread stereo buffer accumulator for live AudioWorklet recording. */
export class WavRecorder {
  private chunks: Float32Array[] = [];
  private totalFrames = 0;
  private sampleRate = WEB_AUDIO_SAMPLE_RATE;
  private recording = false;

  start(sampleRate = WEB_AUDIO_SAMPLE_RATE): void {
    this.chunks = [];
    this.totalFrames = 0;
    this.sampleRate = Number.isFinite(sampleRate) && sampleRate > 0 ? sampleRate : WEB_AUDIO_SAMPLE_RATE;
    this.recording = true;
  }

  pushInterleavedChunk(view: Float32Array): void {
    const frames = view.length >>> 1;
    if (frames <= 0) return;
    const copy = new Float32Array(frames * 2);
    copy.set(view.subarray(0, frames * 2));
    this.chunks.push(copy);
    this.totalFrames += frames;
  }

  stop(): void {
    this.recording = false;
  }

  clear(): void {
    this.chunks = [];
    this.totalFrames = 0;
    this.recording = false;
  }

  isRecording(): boolean {
    return this.recording;
  }

  getFrameCount(): number {
    return this.totalFrames;
  }

  getSampleRate(): number {
    return this.sampleRate;
  }

  getDurationSeconds(): number {
    return this.sampleRate > 0 ? this.totalFrames / this.sampleRate : 0;
  }

  encode(format: WavFormat): Uint8Array {
    return encodeStereoWav(this.chunks, this.totalFrames, this.sampleRate, format);
  }

  createBlob(format: WavFormat): Blob | null {
    if (this.totalFrames <= 0) {
      return null;
    }
    const bytes = this.encode(format);
    return new Blob([bytes.buffer as ArrayBuffer], { type: 'audio/wav' });
  }
}

export function triggerWavDownload(
  blob: Blob,
  format: WavFormat,
  filenamePrefix = 'pianosim-recording',
): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const suffix = format === 'f32' ? '32f' : '24bit';
  a.href = url;
  a.download = `${filenamePrefix}-${suffix}.wav`;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export type RecordingUpdateListener = (frames: number, durationSec: number, recording: boolean) => void;

/** Coordinates Worklet recording lifecycle, drain settlement, and WAV export for WasmAudioHost. */
export class HostRecordingBridge {
  private readonly recorder = new WavRecorder();
  private stopResolver: (() => void) | null = null;
  private listener: RecordingUpdateListener | null = null;

  setListener(cb: RecordingUpdateListener | null): void {
    this.listener = cb;
  }

  start(sampleRate: number, notifyWorklet: () => void): void {
    this.recorder.start(sampleRate);
    notifyWorklet();
    this.notify();
  }

  stop(isReady: boolean, notifyWorkletStop: () => void): Promise<void> {
    if (!this.recorder.isRecording()) return Promise.resolve();
    this.recorder.stop();
    if (!isReady) {
      this.notify();
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        this.stopResolver = null;
        this.notify();
        resolve();
      };
      this.stopResolver = finish;
      notifyWorkletStop();
      window.setTimeout(finish, 250);
    });
  }

  onChunk(samples: ArrayBuffer, frames: number): void {
    this.recorder.pushInterleavedChunk(new Float32Array(samples, 0, frames * 2));
    this.notify();
  }

  onStopped(): void {
    this.stopResolver?.();
  }

  dispose(): void {
    this.stopResolver?.();
    this.stopResolver = null;
    this.recorder.clear();
  }

  isRecording(): boolean {
    return this.recorder.isRecording();
  }

  getFrameCount(): number {
    return this.recorder.getFrameCount();
  }

  getDurationSeconds(): number {
    return this.recorder.getDurationSeconds();
  }

  exportBlob(format: WavFormat): Blob | null {
    return this.recorder.createBlob(format);
  }

  private notify(): void {
    this.listener?.(
      this.recorder.getFrameCount(),
      this.recorder.getDurationSeconds(),
      this.recorder.isRecording(),
    );
  }
}
