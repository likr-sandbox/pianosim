/** Phase 1 Direct Wasm execution benchmarks (Chromium sub-millisecond performance.now()). */

import fs from 'node:fs';
import path from 'node:path';

export const WARMUP_BLOCKS = 64;
export const TIMED_BLOCKS = 1_000;
export const RACH_TIMED_BLOCKS = 2_000;

/** Minimal binary SMF Format 0/1 parser returning chronological events at 48 kHz samples. */
export function parseSmfToSampleEvents(midPath, sampleRate = 48_000) {
  const bytes = fs.readFileSync(midPath);
  let off = 0;
  const u8 = () => bytes[off++];
  const u16 = () => (u8() << 8) | u8();
  const u32 = () => ((u8() << 24) | (u8() << 16) | (u8() << 8) | u8()) >>> 0;
  const vlq = () => {
    let v = 0;
    for (let i = 0; i < 4; i++) {
      const b = u8();
      v = (v << 7) | (b & 0x7f);
      if ((b & 0x80) === 0) return v >>> 0;
    }
    throw new Error('Corrupted SMF VLQ');
  };
  if (bytes.subarray(0, 4).toString('ascii') !== 'MThd') {
    throw new Error(`Invalid SMF header in ${midPath}`);
  }
  off = 4;
  const hdrLen = u32();
  u16(); // format
  const trackCount = u16();
  const division = u16() || 480;
  if (hdrLen > 6) off += hdrLen - 6;

  const raw = [];
  const tempos = [{ tick: 0, uspqn: 500_000 }];
  let order = 0;
  let parsed = 0;
  while (parsed < trackCount && off + 8 <= bytes.length) {
    const tag = bytes.subarray(off, off + 4).toString('ascii');
    off += 4;
    const len = u32();
    const end = Math.min(bytes.length, off + len);
    if (tag !== 'MTrk') {
      off = end;
      continue;
    }
    let tick = 0;
    let running = 0;
    while (off < end) {
      tick += vlq();
      if (off >= end) break;
      let st = bytes[off];
      if (st & 0x80) {
        off++;
        running = st < 0xf0 ? st : 0;
      } else {
        st = running;
      }
      if (st === 0xff) {
        const meta = u8();
        const mlen = vlq();
        if (meta === 0x51 && mlen === 3) {
          tempos.push({ tick, uspqn: ((u8() << 16) | (u8() << 8) | u8()) >>> 0 || 500_000 });
        } else {
          off += mlen;
        }
      } else if (st === 0xf0 || st === 0xf7) {
        off += vlq();
      } else {
        const cmd = st & 0xf0;
        if (cmd === 0x80 || cmd === 0x90) {
          const key = u8() & 0x7f;
          const vel = u8() & 0x7f;
          const isOn = cmd === 0x90 && vel > 0;
          raw.push({ tick, order: order++, type: isOn ? 1 : 0, a: key, b: isOn ? vel : 0 });
        } else if (cmd === 0xb0) {
          const cc = u8() & 0x7f;
          const val = u8() & 0x7f;
          if (cc === 64 || cc === 66 || cc === 67) {
            const kind = cc === 64 ? 0 : cc === 67 ? 1 : 2;
            raw.push({ tick, order: order++, type: 2, a: kind, b: val });
          }
        } else if (cmd === 0xa0 || cmd === 0xe0) {
          off += 2;
        } else if (cmd === 0xc0 || cmd === 0xd0) {
          off += 1;
        }
      }
    }
    off = end;
    parsed++;
  }

  tempos.sort((a, b) => a.tick - b.tick);
  const pts = [];
  let curSec = 0;
  let prevTick = 0;
  let curUs = 500_000;
  for (const t of tempos) {
    if (t.tick > prevTick) {
      curSec += ((t.tick - prevTick) * curUs) / (division * 1_000_000);
      prevTick = t.tick;
    }
    curUs = t.uspqn;
    pts.push({ tick: t.tick, sec: curSec, uspqn: curUs });
  }
  const tickToSec = (t) => {
    let p = pts[0];
    for (let i = 1; i < pts.length; i++) {
      if (pts[i].tick <= t) p = pts[i];
      else break;
    }
    return p.sec + ((t - p.tick) * p.uspqn) / (division * 1_000_000);
  };
  for (const r of raw) r.sec = tickToSec(r.tick);
  raw.sort((x, y) => (x.sec !== y.sec ? x.sec - y.sec : x.type !== y.type ? x.type - y.type : x.order - y.order));
  return raw.map((r) => ({
    sample: Math.max(0, Math.round(r.sec * sampleRate)),
    type: r.type, // 0=noteOff, 1=noteOn, 2=pedal
    a: r.a,
    b: r.b,
  }));
}

export async function runDirectWasmBench(page, repoRoot, wasmPath, presetId) {
  const wasmBase64 = fs.readFileSync(wasmPath).toString('base64');
  const rachPath = path.resolve(repoRoot, 'apps/web/public/demos/06-rachmaninoff-prelude-op3-no2.mid');
  const rachEvents = parseSmfToSampleEvents(rachPath, 48_000);

  return page.evaluate(
    async ({ wasmBase64, warmupCount, timedCount, rachTimedCount, presetId, rachEvents }) => {
      const bin = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));
      const wasmModule = await WebAssembly.compile(bin.buffer);

      const CHORD_KEYS = [36, 43, 48, 52, 55, 60, 64, 67, 71, 72, 76, 79];
      const ARPEGGIO_KEYS = [48, 55, 60, 64, 67, 72, 76, 84];
      const SUSTAIN_20_KEYS = [
        36, 40, 43, 47, 48, 52, 55, 59, 60, 64, 67, 71, 72, 74, 76, 79, 81, 83, 84, 88,
      ];
      const toU16 = (v7) => Math.floor((v7 * 65535) / 127);

      function initContext(wasm) {
        if (typeof wasm.piano_init_with_profile === 'function') {
          return wasm.piano_init_with_profile(48000.0, presetId);
        }
        return wasm.piano_init(48000.0);
      }

      async function executeScenario(type) {
        const instance = await WebAssembly.instantiate(wasmModule, {});
        const wasm = instance.exports;
        const ctx = initContext(wasm);
        wasm.piano_set_reverb(ctx, 1.6, 0.18);

        if (type === 'chord12_arp') {
          wasm.piano_pedal(ctx, 0, 65535, 0);
          for (const key of CHORD_KEYS) wasm.piano_note_on(ctx, key, toU16(104), 0);
          wasm.piano_render(ctx, 128);
        } else if (type === 'sustain20') {
          wasm.piano_pedal(ctx, 0, 65535, 0);
          for (const key of SUSTAIN_20_KEYS) wasm.piano_note_on(ctx, key, toU16(100), 0);
          wasm.piano_render(ctx, 128);
        } else if (type === 'half_pedal_arp') {
          wasm.piano_pedal(ctx, 0, Math.round((45 * 65535) / 127), 0);
          wasm.piano_note_on(ctx, 48, toU16(78), 0);
          wasm.piano_render(ctx, 128);
        }

        const nTimed = type === 'smf_rachmaninoff' ? rachTimedCount : timedCount;
        const totalBlocks = warmupCount + nTimed;
        const timingsUs = new Float64Array(nTimed);
        let arpCursor = 0;
        let susCursor = 0;
        let evCursor = 0;
        let nonFinite = 0;
        let maxPeak = 0;
        let sumSq = 0;
        let totalSamples = 0;

        for (let b = 0; b < totalBlocks; b++) {
          if (type === 'chord12_arp' && b % 4 === 0) {
            const prev = ARPEGGIO_KEYS[arpCursor % ARPEGGIO_KEYS.length];
            const next = ARPEGGIO_KEYS[(arpCursor + 1) % ARPEGGIO_KEYS.length];
            arpCursor++;
            wasm.piano_note_off(ctx, prev, toU16(64), 0);
            wasm.piano_note_on(ctx, next, toU16(112), 64);
          } else if (type === 'sustain20' && b % 8 === 0) {
            const key = SUSTAIN_20_KEYS[susCursor % SUSTAIN_20_KEYS.length];
            susCursor++;
            wasm.piano_note_on(ctx, key, toU16(96), 64);
          } else if (type === 'half_pedal_arp' && b % 46 === 0) {
            const prev = ARPEGGIO_KEYS[arpCursor % ARPEGGIO_KEYS.length];
            const next = ARPEGGIO_KEYS[(arpCursor + 1) % ARPEGGIO_KEYS.length];
            const cc = 42 + ((arpCursor * 3) % 18);
            arpCursor++;
            wasm.piano_pedal(ctx, 0, Math.round((cc * 65535) / 127), 0);
            wasm.piano_note_off(ctx, prev, toU16(48), 0);
            wasm.piano_note_on(ctx, next, toU16(80), 32);
          } else if (type === 'smf_rachmaninoff') {
            const blockStart = b * 128;
            const blockEnd = blockStart + 128;
            while (evCursor < rachEvents.length && rachEvents[evCursor].sample < blockEnd) {
              const ev = rachEvents[evCursor++];
              const offset = Math.max(0, Math.min(127, ev.sample - blockStart));
              if (ev.type === 1) wasm.piano_note_on(ctx, ev.a, toU16(ev.b), offset);
              else if (ev.type === 0) wasm.piano_note_off(ctx, ev.a, toU16(64), offset);
              else {
                const val = ev.a === 2 ? (ev.b >= 64 ? 65535 : 0) : toU16(ev.b);
                wasm.piano_pedal(ctx, ev.a, val, offset);
              }
            }
          }

          if (b < warmupCount) {
            wasm.piano_render(ctx, 128);
          } else {
            const t0 = performance.now();
            wasm.piano_render(ctx, 128);
            const t1 = performance.now();
            timingsUs[b - warmupCount] = (t1 - t0) * 1000.0;

            const outLPtr = wasm.piano_out_l_ptr(ctx);
            const outRPtr = wasm.piano_out_r_ptr(ctx);
            const memF32 = new Float32Array(wasm.memory.buffer);
            const lBase = outLPtr >>> 2;
            const rBase = outRPtr >>> 2;
            for (let j = 0; j < 128; j++) {
              const l = memF32[lBase + j];
              const r = memF32[rBase + j];
              if (!Number.isFinite(l) || !Number.isFinite(r)) {
                nonFinite++;
                continue;
              }
              const mx = Math.max(Math.abs(l), Math.abs(r));
              if (mx > maxPeak) maxPeak = mx;
              const m = 0.5 * (l + r);
              sumSq += m * m;
            }
            totalSamples += 128;
          }
        }

        wasm.piano_destroy(ctx);
        return {
          timingsUs: Array.from(timingsUs),
          peak: maxPeak,
          rms: totalSamples > 0 ? Math.sqrt(sumSq / totalSamples) : 0,
          clipped: maxPeak >= 0.999,
          nonFinite,
        };
      }

      const idle = await executeScenario('idle');
      // Allow V8 TurboFan background SIMD tier-up to settle after initial instantiation.
      {
        const warmInst = await WebAssembly.instantiate(wasmModule, {});
        const ww = warmInst.exports;
        const wctx = initContext(ww);
        ww.piano_pedal(wctx, 0, Math.round((43 * 65535) / 127), 0);
        for (const key of SUSTAIN_20_KEYS) ww.piano_note_on(wctx, key, toU16(104), 0);
        for (let i = 0; i < 48; i++) ww.piano_render(wctx, 128);
        for (const key of SUSTAIN_20_KEYS) ww.piano_note_off(wctx, key, toU16(56), 0);
        for (let i = 0; i < 48; i++) ww.piano_render(wctx, 128);
        ww.piano_destroy(wctx);
        await new Promise((r) => setTimeout(r, 250));
      }
      const chord = await executeScenario('chord12_arp');
      const sustain20 = await executeScenario('sustain20');
      const halfPedalArp = await executeScenario('half_pedal_arp');
      const smfRach = await executeScenario('smf_rachmaninoff');

      return { idle, chord, sustain20, halfPedalArp, smfRach };
    },
    {
      wasmBase64,
      warmupCount: WARMUP_BLOCKS,
      timedCount: TIMED_BLOCKS,
      rachTimedCount: RACH_TIMED_BLOCKS,
      presetId,
      rachEvents,
    },
  );
}
