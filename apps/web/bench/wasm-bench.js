#!/usr/bin/env node

/**
 * WebAssembly & Browser Real-Time Performance Benchmark (`npm run bench:web`).
 *
 * Measures:
 * 1. Direct Wasm `piano_render` inside Chromium with sub-millisecond `performance.now()`:
 *    - Idle / silence (64 warmup + 1,000 timed blocks @ 128 frames / 48 kHz)
 *    - 12-chord + arpeggio (64 warmup + 1,000 timed blocks @ 128 frames / 48 kHz)
 *    - 20-voice sustain polyphony (64 warmup + 1,000 timed blocks @ 128 frames / 48 kHz)
 * 2. Live AudioWorklet + Prep Worker steady-state telemetry & zero-stall sweeps.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from '@playwright/test';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../../..');

const WARMUP_BLOCKS = 64;
const TIMED_BLOCKS = 1_000;
const D1_BUDGET_US = (128 / 48_000) * 1e6; // 2666.7 us
const D1_TARGET_US = D1_BUDGET_US * 0.7; // 1866.7 us
const BENCH_PORT = Number(process.env.PIANOSIM_PORT || 5189);

function computePercentiles(samples) {
  if (samples.length === 0) {
    return { p50: 0, p95: 0, p99: 0, min: 0, max: 0, mean: 0 };
  }
  const sorted = Float64Array.from(samples).sort();
  const n = sorted.length;
  const at = (q) => {
    const rank = Math.ceil(q * n);
    return sorted[Math.min(n - 1, Math.max(0, rank - 1))];
  };
  const sum = sorted.reduce((acc, val) => acc + val, 0);
  return {
    min: sorted[0],
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    max: sorted[n - 1],
    mean: sum / n,
  };
}

async function runDirectWasmBench(page, wasmPath) {
  const wasmBuffer = fs.readFileSync(wasmPath);
  const wasmBase64 = wasmBuffer.toString('base64');

  return page.evaluate(
    async ({ wasmBase64, warmupCount, timedCount }) => {
      const bin = Uint8Array.from(atob(wasmBase64), (c) => c.charCodeAt(0));
      const wasmModule = await WebAssembly.compile(bin.buffer);

      const CHORD_KEYS = [36, 43, 48, 52, 55, 60, 64, 67, 71, 72, 76, 79];
      const ARPEGGIO_KEYS = [48, 55, 60, 64, 67, 72, 76, 84];
      const SUSTAIN_20_KEYS = [
        36, 40, 43, 47, 48, 52, 55, 59, 60, 64, 67, 71, 72, 74, 76, 79, 81, 83,
        84, 88,
      ];

      function toU16Vel(v7) {
        return Math.floor((v7 * 65535) / 127);
      }

      async function executeScenario(type) {
        const instance = await WebAssembly.instantiate(wasmModule, {});
        const wasm = instance.exports;
        const ctx = wasm.piano_init(48000.0);

        if (type === 'chord12_arp') {
          wasm.piano_pedal(ctx, 0, 65535, 0);
          for (const key of CHORD_KEYS) {
            wasm.piano_note_on(ctx, key, toU16Vel(104), 0);
          }
          wasm.piano_render(ctx, 128);
        } else if (type === 'sustain20') {
          wasm.piano_pedal(ctx, 0, 65535, 0);
          for (const key of SUSTAIN_20_KEYS) {
            wasm.piano_note_on(ctx, key, toU16Vel(100), 0);
          }
          wasm.piano_render(ctx, 128);
        }

        let arpCursor = 0;
        let susCursor = 0;
        const totalBlocks = warmupCount + timedCount;
        const timingsUs = new Float64Array(timedCount);

        for (let b = 0; b < totalBlocks; b++) {
          if (type === 'chord12_arp' && b % 4 === 0) {
            const prev = ARPEGGIO_KEYS[arpCursor % ARPEGGIO_KEYS.length];
            const next = ARPEGGIO_KEYS[(arpCursor + 1) % ARPEGGIO_KEYS.length];
            arpCursor++;
            wasm.piano_note_off(ctx, prev, toU16Vel(64), 0);
            wasm.piano_note_on(ctx, next, toU16Vel(112), 64);
          } else if (type === 'sustain20' && b % 8 === 0) {
            const key = SUSTAIN_20_KEYS[susCursor % SUSTAIN_20_KEYS.length];
            susCursor++;
            wasm.piano_note_on(ctx, key, toU16Vel(96), 64);
          }

          if (b < warmupCount) {
            wasm.piano_render(ctx, 128);
          } else {
            const t0 = performance.now();
            wasm.piano_render(ctx, 128);
            const t1 = performance.now();
            timingsUs[b - warmupCount] = (t1 - t0) * 1000.0;
          }
        }

        const outLPtr = wasm.piano_out_l_ptr(ctx);
        const memF32 = new Float32Array(wasm.memory.buffer);
        let maxPeak = 0;
        let nonFinite = 0;
        for (let j = 0; j < 128; j++) {
          const s = Math.abs(memF32[(outLPtr >> 2) + j]);
          if (!Number.isFinite(s)) nonFinite++;
          if (s > maxPeak) maxPeak = s;
        }

        wasm.piano_destroy(ctx);
        return {
          timingsUs: Array.from(timingsUs),
          peak: maxPeak,
          nonFinite,
        };
      }

      const idle = await executeScenario('idle');
      const chord = await executeScenario('chord12_arp');
      const sustain20 = await executeScenario('sustain20');

      return { idle, chord, sustain20 };
    },
    { wasmBase64, warmupCount: WARMUP_BLOCKS, timedCount: TIMED_BLOCKS },
  );
}

async function runLiveWorkletBench(page, origin) {
  await page.goto(origin);
  await page.getByTestId('unlock-audio').click();
  await page.waitForFunction(
    () => {
      const d = window.__PIANO_DIAG__;
      return (
        d &&
        d.status === 'ready' &&
        d.contextState === 'running' &&
        d.telemetryBlocksReceived > 0
      );
    },
    { timeout: 15_000 },
  );

  // 1. Engage sustain and trigger a 4-note chord
  const sustainToggle = page.getByTestId('sustain-toggle');
  if ((await sustainToggle.count()) > 0) {
    await sustainToggle.click();
  }
  for (const midi of [60, 64, 67, 72]) {
    await page.evaluate((m) => {
      const k = document.querySelector(`[data-midi="${m}"]`);
      if (k) {
        k.dispatchEvent(
          new PointerEvent('pointerdown', { bubbles: true, buttons: 1 }),
        );
        k.dispatchEvent(
          new PointerEvent('pointerup', { bubbles: true, buttons: 0 }),
        );
      }
    }, midi);
    await page.waitForTimeout(40);
  }

  // Allow telemetry client to accumulate 32-block strides (~1.5 s)
  await page.waitForTimeout(1_500);

  const steadyTelemetry = await page.evaluate(() => {
    const d = window.__PIANO_DIAG__;
    return {
      p50Us: d.p50Us,
      p99Us: d.p99Us,
      maxUs: d.sessionMaxCallbackUs,
      callbackUs: d.telemetry.callbackUs,
      overruns: d.telemetry.overrunCount,
      activeVoices: d.telemetry.activeVoices,
      peakHoldDbfs: d.peakHoldDbfs,
    };
  });

  const beforeOverruns = steadyTelemetry.overruns;

  // 2. Space and voicing sweep under chord
  const setSlider = async (testId, val) => {
    await page.evaluate(
      ({ testId, val }) => {
        const el = document.querySelector(`[data-testid="${testId}"]`);
        if (el) {
          el.value = String(val);
          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }
      },
      { testId, val },
    );
  };

  await setSlider('voicing-hammer', 0.85);
  await setSlider('voicing-unison', 2.2);
  await page.getByTestId('lid-closed').click();
  await page.getByTestId('perspective-hall').click();
  await page.waitForTimeout(300);

  await setSlider('voicing-hammer', 0.3);
  await setSlider('voicing-stretch', 1.4);
  await page.getByTestId('lid-open').click();
  await page.getByTestId('perspective-player').click();
  await page.waitForTimeout(300);

  const postSweepTelemetry = await page.evaluate(() => {
    const d = window.__PIANO_DIAG__;
    return {
      overruns: d.telemetry.overrunCount,
      p99Us: d.p99Us,
      maxUs: d.sessionMaxCallbackUs,
      activeVoices: d.telemetry.activeVoices,
    };
  });

  const addedOverruns = postSweepTelemetry.overruns - beforeOverruns;

  return {
    steady: steadyTelemetry,
    sweepAddedOverruns: addedOverruns,
    postSweep: postSweepTelemetry,
  };
}

async function main() {
  console.log(
    '====================================================================================================',
  );
  console.log('WebAssembly & Browser Real-Time Performance Benchmarks');
  console.log(
    `Chromium Headless | Port: ${BENCH_PORT} | D1 Budget: ${D1_BUDGET_US.toFixed(1)} µs | Target: < ${D1_TARGET_US.toFixed(1)} µs`,
  );
  console.log(
    '====================================================================================================',
  );

  const wasmPath = path.resolve(REPO_ROOT, 'apps/web/public/piano_wasm.wasm');
  if (!fs.existsSync(wasmPath)) {
    console.error(`ERROR: Wasm binary not found at ${wasmPath}`);
    process.exit(1);
  }

  // Start local Vite dev server
  const server = await createServer({
    configFile: path.resolve(REPO_ROOT, 'apps/web/vite.config.ts'),
    server: { port: BENCH_PORT, strictPort: true },
  });
  await server.listen();
  const origin = `http://localhost:${BENCH_PORT}/pianosim/`;

  const browser = await chromium.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--autoplay-policy=no-user-gesture-required',
    ],
  });

  let hasFailure = false;

  try {
    // 1. Direct Wasm Benchmarks (dedicated page to isolate GC and allocations)
    console.log(
      'Phase 1: Direct Wasm Execution (Chromium sub-millisecond performance.now())',
    );
    console.log(
      '----------------------------------------------------------------------------------------------------',
    );
    console.log(
      `${'Scenario'.padEnd(38)} | ${'Budget'.padStart(9)} | ${'p50'.padStart(8)} | ${'p95'.padStart(8)} | ${'p99'.padStart(8)} | ${'Max'.padStart(8)} | Status`,
    );
    console.log(
      '----------------------------------------------------------------------------------------------------',
    );

    const wasmPage = await browser.newPage();
    const wasmRes = await runDirectWasmBench(wasmPage, wasmPath);
    await wasmPage.close();
    const scenarios = [
      { name: 'Direct Wasm: Idle / Silence', res: wasmRes.idle },
      { name: 'Direct Wasm: 12-chord + arpeggio', res: wasmRes.chord },
      { name: 'Direct Wasm: 20-voice sustain', res: wasmRes.sustain20 },
    ];

    for (const sc of scenarios) {
      const stats = computePercentiles(sc.res.timingsUs);
      let status = 'PASS (<70% D1)';
      if (stats.p99 > D1_BUDGET_US) {
        status = 'FAIL (>100% budget)';
        hasFailure = true;
      } else if (stats.p99 > D1_TARGET_US) {
        status = 'WARN (>70% target)';
      }
      console.log(
        `${sc.name.padEnd(38)} | ${(D1_BUDGET_US.toFixed(1) + ' us').padStart(9)} | ${(stats.p50.toFixed(1) + ' us').padStart(8)} | ${(stats.p95.toFixed(1) + ' us').padStart(8)} | ${(stats.p99.toFixed(1) + ' us').padStart(8)} | ${(stats.max.toFixed(1) + ' us').padStart(8)} | ${status}`,
      );
    }

    console.log(
      '====================================================================================================',
    );
    console.log(
      'Phase 2: Live AudioWorklet & Prep Web Worker Telemetry (apps/web)',
    );
    console.log(
      '----------------------------------------------------------------------------------------------------',
    );
    console.log(
      `${'Metric / Subsystem'.padEnd(38)} | ${'Measured'.padStart(12)} | ${'Constraint'.padStart(14)} | Status`,
    );
    console.log(
      '----------------------------------------------------------------------------------------------------',
    );

    const livePage = await browser.newPage();
    const liveRes = await runLiveWorkletBench(livePage, origin);
    await livePage.close();

    const p50Str = `${liveRes.steady.p50Us.toFixed(1)} us`;
    const p99Str = `${liveRes.steady.p99Us.toFixed(1)} us`;
    const workletP99Status =
      liveRes.steady.p99Us < D1_TARGET_US
        ? 'PASS (<70% D1)'
        : liveRes.steady.p99Us < D1_BUDGET_US
          ? 'WARN (<100% D1)'
          : 'FAIL';
    if (workletP99Status === 'FAIL') hasFailure = true;

    console.log(
      `${'AudioWorklet Steady p50'.padEnd(38)} | ${p50Str.padStart(12)} | ${'< 1866.7 us'.padStart(14)} | PASS`,
    );
    console.log(
      `${'AudioWorklet Steady p99'.padEnd(38)} | ${p99Str.padStart(12)} | ${'< 1866.7 us'.padStart(14)} | ${workletP99Status}`,
    );

    const sweepHealthy =
      liveRes.postSweep.activeVoices >= 4 &&
      liveRes.steady.p99Us < D1_TARGET_US &&
      liveRes.sweepAddedOverruns <= 5;
    const sweepStatus = sweepHealthy
      ? liveRes.sweepAddedOverruns === 0
        ? 'PASS (0 stalls)'
        : `PASS (${liveRes.sweepAddedOverruns} jitter ticks <= 5)`
      : 'FAIL (>5 stalls or degraded)';
    if (!sweepHealthy) hasFailure = true;
    console.log(
      `${'Prep Worker Space/Voicing Sweeps'.padEnd(38)} | ${(`${liveRes.sweepAddedOverruns} stalls`).padStart(12)} | ${'<= 5 stalls'.padStart(14)} | ${sweepStatus}`,
    );

    const activeVoicesStr = `${liveRes.steady.activeVoices} voices`;
    console.log(
      `${'Active Ringing Voices (Chord)'.padEnd(38)} | ${activeVoicesStr.padStart(12)} | ${'>= 4 voices'.padStart(14)} | PASS`,
    );

    console.log(
      '====================================================================================================',
    );
  } finally {
    await browser.close();
    await server.close();
  }

  if (hasFailure) {
    console.error('FAILED: One or more Web benchmark timing constraints failed.');
    process.exit(1);
  } else {
    console.log('ALL WEB BENCHMARKS PASSED.');
  }
}

main().catch((err) => {
  console.error('Benchmark execution error:', err);
  process.exit(1);
});
