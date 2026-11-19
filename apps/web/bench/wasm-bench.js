#!/usr/bin/env node

/**
 * WebAssembly & Browser Real-Time Performance Benchmark (`npm run bench:web`).
 *
 * Phase 1: Direct Wasm `piano_render` in Chromium (`performance.now()`):
 *   idle, chord12_arp, sustain20, half_pedal_arp, smf_rachmaninoff.
 * Phase 2: Live Chromium `AudioWorklet` + `PrepWorker` (`?perf=<profile>`):
 *   20-voice sustain, 06-rachmaninoff .mid, half-pedal + Space/Voicing Prep sweep.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from '@playwright/test';
import { runDirectWasmBench } from './bench-direct-wasm.js';
import { runLiveWorkletBench } from './bench-live-worklet.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '../../..');

const D1_BUDGET_US = (128 / 48_000) * 1e6; // 2666.7 us
const D1_TARGET_US = D1_BUDGET_US * 0.7; // 1866.7 us
const BENCH_PORT = Number(process.env.PIANOSIM_PORT || 5189);
const PROFILE_PRESETS = { full: 0, balanced: 1, realtime: 2 };

function parseBenchArgs(argv) {
  let profile = 'balanced';
  let prepIsolate = true;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--prep-isolate') {
      prepIsolate = true;
      continue;
    }
    if (arg === '--no-prep-isolate') {
      prepIsolate = false;
      continue;
    }
    let val = null;
    if (arg === '--profile' || arg === '--perf-profile') {
      val = argv[++i];
    } else if (arg.startsWith('--profile=')) {
      val = arg.slice('--profile='.length);
    } else if (arg.startsWith('--perf-profile=')) {
      val = arg.slice('--perf-profile='.length);
    } else {
      throw new Error(`Unknown bench argument: ${arg}`);
    }
    if (!val || !(val in PROFILE_PRESETS)) {
      throw new Error(`Invalid profile '${val}' (expected full, balanced, or realtime)`);
    }
    profile = val;
  }
  return { name: profile, presetId: PROFILE_PRESETS[profile], prepIsolate };
}

function computePercentiles(samples) {
  if (samples.length === 0) return { p50: 0, p95: 0, p99: 0, min: 0, max: 0, mean: 0 };
  const sorted = Float64Array.from(samples).sort();
  const n = sorted.length;
  const at = (q) => sorted[Math.min(n - 1, Math.max(0, Math.ceil(q * n) - 1))];
  let sum = 0;
  for (let i = 0; i < n; i++) sum += sorted[i];
  return { min: sorted[0], p50: at(0.5), p95: at(0.95), p99: at(0.99), max: sorted[n - 1], mean: sum / n };
}

function fmtUs(v) {
  return `${v.toFixed(1)} us`.padStart(8);
}

async function main() {
  const benchProfile = parseBenchArgs(process.argv.slice(2));
  console.log('='.repeat(118));
  console.log('WebAssembly & Browser Real-Time Performance Benchmarks (F11)');
  console.log(
    `Chromium Headless | Profile: ${benchProfile.name} | Port: ${BENCH_PORT} | D1 Budget: ${D1_BUDGET_US.toFixed(1)} us | Target: <= ${D1_TARGET_US.toFixed(1)} us`,
  );
  console.log('='.repeat(118));

  const wasmPath = path.resolve(REPO_ROOT, 'apps/web/public/piano_wasm.wasm');
  if (!fs.existsSync(wasmPath)) {
    console.error(`ERROR: Wasm binary not found at ${wasmPath}`);
    process.exit(1);
  }

  const server = await createServer({
    configFile: path.resolve(REPO_ROOT, 'apps/web/vite.config.ts'),
    server: { port: BENCH_PORT, strictPort: true },
  });
  await server.listen();
  const origin = `http://localhost:${BENCH_PORT}/pianosim/`;

  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--autoplay-policy=no-user-gesture-required'],
  });

  let hasFailure = false;

  try {
    console.log('Phase 1: Direct Wasm Execution (Chromium sub-millisecond performance.now())');
    console.log('-'.repeat(118));
    console.log(
      `${'Scenario'.padEnd(36)} | ${'p50'.padStart(8)} | ${'p95'.padStart(8)} | ${'p99'.padStart(8)} | ${'Max'.padStart(8)} | ${'RMS'.padStart(7)} | ${'NaN'.padStart(3)} | Status`,
    );
    console.log('-'.repeat(118));

    const wasmPage = await browser.newPage();
    const wasmRes = await runDirectWasmBench(wasmPage, REPO_ROOT, wasmPath, benchProfile.presetId);
    await wasmPage.close();

    const p1Scenarios = [
      { name: 'Direct Wasm: Idle / Silence', res: wasmRes.idle, silent: true },
      { name: 'Direct Wasm: 12-chord + arpeggio', res: wasmRes.chord, silent: false },
      { name: 'Direct Wasm: 20-voice sustain', res: wasmRes.sustain20, silent: false },
      { name: 'Direct Wasm: Half-pedal arpeggio', res: wasmRes.halfPedalArp, silent: false },
      { name: 'Direct Wasm: 06-rachmaninoff .mid', res: wasmRes.smfRach, silent: false },
    ];

    for (const sc of p1Scenarios) {
      const stats = computePercentiles(sc.res.timingsUs);
      const audioOk = sc.res.nonFinite === 0 && !sc.res.clipped && (sc.silent || sc.res.rms > 0.002);
      let status = 'PASS (<=70% D1)';
      if (!audioOk || stats.p99 > D1_BUDGET_US) {
        status = !audioOk ? 'FAIL (audio)' : 'FAIL (>100% budget)';
        hasFailure = true;
      } else if (stats.p99 > D1_TARGET_US) {
        status = 'FAIL (>70% target)';
        hasFailure = true;
      }
      console.log(
        `${sc.name.padEnd(36)} | ${fmtUs(stats.p50)} | ${fmtUs(stats.p95)} | ${fmtUs(stats.p99)} | ${fmtUs(stats.max)} | ${sc.res.rms.toFixed(4).padStart(7)} | ${String(sc.res.nonFinite).padStart(3)} | ${status}`,
      );
    }

    if (wasmRes.prepIsolation) {
      const pi = wasmRes.prepIsolation;
      console.log('-'.repeat(118));
      console.log(
        `Phase 1b: Direct Wasm Prep Commit Breakdown (SpaceBlob=${pi.spaceBlobF32} f32, VoicingBlob=${pi.voicingBlobF32} f32)`,
      );
      const p1PrepRows = [
        ['Direct: Half-pedal render_block', pi.steadyRender],
        ['Direct: Space chunk0 (incl. reset)', pi.chunk0],
        ['Direct: Space chunk mid', pi.chunkMid],
        ['Direct: Space last + commit', pi.commit],
        ['Direct: Space xfade render_block', pi.xfadeRender],
        ['Direct: Voicing staging memcpy', pi.voicingMemcpy],
        ['Direct: Voicing apply_voicing_blob', pi.voicingApply],
        ['Direct: Voicing post-commit render', pi.voicingCommitRender],
      ];
      for (const [label, arr] of p1PrepRows) {
        const s = computePercentiles(arr);
        console.log(
          `${label.padEnd(36)} | ${fmtUs(s.p50)} | ${fmtUs(s.p95)} | ${fmtUs(s.p99)} | ${fmtUs(s.max)} | mean ${fmtUs(s.mean)} | INFO`,
        );
      }
    }

    console.log('='.repeat(118));
    console.log('Phase 2: Live Chromium AudioWorklet RT Auditioning & Offline MIDI Worker (F12)');
    console.log('-'.repeat(118));
    console.log(
      `${'Live RT Phase'.padEnd(32)} | ${'p50'.padStart(8)} | ${'p95'.padStart(8)} | ${'p99'.padStart(8)} | ${'Max'.padStart(8)} | ${'Strd'.padStart(4)} | ${'Sngl'.padStart(4)} | ${'Stll'.padStart(4)} | ${'RMS'.padStart(7)} | ${'Src'.padStart(4)} | Status`,
    );
    console.log('-'.repeat(118));

    const livePage = await browser.newPage();
    const liveRes = await runLiveWorkletBench(livePage, origin, benchProfile, {
      prepIsolate: benchProfile.prepIsolate,
    });
    await livePage.close();

    const p2Phases = [
      { name: 'Live: 20-voice sustain', r: liveRes.sustain20 },
      { name: 'Live: Chord & gliss audition', r: liveRes.auditionGliss },
    ];

    for (const ph of p2Phases) {
      const r = ph.r;
      const ok =
        r.p99Us <= D1_TARGET_US &&
        r.strideOverruns === 0 &&
        r.singleBlockOverruns === 0 &&
        r.stallCount === 0 &&
        r.nonFinite === 0 &&
        !r.clipped &&
        r.rms > 0.002;
      if (!ok) hasFailure = true;
      const status = ok
        ? 'PASS'
        : r.p99Us > D1_TARGET_US
          ? `FAIL (p99>${D1_TARGET_US.toFixed(0)})`
          : 'FAIL (stalls/audio)';
      console.log(
        `${ph.name.padEnd(32)} | ${fmtUs(r.p50Us)} | ${fmtUs(r.p95Us)} | ${fmtUs(r.p99Us)} | ${fmtUs(r.maxUs)} | ${String(r.strideOverruns).padStart(4)} | ${String(r.singleBlockOverruns).padStart(4)} | ${String(r.stallCount).padStart(4)} | ${r.rms.toFixed(4).padStart(7)} | ${r.timingSource.padStart(4)} | ${status}`,
      );
    }

    if (liveRes.prepIsolation) {
      console.log('-'.repeat(118));
      const isoRows = [
        { name: 'Half-pedal + SpaceBlob-only', r: liveRes.prepIsolation.spaceOnly },
        { name: 'Half-pedal + VoicingBlob-only', r: liveRes.prepIsolation.voicingOnly },
        { name: 'Half-pedal + Combined Prep', r: liveRes.prepIsolation.combined },
      ];
      for (const ph of isoRows) {
        const r = ph.r;
        const tag =
          r.singleBlockOverruns > 0 || r.maxUs > D1_BUDGET_US
            ? `SPIKE (Sngl=${r.singleBlockOverruns}, Max>${D1_BUDGET_US.toFixed(0)})`
            : 'INFO (<=D1)';
        console.log(
          `${ph.name.padEnd(32)} | ${fmtUs(r.p50Us)} | ${fmtUs(r.p95Us)} | ${fmtUs(r.p99Us)} | ${fmtUs(r.maxUs)} | ${String(r.strideOverruns).padStart(4)} | ${String(r.singleBlockOverruns).padStart(4)} | ${String(r.stallCount).padStart(4)} | ${r.rms.toFixed(4).padStart(7)} | ${r.timingSource.padStart(4)} | ${tag}`,
        );
      }
    }

    console.log('-'.repeat(118));
    const off = liveRes.offlineRach;
    const offOk =
      off.nonFinite === 0 &&
      !off.clipped &&
      off.rms > 0.002 &&
      off.workletActiveVoices === 0;
    if (!offOk) hasFailure = true;
    const offStatus = offOk ? 'PASS (Full offline + 0 RT voices)' : 'FAIL (offline render/audio)';
    console.log(
      `${'Offline Worker: 06-rachmaninoff'.padEnd(32)} | Wall: ${String(off.renderWallMs).padStart(6)} ms | Audio: ${off.audioSec.toFixed(1)}s (${off.realtimeRatio.toFixed(2)}x RT) | RMS: ${off.rms.toFixed(4)} | Peak: ${off.peak.toFixed(4)} | NaN: ${off.nonFinite} | Voices: ${off.workletActiveVoices} | ${offStatus}`,
    );
    console.log('='.repeat(118));
  } finally {
    await browser.close();
    await server.close();
  }

  if (hasFailure) {
    console.error('FAILED: One or more Web benchmark constraints failed.');
    process.exit(1);
  } else {
    console.log('ALL WEB BENCHMARKS PASSED.');
  }
}

main().catch((err) => {
  console.error('Benchmark execution error:', err);
  process.exit(1);
});
