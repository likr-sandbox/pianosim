/** Phase 2 Live Chromium AudioWorklet RT auditioning & Offline MIDI Worker stress measurements. */

function summarizeSamples(samples) {
  if (samples.length === 0) return { p50: 0, p95: 0, p99: 0, max: 0 };
  const sorted = Float64Array.from(samples).sort();
  const n = sorted.length;
  const at = (q) => sorted[Math.min(n - 1, Math.max(0, Math.ceil(q * n) - 1))];
  return { p50: at(0.5), p95: at(0.95), p99: at(0.99), max: sorted[n - 1] };
}

async function setSlider(page, testId, val) {
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
}

async function strikeKey(page, midi, release = true) {
  await page.evaluate(
    ({ midi, release }) => {
      const k = document.querySelector(`[data-midi="${midi}"]`);
      if (k) {
        k.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, buttons: 1 }));
        if (release) k.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, buttons: 0 }));
      }
    },
    { midi, release },
  );
}

async function waitAudioBlocks(page, deltaBlocks) {
  await page.evaluate((delta) => {
    return new Promise((resolve, reject) => {
      const start = window.__PIANO_DIAG__?.telemetry.blockCount ?? 0;
      const target = start + delta;
      const deadline = performance.now() + 10_000;
      const tick = () => {
        const cur = window.__PIANO_DIAG__?.telemetry.blockCount ?? 0;
        if (cur >= target) {
          resolve(cur);
          return;
        }
        if (performance.now() > deadline) {
          reject(new Error(`Timed out waiting for ${delta} audio blocks`));
          return;
        }
        setTimeout(tick, 10);
      };
      tick();
    });
  }, deltaBlocks);
}

async function waitForSilence(page) {
  await page.waitForFunction(
    () => {
      const d = window.__PIANO_DIAG__;
      if (!d || !d.analyser) return false;
      const buf = new Float32Array(d.analyser.fftSize);
      d.analyser.getFloatTimeDomainData(buf);
      let sq = 0;
      for (let i = 0; i < buf.length; i++) sq += buf[i] * buf[i];
      return Math.sqrt(sq / buf.length) < 0.001 && d.telemetry.activeVoices === 0;
    },
    { timeout: 5_000 },
  );
}

async function waitForPrepIdle(page, settleBlocks = 12) {
  await page.waitForFunction(
    () => {
      const d = window.__PIANO_DIAG__;
      return (
        d &&
        d.appliedSpaceSeq >= d.requestedSpaceSeq &&
        d.appliedVoicingSeq >= d.requestedVoicingSeq &&
        d.appliedPerfSeq >= d.requestedPerfSeq
      );
    },
    { timeout: 15_000 },
  );
  if (settleBlocks > 0) {
    await waitAudioBlocks(page, settleBlocks);
  }
}

async function selectOption(page, testId, value) {
  await page.evaluate(
    ({ testId, value }) => {
      const el = document.querySelector(`[data-testid="${testId}"]`);
      if (el) {
        el.value = String(value);
        el.dispatchEvent(new Event('change', { bubbles: true }));
      }
    },
    { testId, value },
  );
}

async function collectStrides(page, minStrides, actionFn = null) {
  const hasAction = Boolean(actionFn);
  const startState = await page.evaluate((waitAction) => {
    window.__PIANO_BENCH_DONE__ = !waitAction;
    const d = window.__PIANO_DIAG__;
    return {
      blocks: d.telemetryBlocksReceived,
      strideOverruns: d.telemetry.strideOverruns,
      singleOverruns: d.telemetry.singleBlockOverruns,
      stalls: d.telemetry.stallCount,
    };
  }, hasAction);

  const collectorPromise = page.evaluate(
    ({ startBlocks, minStrides }) =>
      new Promise((resolve) => {
        const strides = [];
        const maxPerBlock = [];
        let lastSeen = startBlocks;
        let peakRms = 0;
        let maxPeak = 0;
        let nonFinite = 0;
        let clipped = false;
        let lastDiag = window.__PIANO_DIAG__;

        const tick = () => {
          const d = window.__PIANO_DIAG__;
          if (d) {
            lastDiag = d;
            if (d.telemetryBlocksReceived > lastSeen) {
              lastSeen = d.telemetryBlocksReceived;
              strides.push(d.telemetry.callbackUs);
              maxPerBlock.push(d.telemetry.maxCallbackUs);
              if (d.telemetry.clipped) clipped = true;
            }
            if (d.analyser) {
              const buf = new Float32Array(d.analyser.fftSize);
              d.analyser.getFloatTimeDomainData(buf);
              let sq = 0;
              for (let i = 0; i < buf.length; i++) {
                const x = buf[i];
                if (!Number.isFinite(x)) nonFinite++;
                else {
                  const ax = Math.abs(x);
                  if (ax > maxPeak) maxPeak = ax;
                  sq += x * x;
                }
              }
              const rms = Math.sqrt(sq / buf.length);
              if (rms > peakRms) peakRms = rms;
            }
          }
          if (strides.length >= minStrides && window.__PIANO_BENCH_DONE__) {
            resolve({
              strides,
              maxPerBlock,
              rms: peakRms,
              peak: maxPeak,
              nonFinite,
              clipped: clipped || maxPeak >= 0.999,
              endStrideOverruns: lastDiag.telemetry.strideOverruns,
              endSingleOverruns: lastDiag.telemetry.singleBlockOverruns,
              endStalls: lastDiag.telemetry.stallCount,
              activeVoices: lastDiag.telemetry.activeVoices,
              timingSource: lastDiag.telemetry.timingSource,
            });
            return;
          }
          setTimeout(tick, 20);
        };
        tick();
      }),
    { startBlocks: startState.blocks, minStrides },
  );

  if (actionFn) {
    try {
      await actionFn();
    } finally {
      await page.evaluate(() => {
        window.__PIANO_BENCH_DONE__ = true;
      });
    }
  }
  const collected = await collectorPromise;
  const strideStats = summarizeSamples(collected.strides);
  const blockMaxStats = summarizeSamples(collected.maxPerBlock);
  return {
    p50Us: strideStats.p50,
    p95Us: strideStats.p95,
    p99Us: strideStats.p99,
    maxUs: Math.max(strideStats.max, blockMaxStats.max),
    strideMaxUs: strideStats.max,
    strideOverruns: collected.endStrideOverruns - startState.strideOverruns,
    singleBlockOverruns: collected.endSingleOverruns - startState.singleOverruns,
    stallCount: collected.endStalls - startState.stalls,
    nonFinite: collected.nonFinite,
    clipped: collected.clipped,
    rms: collected.rms,
    activeVoices: collected.activeVoices,
    timingSource: collected.timingSource,
    strideCount: collected.strides.length,
  };
}

async function measureHalfPedalPrepPhase(page, actionFn) {
  await setSlider(page, 'sustain-slider', 0.5);
  for (const m of [48, 55, 60, 64, 67, 72, 76, 79]) {
    await strikeKey(page, m, true);
  }
  await waitAudioBlocks(page, 16);
  const res = await collectStrides(page, 36, actionFn);
  await setSlider(page, 'sustain-slider', 0.0);
  await page.getByTestId('panic-btn').click();
  await waitForSilence(page);
  return res;
}

export async function runPrepIsolationWorkletBench(page) {
  const spaceOnly = await measureHalfPedalPrepPhase(page, async () => {
    for (let cycle = 0; cycle < 2; cycle++) {
      await page.getByTestId('lid-closed').click();
      await page.getByTestId('perspective-hall').click();
      await page.getByTestId('body-upright-studio').click();
      await waitForPrepIdle(page);

      await page.getByTestId('lid-half').click();
      await page.getByTestId('perspective-binaural').click();
      await page.getByTestId('body-vintage-saloon').click();
      await waitForPrepIdle(page);

      await page.getByTestId('lid-open').click();
      await page.getByTestId('perspective-player').click();
      await page.getByTestId('body-concert-grand').click();
      await waitForPrepIdle(page);
    }
  });

  const voicingOnly = await measureHalfPedalPrepPhase(page, async () => {
    for (let cycle = 0; cycle < 2; cycle++) {
      await page.getByTestId('voicing-a4-chip-415').click();
      await selectOption(page, 'voicing-temperament', '1');
      await setSlider(page, 'voicing-unison', 2.2);
      await setSlider(page, 'voicing-stretch', 1.4);
      await waitForPrepIdle(page);

      await page.getByTestId('voicing-a4-chip-442').click();
      await selectOption(page, 'voicing-temperament', '2');
      await setSlider(page, 'voicing-unison', 1.6);
      await setSlider(page, 'voicing-stretch', 1.2);
      await waitForPrepIdle(page);

      await page.getByTestId('voicing-a4-chip-440').click();
      await selectOption(page, 'voicing-temperament', '0');
      await setSlider(page, 'voicing-unison', 1.0);
      await setSlider(page, 'voicing-stretch', 1.0);
      await waitForPrepIdle(page);
    }
  });

  const combined = await measureHalfPedalPrepPhase(page, async () => {
    for (let cycle = 0; cycle < 2; cycle++) {
      await setSlider(page, 'voicing-hammer', 0.85);
      await page.getByTestId('voicing-a4-chip-415').click();
      await selectOption(page, 'voicing-temperament', '1');
      await setSlider(page, 'voicing-unison', 2.2);
      await page.getByTestId('lid-closed').click();
      await page.getByTestId('perspective-hall').click();
      await page.getByTestId('body-upright-studio').click();
      await waitForPrepIdle(page);

      await setSlider(page, 'voicing-hammer', 0.3);
      await page.getByTestId('voicing-a4-chip-440').click();
      await selectOption(page, 'voicing-temperament', '0');
      await setSlider(page, 'voicing-unison', 1.0);
      await page.getByTestId('lid-open').click();
      await page.getByTestId('perspective-player').click();
      await page.getByTestId('body-concert-grand').click();
      await waitForPrepIdle(page);
    }
  });

  return { spaceOnly, voicingOnly, combined };
}

export async function runLiveWorkletBench(page, origin, benchProfile, options = {}) {
  const url = `${origin}?perf=${benchProfile.name}`;
  await page.goto(url);
  await page.getByTestId('unlock-audio').click();
  await page.waitForFunction(
    (expectedPreset) => {
      const d = window.__PIANO_DIAG__;
      return (
        d &&
        d.status === 'ready' &&
        d.contextState === 'running' &&
        d.telemetryBlocksReceived > 0 &&
        d.perfPreset === expectedPreset
      );
    },
    benchProfile.presetId,
    { timeout: 15_000 },
  );

  // Stress 1: 20-voice sustain interactive auditioning (MIDI 68..87, >= 48 strides ~4.1 s)
  await setSlider(page, 'sustain-slider', 1.0);
  for (let m = 68; m <= 87; m++) {
    await strikeKey(page, m, true);
  }
  await page.waitForFunction(
    () => (window.__PIANO_DIAG__?.telemetry.activeVoices ?? 0) >= 20,
    { timeout: 5_000 },
  );
  const sustain20 = await collectStrides(page, 48);
  await setSlider(page, 'sustain-slider', 0.0);
  await page.getByTestId('panic-btn').click();
  await waitForSilence(page);

  // Stress 2: Interactive chord + glissando keyboard auditioning in AudioWorklet
  await setSlider(page, 'sustain-slider', 0.5);
  for (const m of [48, 55, 60, 64, 67, 72]) {
    await strikeKey(page, m, true);
  }
  const auditionGliss = await collectStrides(page, 48, async () => {
    const glissKeys = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84, 79];
    for (const m of glissKeys) {
      await strikeKey(page, m, true);
      await waitAudioBlocks(page, 6);
    }
  });
  await setSlider(page, 'sustain-slider', 0.0);
  await page.getByTestId('panic-btn').click();
  await waitForSilence(page);

  const prepIsolation = options.prepIsolate !== false ? await runPrepIsolationWorkletBench(page) : null;

  // Phase 2b: Offline Worker render of 06-rachmaninoff-prelude-op3-no2.mid + AudioBufferSourceNode check
  await page.locator('#select-demo').selectOption('rachmaninoff-prelude');
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="transport-play"]')?.hasAttribute('disabled'),
    { timeout: 10_000 },
  );
  const renderT0 = Date.now();
  await page.getByTestId('transport-play').click();
  await page.waitForFunction(
    () => {
      const txt = document.querySelector('[data-testid="offline-render-status"]')?.textContent ?? '';
      if (txt === 'Error') throw new Error('Offline render reported Error');
      return txt === 'Playing' || txt === 'Ready';
    },
    { timeout: 120_000 },
  );
  const renderWallMs = Date.now() - renderT0;

  const offlineMetrics = await page.evaluate(
    () =>
      new Promise((resolve) => {
        let peakRms = 0;
        let maxPeak = 0;
        let nonFinite = 0;
        let maxVoices = 0;
        const startBlocks = window.__PIANO_DIAG__?.telemetry.blockCount ?? 0;
        const tick = () => {
          const d = window.__PIANO_DIAG__;
          if (d) {
            if (d.telemetry.activeVoices > maxVoices) maxVoices = d.telemetry.activeVoices;
            if (d.analyser) {
              const buf = new Float32Array(d.analyser.fftSize);
              d.analyser.getFloatTimeDomainData(buf);
              let sq = 0;
              for (let i = 0; i < buf.length; i++) {
                const x = buf[i];
                if (!Number.isFinite(x)) nonFinite++;
                else {
                  const ax = Math.abs(x);
                  if (ax > maxPeak) maxPeak = ax;
                  sq += x * x;
                }
              }
              const rms = Math.sqrt(sq / buf.length);
              if (rms > peakRms) peakRms = rms;
            }
            if (d.telemetry.blockCount >= startBlocks + 240) {
              resolve({
                rms: peakRms,
                peak: maxPeak,
                nonFinite,
                clipped: maxPeak >= 0.999,
                workletActiveVoices: maxVoices,
              });
              return;
            }
          }
          setTimeout(tick, 20);
        };
        tick();
      }),
  );

  await page.getByTestId('transport-stop').click();
  await page.getByTestId('panic-btn').click();

  const audioSec = 25.0 + 2.0 + 1.6; // Rachmaninoff 25s + tail (2.0 + min(3.2, 1.6))
  const offlineRach = {
    renderWallMs,
    audioSec,
    realtimeRatio: renderWallMs > 0 ? (audioSec * 1000) / renderWallMs : 0,
    ...offlineMetrics,
  };

  return { sustain20, auditionGliss, prepIsolation, offlineRach };
}
