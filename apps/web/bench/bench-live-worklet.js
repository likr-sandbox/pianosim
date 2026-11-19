/** Phase 2 Live Chromium AudioWorklet + Prep Worker stress measurements. */

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

async function waitForPrepIdle(page) {
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
    { timeout: 10_000 },
  );
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

async function collectStrides(page, minStrides, actionFn = null) {
  const startState = await page.evaluate(() => {
    const d = window.__PIANO_DIAG__;
    return {
      blocks: d.telemetryBlocksReceived,
      strideOverruns: d.telemetry.strideOverruns,
      singleOverruns: d.telemetry.singleBlockOverruns,
      stalls: d.telemetry.stallCount,
    };
  });

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
          if (strides.length >= minStrides) {
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
    await actionFn();
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

export async function runLiveWorkletBench(page, origin, benchProfile) {
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

  // Stress 1: 20-voice sustain (MIDI 68..87, >= 48 strides ~4.1 s)
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

  // Stress 2: Heavy .mid (06-rachmaninoff-prelude-op3-no2.mid via #select-demo + transport-play)
  await page.locator('#select-demo').selectOption('rachmaninoff-prelude');
  await page.waitForFunction(
    () => !document.querySelector('[data-testid="transport-play"]')?.hasAttribute('disabled'),
    { timeout: 10_000 },
  );
  await page.getByTestId('transport-play').click();
  await page.waitForFunction(
    () => (window.__PIANO_DIAG__?.telemetry.activeVoices ?? 0) > 0,
    { timeout: 5_000 },
  );
  const rachMid = await collectStrides(page, 50);
  await page.getByTestId('transport-stop').click();
  await page.getByTestId('panic-btn').click();
  await waitForSilence(page);

  // Stress 3: Half-pedal + Space/Voicing Prep sweep
  await setSlider(page, 'sustain-slider', 0.5);
  for (const m of [48, 55, 60, 64, 67, 72, 76, 79]) {
    await strikeKey(page, m, true);
  }
  const prepSweep = await collectStrides(page, 40, async () => {
    await setSlider(page, 'voicing-hammer', 0.85);
    await setSlider(page, 'voicing-unison', 2.2);
    await page.getByTestId('lid-closed').click();
    await page.getByTestId('perspective-hall').click();
    await page.getByTestId('body-upright-studio').click();
    await waitForPrepIdle(page);

    await setSlider(page, 'voicing-hammer', 0.3);
    await setSlider(page, 'voicing-stretch', 1.4);
    await page.getByTestId('lid-open').click();
    await page.getByTestId('perspective-player').click();
    await page.getByTestId('body-concert-grand').click();
    await waitForPrepIdle(page);
  });
  await setSlider(page, 'sustain-slider', 0.0);
  await page.getByTestId('panic-btn').click();

  return { sustain20, rachMid, prepSweep };
}
