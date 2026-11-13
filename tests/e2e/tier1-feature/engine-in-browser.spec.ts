import { test, expect } from '@playwright/test';
import { createPianoSynthesizer } from '../helpers/engine';
import { computeFFTPeak, computeRMS, verifySampleBounds } from '../helpers/audio-analysis';

/**
 * Browser-parity: the production src/audio/PianoSynthesizer.ts module is loaded INSIDE the page
 * through the Vite pipeline (same module the app bundles) and rendered there. Output must match
 * the Node-side render of the same class sample-for-sample (deterministic engine) and show the
 * correct pitch.
 */
async function renderInPage(page: import('@playwright/test').Page, midi: number, velocity: number, n: number) {
  const out = await page.evaluate(
    async ({ midi, velocity, n }) => {
      const url: string = '/pianosim/src/audio/PianoSynthesizer.ts';
      const mod: any = await import(/* @vite-ignore */ url);
      const synth = new mod.PianoSynthesizer(44100);
      synth.noteOn({ midiNote: midi, velocity });
      const l = new Float32Array(n);
      const r = new Float32Array(n);
      synth.render(l, r, n);
      return { l: Array.from(l), r: Array.from(r) };
    },
    { midi, velocity, n }
  );
  return { l: Float32Array.from(out.l), r: Float32Array.from(out.r) };
}

test.describe('Tier 1: Real engine executed in the browser', () => {
  test('T1-ENG-01: in-page PianoSynthesizer renders A4 at 440 Hz, matching Node render', async ({ page }) => {
    await page.goto('./');
    const n = 16384;
    const { l, r } = await renderInPage(page, 69, 100, n);

    const bounds = verifySampleBounds(l);
    expect(bounds.hasNaN).toBe(false);
    expect(bounds.hasInf).toBe(false);
    expect(bounds.absMax).toBeLessThanOrEqual(1.0);
    expect(computeRMS(l)).toBeGreaterThan(0.005);

    const mono = new Float32Array(n);
    for (let i = 0; i < n; i++) mono[i] = 0.5 * (l[i] + r[i]);
    const hz = computeFFTPeak(mono, 44100, 440.0);
    expect(Math.abs(hz - 440.0)).toBeLessThanOrEqual(1.5);

    // Same class, same inputs -> same samples in Node.
    const nl = new Float32Array(n);
    const nr = new Float32Array(n);
    const node = createPianoSynthesizer(44100);
    node.noteOn({ midiNote: 69, velocity: 100 });
    node.render(nl, nr, n);
    let maxDiff = 0;
    for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(nl[i] - l[i]), Math.abs(nr[i] - r[i]));
    expect(maxDiff).toBeLessThan(1e-4);
  });

  test('T1-ENG-02: in-page engine velocity changes loudness', async ({ page }) => {
    await page.goto('./');
    const soft = await renderInPage(page, 60, 30, 4096);
    const loud = await renderInPage(page, 60, 120, 4096);
    expect(computeRMS(loud.l, 0, 2048)).toBeGreaterThan(computeRMS(soft.l, 0, 2048) * 1.5);
  });
});
