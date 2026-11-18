import { expect, test } from '@playwright/test';
import {
  captureStrikeAttack,
  expectAudioDecay,
  sampleDestinationAudio,
  setSlider,
  setupErrorMonitoring,
  unlockAudioEngine,
} from './helpers/audio-diag';

test.describe('E2E Pedals & Voicing Studio Controls', () => {
  let assertNoErrors: () => void;

  test.beforeEach(async ({ page }) => {
    assertNoErrors = setupErrorMonitoring(page);
    await page.goto('./');
    await unlockAudioEngine(page);
  });

  test.afterEach(async ({ page }) => {
    await page.getByTestId('panic-btn').click();
    assertNoErrors();
  });

  test('Damper pedal: continuous half-pedal clearance decay rate ordering', async ({ page }) => {
    const a4Key = page.locator('[data-midi="69"]');

    // 1. Full sustain (1.0): damper disengaged
    await setSlider(page, 'sustain-slider', 1.0);
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
    const sustainedInit = await sampleDestinationAudio(page);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(350);
    const sustainedDecayed = await sampleDestinationAudio(page);
    const sustainRatio = sustainedDecayed.rms / sustainedInit.rms;

    // Reset silence
    await page.getByTestId('panic-btn').click();
    await page.waitForTimeout(200);

    // 2. Closed damper (0.0): full resting preload damping
    await setSlider(page, 'sustain-slider', 0.0);
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
    const closedInit = await sampleDestinationAudio(page);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(350);
    const closedDecayed = await sampleDestinationAudio(page);
    const closedRatio = closedDecayed.rms / closedInit.rms;

    // Reset silence
    await page.getByTestId('panic-btn').click();
    await page.waitForTimeout(200);

    // 3. Half-pedal (0.5): ~32 um partial contact clearance
    await setSlider(page, 'sustain-slider', 0.5);
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
    const halfInit = await sampleDestinationAudio(page);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(350);
    const halfDecayed = await sampleDestinationAudio(page);
    const halfRatio = halfDecayed.rms / halfInit.rms;

    // Strictly monotonic decay rate ordering: closed decays fastest, sustain slowest
    expect(closedRatio).toBeLessThan(halfRatio);
    expect(halfRatio).toBeLessThan(sustainRatio);
    expect(halfDecayed.nonFiniteCount).toBe(0);
    expect(halfDecayed.peak).toBeLessThanOrEqual(1.0);
  });

  test('Sostenuto pedal: mechanical latching sustains prior note while subsequent notes decay', async ({
    page,
  }) => {
    const note1 = page.locator('[data-midi="60"]'); // C4 (261.6 Hz)
    const note2 = page.locator('[data-midi="67"]'); // G4 (392.0 Hz)

    // Step 0: Measure unlatched C4 baseline release decay (damper closes immediately)
    await note1.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
    const unlatchedHeld = await sampleDestinationAudio(page);
    expect(unlatchedHeld.rms).toBeGreaterThan(0.002);

    await note1.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(300);
    const unlatchedReleased = await sampleDestinationAudio(page);
    expect(unlatchedReleased.nonFiniteCount).toBe(0);
    expect(unlatchedReleased.rms).toBeLessThan(unlatchedHeld.rms * 0.05);

    await page.getByTestId('panic-btn').click();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(0.0005);

    // Step 1: Strike and hold Note 1
    await note1.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(300);
    const note1Held = await sampleDestinationAudio(page);
    expect(note1Held.rms).toBeGreaterThan(0.002);

    // Step 2: Engage Sostenuto (latches Note 1 damper)
    await page.getByTestId('pedal-sostenuto').click();
    expect(await page.getByTestId('pedal-sostenuto').getAttribute('aria-pressed')).toBe('true');

    // Step 3: Release Note 1 - must sustain far above unlatched closed-damper baseline
    await note1.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(300);
    const note1Latched = await sampleDestinationAudio(page);
    expect(note1Latched.nonFiniteCount).toBe(0);
    expect(note1Latched.rms).toBeGreaterThan(0.002);
    expect(note1Latched.rms).toBeGreaterThan(unlatchedReleased.rms * 10);
    const latchedRetention = note1Latched.rms / note1Held.rms;
    const unlatchedRetention = unlatchedReleased.rms / unlatchedHeld.rms;
    expect(latchedRetention).toBeGreaterThan(unlatchedRetention * 10);

    // Step 4: Play Note 2 staccato (struck AFTER sostenuto engaged)
    await note2.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(150);
    const note2Held = await sampleDestinationAudio(page);
    expect(note2Held.rms).toBeGreaterThan(note1Latched.rms);
    await note2.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(400); // Allow Note 2 to decay

    // Step 5: Verify Note 2 decayed while Note 1 still rings
    const sustainedState = await sampleDestinationAudio(page);
    expect(sustainedState.nonFiniteCount).toBe(0);
    expect(sustainedState.rms).toBeGreaterThan(0.001);
    expect(sustainedState.rms).toBeGreaterThan(unlatchedReleased.rms * 5);
    expect(sustainedState.rms).toBeLessThan(note2Held.rms * 0.65);
    expect(sustainedState.peakFreqHz).toBeGreaterThanOrEqual(255);
    expect(sustainedState.peakFreqHz).toBeLessThanOrEqual(268); // Peak is Note 1 (C4), not G4

    // Step 6: Release Sostenuto pedal - Note 1 now dampens and decays
    await page.getByTestId('pedal-sostenuto').click();
    await expect
      .poll(async () => (await sampleDestinationAudio(page)).rms, { timeout: 3000 })
      .toBeLessThan(sustainedState.rms * 0.25);
  });

  test('Una Corda pedal: strike weight attenuation and high-frequency softening', async ({
    page,
  }) => {
    // Capture onset-aligned 85 ms attack windows via lossless f32 AudioWorklet recording
    // to eliminate wall-clock phase jitter against trichord prompt decay.
    const normalC5 = await captureStrikeAttack(page, 72, 200, false);
    const unaC5 = await captureStrikeAttack(page, 72, 200, true);
    const normalA0 = await captureStrikeAttack(page, 21, 20, false);
    const unaA0 = await captureStrikeAttack(page, 21, 20, true);

    expect(unaC5.nonFiniteCount).toBe(0);
    expect(unaC5.attackPeak).toBeLessThan(normalC5.attackPeak * 0.8);
    expect(unaC5.attackRms).toBeLessThan(normalC5.attackRms * 0.85);
    expect(unaC5.spectralCentroidHz).toBeLessThan(normalC5.spectralCentroidHz * 0.92);

    expect(unaA0.nonFiniteCount).toBe(0);
    expect(unaA0.attackPeak).toBeLessThan(normalA0.attackPeak * 0.92);
    expect(unaA0.attackRms).toBeLessThan(normalA0.attackRms);
    expect(unaA0.spectralCentroidHz).toBeLessThan(normalA0.spectralCentroidHz * 0.88);

    await page.getByTestId('pedal-una-corda').click(); // Disengage
  });

  test('Voicing panel: dynamic parameter sweeping during sounding chord produces no clicks or NaN', async ({
    page,
  }) => {
    // Strike and hold C major chord
    await page.locator('[data-midi="60"]').dispatchEvent('pointerdown', { buttons: 1 });
    await page.locator('[data-midi="64"]').dispatchEvent('pointerdown', { buttons: 1 });
    await page.locator('[data-midi="67"]').dispatchEvent('pointerdown', { buttons: 1 });
    await setSlider(page, 'sustain-slider', 1.0);
    await page.waitForTimeout(300);

    const initial = await sampleDestinationAudio(page);
    expect(initial.rms).toBeGreaterThan(0.003);

    // Concurrently sweep all 4 voicing parameters across extremes
    await setSlider(page, 'voicing-hammer', 0.95);
    await setSlider(page, 'voicing-unison', 3.5);
    await setSlider(page, 'voicing-stretch', 1.8);
    await setSlider(page, 'voicing-length', 1.6);
    await page.waitForTimeout(200);

    const sweptHigh = await sampleDestinationAudio(page);
    expect(sweptHigh.nonFiniteCount).toBe(0);
    expect(sweptHigh.peak).toBeLessThanOrEqual(1.0);
    expect(sweptHigh.rms).toBeGreaterThan(0.001);

    // Sweep in opposite direction
    await setSlider(page, 'voicing-hammer', 0.05);
    await setSlider(page, 'voicing-unison', 0.0);
    await setSlider(page, 'voicing-stretch', 0.1);
    await setSlider(page, 'voicing-length', 0.6);
    await page.waitForTimeout(200);

    const sweptLow = await sampleDestinationAudio(page);
    expect(sweptLow.nonFiniteCount).toBe(0);
    expect(sweptLow.peak).toBeLessThanOrEqual(1.0);
    expect(sweptLow.rms).toBeGreaterThan(0.001);

    // Release keys and sustain
    await page.locator('[data-midi="60"]').dispatchEvent('pointerup', { buttons: 0 });
    await page.locator('[data-midi="64"]').dispatchEvent('pointerup', { buttons: 0 });
    await page.locator('[data-midi="67"]').dispatchEvent('pointerup', { buttons: 0 });
    await setSlider(page, 'sustain-slider', 0.0);
  });

  test('Space panel: lid and perspective reconfigurations execute clickless crossfades', async ({
    page,
  }) => {
    const a4Key = page.locator('[data-midi="69"]');
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await setSlider(page, 'sustain-slider', 1.0);
    await page.waitForTimeout(250);

    // Switch to Closed lid while A4 is sounding, allow PrepWorker + BlobStreamer (~300ms) + 8ms crossfade to settle
    await page.getByTestId('lid-closed').click();
    await page.waitForTimeout(450);
    const closedCrossfade = await sampleDestinationAudio(page);
    expect(closedCrossfade.nonFiniteCount).toBe(0);
    expect(closedCrossfade.peak).toBeLessThanOrEqual(1.0);

    // Measure settled Closed lid spectrum at consistent post-attack time
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(150);
    const closedLid = await sampleDestinationAudio(page);
    expect(closedLid.nonFiniteCount).toBe(0);
    expect(closedLid.peak).toBeLessThanOrEqual(1.0);

    // Switch back to Open lid while A4 is sounding and verify clickless crossfade + restored HF energy
    await page.getByTestId('lid-open').click();
    await page.waitForTimeout(450);
    const openCrossfade = await sampleDestinationAudio(page);
    expect(openCrossfade.nonFiniteCount).toBe(0);
    expect(openCrossfade.peak).toBeLessThanOrEqual(1.0);

    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(150);
    const openLid = await sampleDestinationAudio(page);
    expect(openLid.highFreqRatio).toBeGreaterThan(closedLid.highFreqRatio);

    // Switch through listening perspectives
    const perspectives = [
      'perspective-close',
      'perspective-hall',
      'perspective-binaural',
      'perspective-player',
    ];
    for (const p of perspectives) {
      await page.getByTestId(p).click();
      await page.waitForTimeout(120);
      const stats = await sampleDestinationAudio(page);
      expect(stats.nonFiniteCount).toBe(0);
      expect(stats.peak).toBeLessThanOrEqual(1.0);
      expect(stats.rms).toBeGreaterThan(0.001);
    }

    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await setSlider(page, 'sustain-slider', 0.0);
  });

  test('Tier 2 Boundary: Sostenuto engaged with no keys held latches zero dampers and subsequent notes decay', async ({
    page,
  }) => {
    // Edge case: engage Sostenuto when no keys are depressed
    const sostenutoBtn = page.getByTestId('pedal-sostenuto');
    await sostenutoBtn.click();
    expect(await sostenutoBtn.getAttribute('aria-pressed')).toBe('true');

    // Confirm no active/latched key indicators
    expect(await page.locator('.piano-key.is-active').count()).toBe(0);

    // Strike a note AFTER Sostenuto is engaged (e.g. E4 MIDI 64)
    const e4Key = page.locator('[data-midi="64"]');
    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(250);
    const heldStats = await sampleDestinationAudio(page);
    expect(heldStats.rms).toBeGreaterThan(0.002);
    expect(heldStats.nonFiniteCount).toBe(0);

    // Release key: because Sostenuto was engaged before key-down, damper must NOT be latched
    await e4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(350);
    const decayedStats = await sampleDestinationAudio(page);
    expect(decayedStats.rms).toBeLessThan(heldStats.rms * 0.35);
    expect(decayedStats.nonFiniteCount).toBe(0);
    expect(decayedStats.peak).toBeLessThanOrEqual(1.0);

    // Disengage Sostenuto
    await sostenutoBtn.click();
    expect(await sostenutoBtn.getAttribute('aria-pressed')).toBe('false');
  });

  test('Tier 2 Boundary: Voicing sliders accept exact boundary limits (0.0/1.0, 0.0/4.0, 0.0/2.0, 0.5/2.0)', async ({
    page,
  }) => {
    // 1. Min boundary limits
    await setSlider(page, 'voicing-hammer', 0.0);
    await setSlider(page, 'voicing-unison', 0.0);
    await setSlider(page, 'voicing-stretch', 0.0);
    await setSlider(page, 'voicing-length', 0.5);

    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(0.0, 2);
    expect(Number(await page.getByTestId('voicing-unison').inputValue())).toBeCloseTo(0.0, 2);
    expect(Number(await page.getByTestId('voicing-stretch').inputValue())).toBeCloseTo(0.0, 2);
    expect(Number(await page.getByTestId('voicing-length').inputValue())).toBeCloseTo(0.5, 2);

    // Play note at minimum boundaries and verify numerical stability
    const a4Key = page.locator('[data-midi="69"]');
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(250);
    const minStats = await sampleDestinationAudio(page);
    expect(minStats.nonFiniteCount).toBe(0);
    expect(minStats.peak).toBeLessThanOrEqual(1.0);
    expect(minStats.rms).toBeGreaterThan(0.001);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(250);

    // 2. Max boundary limits
    await setSlider(page, 'voicing-hammer', 1.0);
    await setSlider(page, 'voicing-unison', 4.0);
    await setSlider(page, 'voicing-stretch', 2.0);
    await setSlider(page, 'voicing-length', 2.0);

    expect(Number(await page.getByTestId('voicing-hammer').inputValue())).toBeCloseTo(1.0, 2);
    expect(Number(await page.getByTestId('voicing-unison').inputValue())).toBeCloseTo(4.0, 2);
    expect(Number(await page.getByTestId('voicing-stretch').inputValue())).toBeCloseTo(2.0, 2);
    expect(Number(await page.getByTestId('voicing-length').inputValue())).toBeCloseTo(2.0, 2);

    // Play note at maximum boundaries and verify numerical stability
    await a4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(250);
    const maxStats = await sampleDestinationAudio(page);
    expect(maxStats.nonFiniteCount).toBe(0);
    expect(maxStats.peak).toBeLessThanOrEqual(1.0);
    expect(maxStats.rms).toBeGreaterThan(0.001);
    await a4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(250);

    // Reset to nominal defaults
    await setSlider(page, 'voicing-hammer', 0.5);
    await setSlider(page, 'voicing-unison', 1.0);
    await setSlider(page, 'voicing-stretch', 1.0);
    await setSlider(page, 'voicing-length', 1.0);
  });

  test('Tier 4 Scenario 3: Classical Expressive Pedaling (Una Corda + Half-Pedal + Sostenuto)', async ({
    page,
  }) => {
    // Step A: Engage Una Corda soft pedal for expressive timbral softening
    const unaCordaBtn = page.getByTestId('pedal-una-corda');
    await unaCordaBtn.click();
    expect(await unaCordaBtn.getAttribute('aria-pressed')).toBe('true');

    // Step B: Strike and hold bass anchor chord (C4 MIDI 60 + G4 MIDI 67)
    const c4Key = page.locator('[data-midi="60"]');
    const g4Key = page.locator('[data-midi="67"]');
    await c4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await g4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(250);
    const bassHeld = await sampleDestinationAudio(page);
    expect(bassHeld.rms).toBeGreaterThan(0.002);
    expect(bassHeld.nonFiniteCount).toBe(0);

    // Step C: Engage Sostenuto to selectively latch the bass dampers
    const sostenutoBtn = page.getByTestId('pedal-sostenuto');
    await sostenutoBtn.click();
    expect(await sostenutoBtn.getAttribute('aria-pressed')).toBe('true');

    // Release bass keys — bass strings continue ringing via Sostenuto latch
    await c4Key.dispatchEvent('pointerup', { buttons: 0 });
    await g4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(250);
    const bassLatched = await sampleDestinationAudio(page);
    expect(bassLatched.rms).toBeGreaterThan(0.001);
    expect(bassLatched.nonFiniteCount).toBe(0);

    // Step D: Modulate continuous half-pedal (0.5 clearance) while playing melody notes
    await setSlider(page, 'sustain-slider', 0.5);
    const e4Key = page.locator('[data-midi="64"]');
    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    await page.waitForTimeout(150);
    await e4Key.dispatchEvent('pointerup', { buttons: 0 });
    await page.waitForTimeout(200);

    const halfPedalAudio = await sampleDestinationAudio(page);
    expect(halfPedalAudio.nonFiniteCount).toBe(0);
    expect(halfPedalAudio.peak).toBeLessThanOrEqual(1.0);
    expect(halfPedalAudio.rms).toBeGreaterThan(0.001);

    // Step E: Release half-pedal to 0.0 while Sostenuto remains engaged
    await setSlider(page, 'sustain-slider', 0.0);
    await page.waitForTimeout(300);
    const onlySostAudio = await sampleDestinationAudio(page);
    expect(onlySostAudio.nonFiniteCount).toBe(0);
    expect(onlySostAudio.peak).toBeLessThanOrEqual(1.0);
    expect(onlySostAudio.rms).toBeGreaterThan(0.0008);

    // Step F: Release Sostenuto and Una Corda, verifying clean decay
    await sostenutoBtn.click();
    await unaCordaBtn.click();
    await expectAudioDecay(page, onlySostAudio.rms);
  });

  test('Tier 4 Scenario 4: Dynamic Studio Voicing & Spatial Reconfiguration Under Heavy Playing', async ({
    page,
  }) => {
    // Step A: Strike sustained heavy polyphonic chord (C4, E4, G4, C5)
    await page.locator('[data-midi="60"]').dispatchEvent('pointerdown', { buttons: 1 });
    await page.locator('[data-midi="64"]').dispatchEvent('pointerdown', { buttons: 1 });
    await page.locator('[data-midi="67"]').dispatchEvent('pointerdown', { buttons: 1 });
    await page.locator('[data-midi="72"]').dispatchEvent('pointerdown', { buttons: 1 });
    await setSlider(page, 'sustain-slider', 1.0);
    await page.waitForTimeout(300);

    const initialChord = await sampleDestinationAudio(page);
    expect(initialChord.rms).toBeGreaterThan(0.003);
    expect(initialChord.nonFiniteCount).toBe(0);
    expect(initialChord.peak).toBeLessThanOrEqual(1.0);

    // Release key presses while sustain holds the heavy chord
    await page.locator('[data-midi="60"]').dispatchEvent('pointerup', { buttons: 0 });
    await page.locator('[data-midi="64"]').dispatchEvent('pointerup', { buttons: 0 });
    await page.locator('[data-midi="67"]').dispatchEvent('pointerup', { buttons: 0 });
    await page.locator('[data-midi="72"]').dispatchEvent('pointerup', { buttons: 0 });

    // Step B: Concurrently sweep voicing parameters under active playing
    await setSlider(page, 'voicing-hammer', 0.85);
    await setSlider(page, 'voicing-unison', 2.8);
    await setSlider(page, 'voicing-stretch', 1.6);
    await setSlider(page, 'voicing-length', 1.4);
    await page.waitForTimeout(150);

    const sweptStats = await sampleDestinationAudio(page);
    expect(sweptStats.nonFiniteCount).toBe(0);
    expect(sweptStats.peak).toBeLessThanOrEqual(1.0);
    expect(sweptStats.rms).toBeGreaterThan(0.001);

    // Step C: Switch through lid states (Open -> Half -> Closed -> Open)
    const lidStates = ['lid-half', 'lid-closed', 'lid-open'];
    for (const lid of lidStates) {
      await page.getByTestId(lid).click();
      await page.waitForTimeout(120);
      const lidAudio = await sampleDestinationAudio(page);
      expect(lidAudio.nonFiniteCount).toBe(0);
      expect(lidAudio.peak).toBeLessThanOrEqual(1.0);
      expect(lidAudio.rms).toBeGreaterThan(0.001);
    }

    // Step D: Switch through listener perspectives under active playing
    const perspectives = [
      'perspective-close',
      'perspective-hall',
      'perspective-binaural',
      'perspective-player',
    ];
    for (const persp of perspectives) {
      await page.getByTestId(persp).click();
      await page.waitForTimeout(120);
      const perspAudio = await sampleDestinationAudio(page);
      expect(perspAudio.nonFiniteCount).toBe(0);
      expect(perspAudio.peak).toBeLessThanOrEqual(1.0);
      expect(perspAudio.rms).toBeGreaterThan(0.001);
    }

    // Step E: Release sustain and verify clean natural decay
    await setSlider(page, 'sustain-slider', 0.0);
    const endStats = await sampleDestinationAudio(page);
    await expectAudioDecay(page, endStats.rms);
  });
});
