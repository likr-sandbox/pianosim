import { test, expect } from '@playwright/test';
import { createPianoSynthesizer, renderMono } from '../helpers/engine';
import { verifySampleBounds } from '../helpers/audio-analysis';
import { ensurePianoKeyboard, trackConsoleErrors, clickPianoKey } from '../helpers/dom-driver';

test.describe('Tier 4: Application Scenario - Extended Endurance Session', () => {
  test('T4-APP-05: Extended 200-Event Performance Session', async ({ page }) => {
    const tracker = trackConsoleErrors(page);
    await page.goto('./');
    await ensurePianoKeyboard(page);

    const synth = createPianoSynthesizer(44100);

    // 200 musical events alternating between bass accompaniment and treble melody
    const bassNotes = [36, 40, 43, 48, 52, 55]; // C2-G3
    const melodyNotes = [60, 64, 67, 72, 76, 79, 84]; // C4-C6

    const blockSize = 512;
    const blockBuf = new Float32Array(blockSize);

    let pedalState = false;

    for (let eventIdx = 0; eventIdx < 200; eventIdx++) {
      // Toggle pedal periodically
      if (eventIdx % 40 === 0) {
        pedalState = !pedalState;
        synth.setSustainPedal(pedalState);
      }

      if (eventIdx % 2 === 0) {
        const bassMidi = bassNotes[eventIdx % bassNotes.length];
        synth.noteOn({ midiNote: bassMidi, velocity: 80 });
      } else {
        const melodyMidi = melodyNotes[eventIdx % melodyNotes.length];
        synth.noteOn({ midiNote: melodyMidi, velocity: 90 });
      }

      // Render 512 samples per event
      renderMono(synth, blockBuf, blockSize);

      // Randomly release notes
      if (eventIdx % 3 === 0) {
        const releaseMidi = eventIdx % 2 === 0 ? bassNotes[eventIdx % bassNotes.length] : melodyNotes[eventIdx % melodyNotes.length];
        synth.noteOff({ midiNote: releaseMidi });
      }

      // Check stability on every 20th block
      if (eventIdx % 20 === 0) {
        const bounds = verifySampleBounds(blockBuf);
        expect(bounds.hasNaN).toBe(false);
        expect(bounds.hasInf).toBe(false);
        expect(bounds.absMax).toBeLessThanOrEqual(1.0);
      }

      // Trigger DOM clicks periodically
      if (eventIdx % 50 === 0) {
        await clickPianoKey(page, 60);
      }
    }

    expect(tracker.errors).toHaveLength(0);
    expect(tracker.pageErrors).toHaveLength(0);
  });
});
