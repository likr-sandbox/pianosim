import { test, expect } from '@playwright/test';
import { ensurePianoKeyboard, isKeyActive, pressPianoKey, releasePianoKey } from '../helpers/dom-driver';

test.describe('Tier 3: Combination - Multi-Input Concurrency', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);
    const keyCount = await page.locator('#piano-keys .key').count();
    expect(keyCount).toBe(88);
  });

  test('T3-CMB-03: Concurrent Pointer + Computer Keyboard', async ({ page }) => {
    // Pointer holds C3 (MIDI 48)
    await pressPianoKey(page, 48);
    expect(await isKeyActive(page, 48)).toBe(true);

    // Simultaneously press computer keyboard KeyK (C5, MIDI 72)
    await page.keyboard.down('KeyK');
    expect(await isKeyActive(page, 72)).toBe(true);
    expect(await isKeyActive(page, 48)).toBe(true);

    // Release computer keyboard KeyK
    await page.keyboard.up('KeyK');
    expect(await isKeyActive(page, 72)).toBe(false);
    // Pointer key C3 continues holding
    expect(await isKeyActive(page, 48)).toBe(true);

    // Release pointer key
    await releasePianoKey(page, 48);
    expect(await isKeyActive(page, 48)).toBe(false);
  });

  test('T3-CMB-05: Octave Shift While Holding Key', async ({ page }) => {
    // Press and hold KeyA (maps to C4, MIDI 60)
    await page.keyboard.down('KeyA');
    expect(await isKeyActive(page, 60)).toBe(true);

    // Shift octave down while holding KeyA
    await page.keyboard.press('KeyZ');

    // Release KeyA
    await page.keyboard.up('KeyA');

    // Neither C4 (60) nor C3 (48) should be stuck active
    expect(await isKeyActive(page, 60)).toBe(false);
    expect(await isKeyActive(page, 48)).toBe(false);
  });
});
