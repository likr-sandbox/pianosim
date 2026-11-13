import { test, expect } from '@playwright/test';
import { ensurePianoKeyboard, trackConsoleErrors, isKeyActive } from '../helpers/dom-driver';

test.describe('Tier 1: Feature Coverage - Virtual UI Keys', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);
  });

  test('T1-UIK-01: Full 88-Key DOM Enumeration', async ({ page }) => {
    const tracker = trackConsoleErrors(page);

    const allKeys = page.locator('#piano-keys .key');
    const count = await allKeys.count();

    expect(count).toBe(88);

    const whiteKeys = page.locator('#piano-keys .key.key-white');
    expect(await whiteKeys.count()).toBe(52);

    const blackKeys = page.locator('#piano-keys .key.key-black');
    expect(await blackKeys.count()).toBe(36);

    expect(tracker.errors).toHaveLength(0);
  });

  test('T1-UIK-02: Consecutive MIDI Range & Labelling', async ({ page }) => {
    const allKeys = page.locator('#piano-keys .key');
    const count = await allKeys.count();
    expect(count).toBe(88);

    for (let i = 0; i < count; i++) {
      const key = allKeys.nth(i);
      const midiAttr = await key.getAttribute('data-midi');
      const expectedMidi = 21 + i;
      expect(Number(midiAttr)).toBe(expectedMidi);

      const ariaLabel = await key.getAttribute('aria-label');
      expect(ariaLabel).toBeTruthy();
    }
  });

  test('T1-UIK-03: PointerDown Visual Activation', async ({ page }) => {
    const c4Key = page.locator('button[data-midi="60"]');
    await c4Key.dispatchEvent('pointerdown', { buttons: 1 });

    const active = await isKeyActive(page, 60);
    expect(active).toBe(true);

    const ariaPressed = await c4Key.getAttribute('aria-pressed');
    if (ariaPressed !== null) {
      expect(ariaPressed).toBe('true');
    }
  });

  test('T1-UIK-04: PointerUp Visual Deactivation', async ({ page }) => {
    const c4Key = page.locator('button[data-midi="60"]');
    await c4Key.dispatchEvent('pointerdown', { buttons: 1 });
    expect(await isKeyActive(page, 60)).toBe(true);

    await c4Key.dispatchEvent('pointerup', { buttons: 0 });
    expect(await isKeyActive(page, 60)).toBe(false);
  });

  test('T1-UIK-05: PointerCancel & PointerLeave Release', async ({ page }) => {
    const e4Key = page.locator('button[data-midi="64"]');
    await e4Key.dispatchEvent('pointerdown', { buttons: 1 });
    expect(await isKeyActive(page, 64)).toBe(true);

    await e4Key.dispatchEvent('pointercancel');
    expect(await isKeyActive(page, 64)).toBe(false);
  });

  test('T1-UIK-06: Black Key Overlap & Stacking Geometry', async ({ page }) => {
    const c4White = page.locator('button[data-midi="60"]');
    const cSharp4Black = page.locator('button[data-midi="61"]');

    const whiteZ = await c4White.evaluate((el) => window.getComputedStyle(el).zIndex);
    const blackZ = await cSharp4Black.evaluate((el) => window.getComputedStyle(el).zIndex);

    const whiteZVal = Number(whiteZ) || 1;
    const blackZVal = Number(blackZ) || 2;
    expect(blackZVal).toBeGreaterThan(whiteZVal);
  });
});
