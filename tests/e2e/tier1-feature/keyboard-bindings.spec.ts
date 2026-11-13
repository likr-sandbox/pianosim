import { test, expect } from '@playwright/test';
import { ensurePianoKeyboard, isKeyActive, trackConsoleErrors } from '../helpers/dom-driver';

test.describe('Tier 1: Feature Coverage - Computer Keyboard Bindings', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('./');
    await ensurePianoKeyboard(page);
    const keyCount = await page.locator('#piano-keys .key').count();
    expect(keyCount).toBe(88);
  });

  test('T1-KBD-01: Home Row White Key Mapping', async ({ page }) => {
    const tracker = trackConsoleErrors(page);

    await page.keyboard.down('KeyA');
    expect(await isKeyActive(page, 60)).toBe(true);
    await page.keyboard.up('KeyA');
    expect(await isKeyActive(page, 60)).toBe(false);

    await page.keyboard.down('KeyS');
    expect(await isKeyActive(page, 62)).toBe(true);
    await page.keyboard.up('KeyS');
    expect(await isKeyActive(page, 62)).toBe(false);

    await page.keyboard.down('KeyD');
    expect(await isKeyActive(page, 64)).toBe(true);
    await page.keyboard.up('KeyD');
    expect(await isKeyActive(page, 64)).toBe(false);

    expect(tracker.errors).toHaveLength(0);
  });

  test('T1-KBD-02: Accidental Row Black Key Mapping', async ({ page }) => {
    await page.keyboard.down('KeyW');
    expect(await isKeyActive(page, 61)).toBe(true);
    await page.keyboard.up('KeyW');
    expect(await isKeyActive(page, 61)).toBe(false);

    await page.keyboard.down('KeyE');
    expect(await isKeyActive(page, 63)).toBe(true);
    await page.keyboard.up('KeyE');
    expect(await isKeyActive(page, 63)).toBe(false);
  });

  test('T1-KBD-03: Computer Key Release (Keyup)', async ({ page }) => {
    await page.keyboard.down('KeyA');
    expect(await isKeyActive(page, 60)).toBe(true);

    await page.keyboard.up('KeyA');
    expect(await isKeyActive(page, 60)).toBe(false);
  });

  test('T1-KBD-04: Auto-Repeat Event Suppression', async ({ page }) => {
    // Dispatch repeated keydown events with repeat: true
    await page.keyboard.down('KeyA');
    expect(await isKeyActive(page, 60)).toBe(true);

    // Simulated browser auto-repeat
    await page.evaluate(() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyA', repeat: true }));
    });
    expect(await isKeyActive(page, 60)).toBe(true);

    await page.keyboard.up('KeyA');
    expect(await isKeyActive(page, 60)).toBe(false);
  });

  test('T1-KBD-05: Octave Shift Mapping Adjustment', async ({ page }) => {
    // Shift octave up by pressing KeyX
    await page.keyboard.press('KeyX');

    // Press KeyA: should now trigger C5 (72) instead of C4 (60)
    await page.keyboard.down('KeyA');
    expect(await isKeyActive(page, 72)).toBe(true);
    expect(await isKeyActive(page, 60)).toBe(false);

    await page.keyboard.up('KeyA');
    expect(await isKeyActive(page, 72)).toBe(false);
  });
});
