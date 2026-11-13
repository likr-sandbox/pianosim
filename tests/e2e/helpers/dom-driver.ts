/**
 * DOM Driver and Interaction Helpers for pianosim Playwright Tests
 * Provides opaque-box user gesture emulation (pointerdown/up/move glissando,
 * keyboard mapping, sustain pedal, element inspection) and console tracking.
 * Strictly passive driver: zero synthetic DOM injection or event listener mocking.
 */

import type { Page, Locator } from '@playwright/test';

export interface ConsoleErrorTracker {
  errors: string[];
  pageErrors: Error[];
}

/**
 * Setup console error and unhandled rejection tracking on page.
 */
export function trackConsoleErrors(page: Page): ConsoleErrorTracker {
  const tracker: ConsoleErrorTracker = {
    errors: [],
    pageErrors: [],
  };

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      tracker.errors.push(msg.text());
    }
  });

  page.on('pageerror', (err) => {
    tracker.pageErrors.push(err);
  });

  return tracker;
}

/**
 * Standard note names for MIDI 0-127
 */
const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

export function getMidiNoteName(midi: number): { note: string; isBlack: boolean } {
  const noteIndex = ((midi % 12) + 12) % 12;
  const octave = Math.floor(midi / 12) - 1;
  const name = NOTE_NAMES[noteIndex];
  const isBlack = name.includes('#');
  return { note: `${name}${octave}`, isBlack };
}

/**
 * Ensure piano application container and keyboard bed are mounted in DOM.
 * Strictly passive synchronization: does NOT inject DOM elements or event listeners.
 */
export async function ensurePianoKeyboard(page: Page, options?: { timeout?: number }): Promise<void> {
  await page.waitForSelector('#app', { state: 'attached', timeout: options?.timeout });
  await page.waitForSelector('#piano-keys', { state: 'attached', timeout: options?.timeout });
}

/**
 * Click a virtual piano key by MIDI note
 */
export async function clickPianoKey(page: Page, midiNote: number): Promise<void> {
  const locator = page.locator(`button[data-midi="${midiNote}"]`);
  await locator.click();
}

/**
 * Press down a virtual piano key (pointerdown)
 */
export async function pressPianoKey(page: Page, midiNote: number): Promise<void> {
  const locator = page.locator(`button[data-midi="${midiNote}"]`);
  await locator.dispatchEvent('pointerdown', { buttons: 1, pointerType: 'mouse' });
}

/**
 * Release a virtual piano key (pointerup)
 */
export async function releasePianoKey(page: Page, midiNote: number): Promise<void> {
  const locator = page.locator(`button[data-midi="${midiNote}"]`);
  await locator.dispatchEvent('pointerup', { buttons: 0, pointerType: 'mouse' });
}

/**
 * Perform a physical mouse glissando sweep between two MIDI keys
 */
export async function dispatchPointerGlissando(
  page: Page,
  startMidi: number,
  endMidi: number,
  options?: { yOffsetRatio?: number }
): Promise<void> {
  const startEl = page.locator(`button[data-midi="${startMidi}"]`);
  const endEl = page.locator(`button[data-midi="${endMidi}"]`);

  if ((await startEl.count()) === 0 || (await endEl.count()) === 0) {
    return;
  }

  await startEl.waitFor({ state: 'visible' });
  await endEl.waitFor({ state: 'visible' });

  const startBox = await startEl.boundingBox();
  const endBox = await endEl.boundingBox();

  if (startBox && endBox) {
    const yRatio = options?.yOffsetRatio ?? 0.75;
    const startX = startBox.x + startBox.width / 2;
    const startY = startBox.y + startBox.height * yRatio;
    const endX = endBox.x + endBox.width / 2;
    const endY = endBox.y + endBox.height * yRatio;

    await page.mouse.move(startX, startY);
    await page.mouse.down();
    const steps = Math.max(10, Math.abs(endMidi - startMidi) * 3);
    await page.mouse.move(endX, endY, { steps });
    await page.mouse.up();
  }
}

/**
 * Get piano key locator
 */
export function getKeyElement(page: Page, midiNote: number): Locator {
  return page.locator(`button[data-midi="${midiNote}"]`);
}

/**
 * Check if a piano key has the .active CSS class or aria-pressed="true"
 */
export async function isKeyActive(page: Page, midiNote: number): Promise<boolean> {
  const el = getKeyElement(page, midiNote);
  if ((await el.count()) === 0) return false;
  const classes = await el.getAttribute('class');
  const ariaPressed = await el.getAttribute('aria-pressed');
  return (classes ? classes.includes('active') : false) || ariaPressed === 'true';
}
