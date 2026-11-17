/** Keyboard, pointer, and live MIDI touch velocity curve mapping. */

export type TouchCurveMode = 'light' | 'natural' | 'heavy';

export const DEFAULT_TOUCH_CURVE: TouchCurveMode = 'natural';

export const TOUCH_CURVE_OPTIONS: ReadonlyArray<{
  readonly mode: TouchCurveMode;
  readonly label: string;
}> = [
  { mode: 'light', label: 'Light' },
  { mode: 'natural', label: 'Natural' },
  { mode: 'heavy', label: 'Heavy' },
];

const LIGHT_GAMMA = 0.65;
const HEAVY_GAMMA = 1.65;

/** Maps normalized `[0, 1]` velocity through the selected touch curve (`natural` is identity). */
export function applyTouchCurveNormalized(norm01: number, mode: TouchCurveMode): number {
  if (!Number.isFinite(norm01) || norm01 <= 0) {
    return 0;
  }
  const clamped = Math.min(1, norm01);
  if (mode === 'natural' || clamped >= 1) {
    return clamped;
  }
  const gamma = mode === 'light' ? LIGHT_GAMMA : HEAVY_GAMMA;
  return Math.pow(clamped, gamma);
}

/**
 * Maps a 7-bit MIDI velocity (`0..=127`) through the selected touch curve:
 * - `0` always maps to `0` (preserving MIDI Note-On vel=0 Note-Off semantics).
 * - Non-zero velocities (`1..=127`) remain `>= 1`.
 * - `natural` is exact identity.
 */
export function applyTouchCurve(vel7: number, mode: TouchCurveMode): number {
  if (!Number.isFinite(vel7)) {
    return 0;
  }
  const v = Math.max(0, Math.min(127, Math.trunc(vel7)));
  if (v === 0 || mode === 'natural' || v === 127) {
    return v;
  }
  const curved = Math.round(applyTouchCurveNormalized(v / 127, mode) * 127);
  return Math.max(1, Math.min(127, curved));
}
