/** PerfProfile constants, 18-knob metadata, presets (synced with piano-spec), and reconfiguration tiers. */

export const PERF_PRESET_FULL = 0 as const;
export const PERF_PRESET_BALANCED = 1 as const;
export const PERF_PRESET_REALTIME = 2 as const;

export type PerfPresetId =
  | typeof PERF_PRESET_FULL
  | typeof PERF_PRESET_BALANCED
  | typeof PERF_PRESET_REALTIME;

export type PerfPresetSlug = 'full' | 'balanced' | 'realtime';

/** Default live AudioWorklet performance preset (Offline WAV/render defaults to Full). */
export const DEFAULT_LIVE_PERF_PRESET: PerfPresetId = PERF_PRESET_BALANCED;

export interface PerfProfileConfig {
  readonly horizMaxHz: number;
  readonly deadModePruneDb: number;
  readonly lowOrderEnergyHorizonDb: number;
  readonly energyHorizonDb: number;
  readonly savDormancyDb: number;
  readonly savMaxKeyIdx: number;
  readonly bassPhantomSparse: boolean;
  readonly passiveResMaxHz: number;
  readonly irDurationSecs: number;
  readonly spatialTailSplitTap: number;
  readonly firHeadTaps: number;
  readonly firBlockFft: boolean;
  readonly hallFdnLines: number;
  readonly actionOversample: number;
  readonly polynomialAdaa: boolean;
  readonly shankFlexEnabled: boolean;
  readonly damperControlDecim: number;
  readonly hertzNonlinear: boolean;
  readonly activeBridgePoles: number;
}

export type PerfKnobKey = keyof PerfProfileConfig;

export const PERF_PRESET_SLUGS: Readonly<Record<PerfPresetId, PerfPresetSlug>> = {
  [PERF_PRESET_FULL]: 'full',
  [PERF_PRESET_BALANCED]: 'balanced',
  [PERF_PRESET_REALTIME]: 'realtime',
};

export const PERF_PRESET_LABELS: Readonly<Record<PerfPresetId, string>> = {
  [PERF_PRESET_FULL]: 'Full',
  [PERF_PRESET_BALANCED]: 'Balanced',
  [PERF_PRESET_REALTIME]: 'Realtime',
};

/** Exact calibrated preset configs matching `crates/piano-spec/src/perf_profile.rs` (cb85fc8). */
export const PERF_PRESET_CONFIGS: Readonly<Record<PerfPresetId, PerfProfileConfig>> = {
  [PERF_PRESET_FULL]: {
    horizMaxHz: 20_000,
    deadModePruneDb: -120,
    lowOrderEnergyHorizonDb: -120,
    energyHorizonDb: -120,
    savDormancyDb: -55,
    savMaxKeyIdx: 52,
    bassPhantomSparse: false,
    passiveResMaxHz: 20_000,
    irDurationSecs: 0.5,
    spatialTailSplitTap: 65_535,
    firHeadTaps: 64,
    firBlockFft: false,
    hallFdnLines: 8,
    actionOversample: 8,
    polynomialAdaa: false,
    shankFlexEnabled: true,
    damperControlDecim: 1,
    hertzNonlinear: true,
    activeBridgePoles: 10,
  },
  [PERF_PRESET_BALANCED]: {
    horizMaxHz: 3_500,
    deadModePruneDb: -50,
    lowOrderEnergyHorizonDb: -112,
    energyHorizonDb: -96,
    savDormancyDb: -42,
    savMaxKeyIdx: 39,
    bassPhantomSparse: true,
    passiveResMaxHz: 4_000,
    irDurationSecs: 0.25,
    spatialTailSplitTap: 4_096,
    firHeadTaps: 64,
    firBlockFft: false,
    hallFdnLines: 8,
    actionOversample: 4,
    polynomialAdaa: true,
    shankFlexEnabled: true,
    damperControlDecim: 8,
    hertzNonlinear: false,
    activeBridgePoles: 8,
  },
  [PERF_PRESET_REALTIME]: {
    horizMaxHz: 2_800,
    deadModePruneDb: -45,
    lowOrderEnergyHorizonDb: -108,
    energyHorizonDb: -84,
    savDormancyDb: -38,
    savMaxKeyIdx: 26,
    bassPhantomSparse: true,
    passiveResMaxHz: 3_500,
    irDurationSecs: 0.18,
    spatialTailSplitTap: 2_048,
    firHeadTaps: 32,
    firBlockFft: true,
    hallFdnLines: 4,
    actionOversample: 4,
    polynomialAdaa: true,
    shankFlexEnabled: false,
    damperControlDecim: 16,
    hertzNonlinear: false,
    activeBridgePoles: 8,
  },
};

export const PERF_KNOB_ID: Readonly<Record<PerfKnobKey, number>> = {
  energyHorizonDb: 0,
  savDormancyDb: 1,
  savMaxKeyIdx: 2,
  passiveResMaxHz: 3,
  hallFdnLines: 4,
  damperControlDecim: 5,
  hertzNonlinear: 6,
  activeBridgePoles: 7,
  shankFlexEnabled: 8,
  polynomialAdaa: 9,
  horizMaxHz: 10,
  deadModePruneDb: 11,
  bassPhantomSparse: 12,
  irDurationSecs: 13,
  spatialTailSplitTap: 14,
  firHeadTaps: 15,
  firBlockFft: 16,
  actionOversample: 17,
  lowOrderEnergyHorizonDb: 18,
};

export type PerfKnobControlKind = 'range' | 'select' | 'bool';
export type PerfReconfigTier = 'none' | 'tier1' | 'tier2' | 'tier3';

export interface PerfKnobSpec {
  readonly key: PerfKnobKey;
  readonly knobId: number;
  readonly snakeCase: string;
  readonly label: string;
  readonly stage: 'Stage 1' | 'Stage 2' | 'Stage 3' | 'Stage 4';
  readonly kind: PerfKnobControlKind;
  readonly min?: number;
  readonly max?: number;
  readonly step?: number;
  readonly options?: ReadonlyArray<number>;
}

export const PERF_KNOB_SPECS: ReadonlyArray<PerfKnobSpec> = [
  { key: 'actionOversample', knobId: 17, snakeCase: 'action_oversample', label: 'Action Oversample', stage: 'Stage 1', kind: 'select', options: [2, 4, 8] },
  { key: 'polynomialAdaa', knobId: 9, snakeCase: 'polynomial_adaa', label: 'Polynomial ADAA', stage: 'Stage 1', kind: 'bool' },
  { key: 'shankFlexEnabled', knobId: 8, snakeCase: 'shank_flex_enabled', label: 'Shank Flex', stage: 'Stage 1', kind: 'bool' },
  { key: 'damperControlDecim', knobId: 5, snakeCase: 'damper_control_decim', label: 'Damper Decim', stage: 'Stage 1', kind: 'select', options: [1, 2, 4, 8, 16] },
  { key: 'horizMaxHz', knobId: 10, snakeCase: 'horiz_max_hz', label: 'Horiz Max Hz', stage: 'Stage 2', kind: 'range', min: 500, max: 20_000, step: 100 },
  { key: 'deadModePruneDb', knobId: 11, snakeCase: 'dead_mode_prune_db', label: 'Mode Prune dB', stage: 'Stage 2', kind: 'range', min: -120, max: -20, step: 1 },
  { key: 'lowOrderEnergyHorizonDb', knobId: 18, snakeCase: 'low_order_energy_horizon_db', label: 'Low Energy Horizon dB', stage: 'Stage 2', kind: 'range', min: -120, max: -40, step: 1 },
  { key: 'energyHorizonDb', knobId: 0, snakeCase: 'energy_horizon_db', label: 'Energy Horizon dB', stage: 'Stage 2', kind: 'range', min: -120, max: -40, step: 1 },
  { key: 'savDormancyDb', knobId: 1, snakeCase: 'sav_dormancy_db', label: 'SAV Dormancy dB', stage: 'Stage 2', kind: 'range', min: -80, max: -20, step: 1 },
  { key: 'savMaxKeyIdx', knobId: 2, snakeCase: 'sav_max_key_idx', label: 'SAV Max Key', stage: 'Stage 2', kind: 'range', min: 0, max: 52, step: 1 },
  { key: 'bassPhantomSparse', knobId: 12, snakeCase: 'bass_phantom_sparse', label: 'Sparse Bass Phantom', stage: 'Stage 2', kind: 'bool' },
  { key: 'passiveResMaxHz', knobId: 3, snakeCase: 'passive_res_max_hz', label: 'Passive Res Hz', stage: 'Stage 2', kind: 'range', min: 1_000, max: 20_000, step: 100 },
  { key: 'hertzNonlinear', knobId: 6, snakeCase: 'hertz_nonlinear', label: 'Hertz Nonlinear', stage: 'Stage 3', kind: 'bool' },
  { key: 'activeBridgePoles', knobId: 7, snakeCase: 'active_bridge_poles', label: 'Bridge Poles', stage: 'Stage 3', kind: 'range', min: 4, max: 10, step: 1 },
  { key: 'irDurationSecs', knobId: 13, snakeCase: 'ir_duration_secs', label: 'Body IR Secs', stage: 'Stage 4', kind: 'range', min: 0.05, max: 0.5, step: 0.01 },
  { key: 'spatialTailSplitTap', knobId: 14, snakeCase: 'spatial_tail_split_tap', label: 'Spatial Split Tap', stage: 'Stage 4', kind: 'select', options: [1024, 2048, 4096, 8192, 65535] },
  { key: 'firHeadTaps', knobId: 15, snakeCase: 'fir_head_taps', label: 'FIR Head Taps', stage: 'Stage 4', kind: 'select', options: [16, 32, 64] },
  { key: 'firBlockFft', knobId: 16, snakeCase: 'fir_block_fft', label: 'FIR Block FFT', stage: 'Stage 4', kind: 'bool' },
  { key: 'hallFdnLines', knobId: 4, snakeCase: 'hall_fdn_lines', label: 'Hall FDN Lines', stage: 'Stage 4', kind: 'select', options: [4, 8, 16] },
];

export function clonePerfConfig(cfg: PerfProfileConfig): PerfProfileConfig {
  return { ...cfg };
}

export function encodeKnobValue(key: PerfKnobKey, cfg: PerfProfileConfig): number {
  const val = cfg[key];
  return typeof val === 'boolean' ? (val ? 1 : 0) : val;
}

export function perfConfigsEqual(a: PerfProfileConfig, b: PerfProfileConfig): boolean {
  for (const spec of PERF_KNOB_SPECS) {
    if (a[spec.key] !== b[spec.key]) return false;
  }
  return true;
}

export function classifyPerfChange(
  prev: PerfProfileConfig,
  next: PerfProfileConfig,
): PerfReconfigTier {
  const tier3Changed =
    prev.horizMaxHz !== next.horizMaxHz ||
    prev.deadModePruneDb !== next.deadModePruneDb ||
    prev.bassPhantomSparse !== next.bassPhantomSparse ||
    prev.actionOversample !== next.actionOversample ||
    (prev.polynomialAdaa && !next.polynomialAdaa);
  if (tier3Changed) return 'tier3';

  const tier2Changed =
    prev.irDurationSecs !== next.irDurationSecs ||
    prev.spatialTailSplitTap !== next.spatialTailSplitTap ||
    prev.firHeadTaps !== next.firHeadTaps ||
    prev.firBlockFft !== next.firBlockFft;
  if (tier2Changed) return 'tier2';

  if (!perfConfigsEqual(prev, next)) return 'tier1';
  return 'none';
}

export function resolvePerfPresetBySlug(slug: string | null | undefined): PerfPresetId {
  const norm = (slug ?? '').trim().toLowerCase();
  if (norm === 'full') return PERF_PRESET_FULL;
  if (norm === 'realtime') return PERF_PRESET_REALTIME;
  if (norm === 'balanced') return PERF_PRESET_BALANCED;
  return DEFAULT_LIVE_PERF_PRESET;
}

export function resolvePerfPresetFromUrl(search?: string): PerfPresetId {
  const rawSearch =
    search ?? (typeof globalThis.location !== 'undefined' ? globalThis.location.search : '');
  if (!rawSearch) return DEFAULT_LIVE_PERF_PRESET;
  const params = new URLSearchParams(rawSearch);
  return resolvePerfPresetBySlug(params.get('perf'));
}
