# E2E Test Infra: pianosim

## Test Philosophy
- **Opaque-Box & Requirement-Driven**: Tests interact exclusively with observable external boundaries (DOM elements, pointer/keyboard gestures, public `PianoSynthesizer` audio buffer render calls, and standard `AudioContext` states). Tests have zero reliance on private implementation details.
- **Methodology**: Systematic 4-tier testing incorporating Category-Partition, Boundary Value Analysis (BVA), Pairwise Combinatorial Interaction Testing, and Real-World Application Workload Scenarios.
- **Dual-Mode Verification**:
  - Live Browser Interaction & Web Audio Lifecycle (DOM keyboard, glissando via `document.elementFromPoint`, keyboard shortcuts, and AudioContext unlock) executed via Playwright Headless Chromium.
  - High-precision acoustic DSP evaluation (FFT parabolic interpolation, velocity RMS scaling, damper release slope, and 0 NaN/Inf stability) executed via decoupled `PianoSynthesizer` audio buffer verification.

## Feature Inventory
Derived from `ORIGINAL_REQUEST.md` (R1–R3) and `PROJECT.md § Feature Inventory`:

| # | Feature | Source (requirement) | Tier 1 (Feature Coverage) | Tier 2 (Boundaries) | Tier 3 (Combinations) |
|---|---------|----------------------|:-------------------------:|:-------------------:|:---------------------:|
| 1 | Physical Simulation Synthesis Engine | ORIGINAL_REQUEST §R1, PROJECT §2 | 6 tests (T1-SYN-01..06) | 3 tests | ✓ (T3-CMB-01..08) |
| 2 | 88-Key Semantic Virtual Keyboard UI | ORIGINAL_REQUEST §R2, PROJECT §8 | 6 tests (T1-UIK-01..06) | 3 tests | ✓ (T3-CMB-03, 04) |
| 3 | Computer Keyboard Mapping & Repeat Suppression | ORIGINAL_REQUEST §R2, PROJECT §10 | 5 tests (T1-KBD-01..05) | 1 test | ✓ (T3-CMB-03, 05) |
| 4 | Sustain Pedal & Damper Dynamics | ORIGINAL_REQUEST §R1, PROJECT §4 | 5 tests (T1-SUS-01..05) | 2 tests | ✓ (T3-CMB-01, 06, 07, 08) |
| 5 | Web Audio Autoplay Unlock & Lifecycle | ORIGINAL_REQUEST §R2, PROJECT §11 | 5 tests (T1-AUL-01..05) | 1 test | ✓ (T3-CMB-03) |
| 6 | Pitch & Fundamental Frequency Accuracy | ORIGINAL_REQUEST §R1, R3, PROJECT §12 | 6 tests (T1-PIT-01..06) | 3 tests | ✓ (T3-CMB-01, 04) |
| 7 | Velocity Dynamics & Spectral Centroid | ORIGINAL_REQUEST §R1, R3, PROJECT §13 | 5 tests (T1-VEL-01..05) | 3 tests | ✓ (T3-CMB-02) |
| 8 | Damping & Rapid Release Decay | ORIGINAL_REQUEST §R1, R3, PROJECT §14 | 5 tests (T1-DMP-01..05) | 2 tests | ✓ (T3-CMB-01, 02) |
| **Total** | **8 Core Requirement Features** | | **49 tests** | **13 tests** | **9 tests** |

## Test Architecture
- **Test Runner**:
  - `npm run test:e2e`: Executes Playwright E2E test runner (`playwright test`).
  - Runner invokes Headless Chromium with `--autoplay-policy=no-user-gesture-required` and `--no-sandbox` on local Vite web server (`http://localhost:5173/pianosim/`; override with `PIANOSIM_PORT=<port>` when 5173 is unavailable, e.g. `PIANOSIM_PORT=5188 npx playwright test`).
  - Unit & acoustic sanity tests run independently via `npm test` (`vitest run`).
- **Pass/Fail Semantics**:
  - Strict exit code 0 requirement.
  - Zero console errors (`page.on('console', msg => { if (msg.type() === 'error') ... })`).
  - Peak sample bounds: $\max |x[n]| \le 1.0$.
  - Numerical stability: 0 NaN, 0 Inf across all rendered frames.
  - Fundamental pitch accuracy within $\pm 1.5$ Hz for standard notes (A4 = 440 Hz).
  - Velocity dynamic range $\ge 12$ dB between $v=32$ and $v=127$, spectral centroid elevation ratio $\ge 1.3\times$.
  - Damper decay rate $\ge 10\times$ faster than natural sustain decay rate.
- **Directory Layout**:
```
tests/e2e/
├── tier1-feature/
│   ├── synthesis.spec.ts         # 6 tests: buffer format, silence, energy, finite bounds, sample rates, allNotesOff
│   ├── ui-keys.spec.ts           # 6 tests: 88 keys (52W/36B), MIDI range 21-108, pointerdown/up/cancel, z-index
│   ├── keyboard-bindings.spec.ts # 5 tests: white/black keybindings, keyup, repeat suppression, octave shift
│   ├── sustain-pedal.spec.ts     # 5 tests: spacebar toggle, sustain hold on keyup, pedal release drop, visual sync
│   ├── audio-unlock.spec.ts      # 5 tests: suspended state, unlock button resume, key click resume, idempotency
│   ├── pitch-accuracy.spec.ts    # 6 tests: A4 (440Hz), C4 (261.63Hz), A1, A5, semitone ratio, frequency display
│   ├── velocity-response.spec.ts # 5 tests: RMS monotonic scaling, dynamic range >=12dB, spectral centroid, max velocity
│   └── damping-release.spec.ts   # 5 tests: sustain decay slope, noteOff damper drop, decay ratio >=10x, anti-click
├── tier2-boundary/
│   ├── register-boundaries.spec.ts # 3 tests: A0 (27.5Hz), C8 (4186Hz), register crossover continuity
│   ├── velocity-extremes.spec.ts   # 3 tests: velocity 0 (silent), velocity 1 (pianississimo), velocity 127
│   ├── voice-exhaustion.spec.ts    # 2 tests: rapid 20ms note tremolo, polyphony saturation & 24-voice stealing
│   └── invalid-inputs.spec.ts      # 5 tests: out-of-range low/high MIDI, non-numeric velocity, redundant noteOff, rapid unlock
├── tier3-combination/
│   ├── chords-sustain.spec.ts          # 3 tests: polyphonic chords + sustain, pedal pumping, panic allNotesOff
│   ├── staccato-velocity.spec.ts       # 1 test: rapid alternating staccato with velocity dynamics
│   ├── multi-input-concurrency.spec.ts # 2 tests: pointer + computer keyboard simultaneous input, octave shift during hold
│   └── glissando-sustain.spec.ts       # 2 tests: 16-key pointer glissando with sustain, half-pedal transient catch
├── tier4-application/
│   ├── bach-prelude.spec.ts      # 1 test: Bach BWV 846 arpeggio motif with continuous sustain resonance
│   ├── fur-elise.spec.ts         # 1 test: Beethoven Für Elise opening motif with expressive dynamics & noteOffs
│   ├── cadence-chords.spec.ts    # 1 test: ii-V-I cadential progression with dynamic swell and long resonance decay
│   ├── glissando-stress.spec.ts  # 1 test: 48-note chromatic glissando burst under voice-stealing cap
│   └── endurance-session.spec.ts # 1 test: 200-event extended performance session asserting zero memory leak
└── helpers/
    ├── audio-analysis.ts         # FFT, parabolic peak interpolation, RMS, spectral centroid calculation
    ├── dom-driver.ts             # Pointer and keyboard event dispatch helpers and DOM assertions
    ├── engine.ts                 # binding to the REAL src/audio/PianoSynthesizer (no mock); renderMono() L/R render + mixdown
    └── page-audio.ts             # browser AudioContext spy + passive AnalyserNode tap on the real audio output
```

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity | Pass Criteria |
|---|----------|--------------------|------------|---------------|
| 1 | Bach Prelude BWV 846 Arpeggios (`bach-prelude.spec.ts`) | Synthesis, Sustain, Pitch, Damping | High | 16-note arpeggio with pedal down; FFT confirms distinct harmonic peaks; peak $\le 1.0$; 0 NaN/Inf |
| 2 | Beethoven Für Elise Motif (`fur-elise.spec.ts`) | UI Keys, Keyboard, Pitch, Velocity, Damping | Medium | Expressive melodic phrase; rapid semitone alternation; accurate pitch contour; crisp damper quenching |
| 3 | ii-V-I Cadential Progressions (`cadence-chords.spec.ts`) | Chords, Velocity Dynamic Swell, Sustain | High | 4-voice polyphony per chord; dynamic RMS swell matches $p \to mf \to f$; smooth multi-second decay |
| 4 | Chromatic Glissando Burst (`glissando-stress.spec.ts`) | Glissando, Voice Stealing, Polyphony Bound | High | 48 rapid chromatic notes under pedal; active voices capped at 24; smooth crossfade; no audio dropout |
| 5 | Extended Two-Hand Session (`endurance-session.spec.ts`) | All UI & Audio Features Concurrently | High | 200 alternating musical events over 30s; bounded memory; 0 console errors; zero audio corruption |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: 49 test cases (incl. `engine-in-browser` and `ui-audio-path`: real engine executed in the page and observed at the AudioContext destination) ($\ge 5$ isolated tests for each of the 8 features).
- **Tier 2 (Boundary & Corner Cases)**: 13 test cases (extremes of pitch, velocity, polyphony, and error handling).
- **Tier 3 (Cross-Feature Combinations)**: 9 test cases (concurrency, pedal pumping, chord sustain, glissando sustain).
- **Tier 4 (Real-World Application Scenarios)**: 5 realistic application-level musical workloads.
- **Grand Total**: **76 distinct, opaque-box E2E test cases**.
