# pianosim

Pure-Rust physical-modeling grand piano synthesizer monorepo targeting single-threaded Web `AudioWorkletGlobalScope` (`wasm32-unknown-unknown` + `simd128`) with off-thread Prep Web Worker synthesis and native verification/rendering (`piano-cli`).

> The prior v1 TypeScript DWG prototype is archived at Git tag `v0.1.0-ts-dwg`.

## Architecture Overview

1. **Layer 1 — Offline Modal Generation & Adaptive Async IR/Voicing Synthesis**
   - `piano-body-gen`: Pure-Rust `15x20` orthotropic Rayleigh-Ritz + statistical structural-acoustic solver generating 3 `.pbdy` body archetypes (`ConcertGrand`, `UprightStudio`, `VintageSaloon`), extracting 10 reduced bridge poles and pole-canceled radiation modes.
   - `piano-radiator::ir_synth` & Prep Web Worker: Sample-rate adaptive 9-zone stereo IR synthesis (8 bridge zones + 1 keybed zone) with `3.8 kHz` HPF velvet diffuse tail, NUPOLS pre-FFT, and tuning table precomputation transferred zero-copy via `Transferable ArrayBuffer`.
2. **Layer 2 — Zero-Allocation 4-Stage Real-Time DSP**
   - **Stage 1 (`piano-action`)**: Local `8x` (384 kHz) Timoshenko shank, two-timescale Stulov hysteresis felt, K-method + shared 24-point $p$-grid Hermite ADAA, `460`-slot unilateral $C^2$ hysteresis damper pads, and Zone 8 strike/release keybed thump.
   - **Stage 2 (`piano-strings`)**: `~26,184`-slot 64-byte aligned SIMD ZOH modal bank across 88 keys / 230 strings, finite hammer width spatial window $\Phi_w(j)$, 3-mechanism $T_{60}$ damping, `6 kHz` treble WDF reconnection, drift-regulated SAV nonlinearity, `3x6` bass phantom partials, and physical energy-horizon pruning.
   - **Stage 3 (`piano-bridge`)**: 10-pole reduced-mode `8x8` WDF bidirectional bridge coupling, SAV Hertz pin contact, tuning-coupled 64-mode `DuplexBank`, and emergent 3-pedal mechanics (Sustain, continuous Una Corda, Sostenuto).
   - **Stage 4 (`piano-radiator`)**: Zero-added-latency 64-tap SIMD FIR head + Staggered MIMO NUPOLS convolution + 3D ER / `8x8` Hadamard `HallFdn` + zero-latency $C^2$ `StereoLimiter`.
3. **Web Studio (`apps/web`)**: 88-key multi-touch/glissando keyboard, Web MIDI 1.0/2.0, SMF playback & 6 humanized demos, voicing/tuning (`A4`, 4 classical temperaments, `Condition`), 3 body archetypes & 4 perspectives, Touch Curves, 4 presets, 24-bit TPDF / 32-bit float WAV recorder, and full-block telemetry HUD.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design, 4-stage physical model, and FTO clean-room invariants
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Monorepo crate/web layout and completed M1–M8 + Wave 1–4 milestones
- [`docs/DECISIONS_AND_FALLBACKS.md`](docs/DECISIONS_AND_FALLBACKS.md) — Deferred options (D1–D8) and recovery seams

## Workspace Commands

```bash
export PATH="$HOME/.cargo/bin:$PATH"

cargo fmt --check
cargo run -p piano-cli -- check-lines
cargo clippy --workspace -- -D warnings
cargo clippy --package piano-wasm --target wasm32-unknown-unknown -- -D warnings
cargo test --workspace
npm run build
cargo run -p piano-cli -- check-wasm-simd apps/web/public/piano_wasm.wasm
npm run test:e2e
```
