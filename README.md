# pianosim

Pure-Rust physical-modeling grand piano synthesizer monorepo targeting single-threaded Web `AudioWorkletGlobalScope` (`wasm32-unknown-unknown` + `simd128`) with off-thread Prep / Offline Render Web Workers and native verification/rendering (`piano-cli`).

> The prior v1 TypeScript DWG prototype is archived at Git tag `v0.1.0-ts-dwg`.

## Architecture Overview

1. **Layer 1 — Offline Modal Generation, Adaptive IR/Voicing & Progressive MIDI Rendering**
   - `piano-body-gen`: Pure-Rust `15x20` orthotropic Rayleigh-Ritz + statistical structural-acoustic solver (crown monopole, 2D volume integral, Helmholtz air cavity, bridge pin-line rocking) generating 3 `.pbdy` body archetypes (`ConcertGrand` `1,338` modes, `UprightStudio` `1,287`, `VintageSaloon` `1,297`), extracting 10 literature-calibrated bridge poles and pole-canceled radiation modes.
   - `piano-radiator::ir_synth` & Prep Web Worker: Sample-rate adaptive 9-zone stereo IR synthesis with `3.8 kHz` HPF velvet diffuse tail, `SpaceBlob` v2 (`SPACE_BLOB_MAGIC_V2`) pre-FFT, and tuning table precomputation streamed zero-spike via `BlobStreamer` (`2,048` float chunks -> `piano_space_chunk_ptr` + 16-slice standby clear + 11-block `VoicingCommitState`).
   - `offline-render-worker.ts` & `OfflineStreamPlayer`: Off-thread `Full`-profile rendering of canonical/uploaded `.mid` files with progressive streaming playback (`1.5 s` initial + `0.75 s` chunks) and 24-bit TPDF / 32-bit float WAV export.
2. **Layer 2 — Zero-Allocation 4-Stage Real-Time DSP (`PerfProfile`: `Full` / `Balanced` / `Realtime`, 19 knobs)**
   - **Stage 1 (`piano-action`)**: Selectable `2/4/8x` local oversampled Timoshenko shank, `AgraffeDelayLine` short-string reflection, two-timescale Stulov felt, shared 24-point $p$-grid Hermite ADAA or zero-LUT `PolyPowerApprox`, `464`-slot SoA unilateral $C^2$ damper pads with decimated control-rate clearance/peel interpolation (`step_damper_pair`), and Zone 8 keybed thump.
   - **Stage 2 (`piano-strings`)**: Up to `~26,184`-slot 64-byte aligned SIMD ZOH modal bank across 88 keys / 230 strings, dynamic $\Phi_w(j)$ and `2.6 kHz`-rolled-off `damper_phi`, 3-mechanism $T_{60}$ damping, `6 kHz` treble WDF reconnection, drift-regulated SAV nonlinearity, `3x6` / sparse `2x2` bass phantom partials, and intra-string mode-order energy-horizon pruning.
   - **Stage 3 (`piano-bridge`)**: `4..=10`-pole reduced-mode `8x8` WDF bidirectional bridge coupling, selectable SAV/linearized Hertz pin contact, tuning-coupled 64-mode `DuplexBank`, and emergent 3-pedal mechanics (Sustain, continuous Una Corda, Sostenuto).
   - **Stage 4 (`piano-radiator`)**: Zero-added-latency `16/32/64`-tap SIMD FIR head or `BlockFftHead` + Staggered MIMO / stereo-collapsed NUPOLS convolution + 3D ER / `4/8/16`-line Hadamard `HallFdn` + zero-latency $C^2$ `StereoLimiter`.
3. **Web Studio (`apps/web`)**: 88-key multi-touch/glissando keyboard, Web MIDI 1.0/2.0, progressive `Full`-quality `.mid` demo/file streaming (`apps/web/public/demos/*.mid`), low-latency `AudioWorklet` live auditioning (`Balanced` default, `PerfPanel` / `?perf=`), voicing/tuning (`A4`, 4 classical temperaments, `Condition`), 3 body archetypes & 4 perspectives, Touch Curves, 4 presets, 24-bit TPDF / 32-bit float WAV recorder, and full-block telemetry HUD.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design, 4-stage physical model, `PerfProfile`, and FTO clean-room invariants
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — Monorepo crate/web layout and completed M1–M8 + Wave 1–4 + `v0.3.0` milestones
- [`docs/DECISIONS_AND_FALLBACKS.md`](docs/DECISIONS_AND_FALLBACKS.md) — Deferred options (D1–D8), revised D1 two-path Web execution, and recovery seams

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
npm run bench:web
cargo run --release -p piano-cli -- scenarios
cargo run --release -p piano-cli -- bench --perf-profile balanced
cargo run --release -p piano-cli -- diag-buzz
```
