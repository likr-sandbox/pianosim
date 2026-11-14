# pianosim

Pure-Rust physical-modeling grand piano synthesizer monorepo targeting single-threaded Web `AudioWorkletGlobalScope` (`wasm32-unknown-unknown` + `simd128`) and native verification/benchmarking (`piano-cli`).

> The prior v1 TypeScript DWG prototype is archived at Git tag `v0.1.0-ts-dwg`.

## Architecture Overview

1. **Layer 1 — Offline Modal Generation & Adaptive IR Synthesis**
   - `piano-body-gen`: Pure-Rust structural-acoustic solver (orthotropic soundboard, ribs, rim, 3D air cavity, cast-iron frame) extracting reduced bridge modes and pole-canceled radiation modes.
   - `piano-radiator::ir_synth`: Host sample-rate adaptive 9-zone stereo IR synthesis (8 bridge zones + 1 keybed zone).
2. **Layer 2 — Zero-Allocation 4-Stage Real-Time DSP**
   - **Stage 1 (`piano-action`)**: Local 8x (384 kHz) Timoshenko shank, Stulov hysteresis felt, K-method/ADAA contact, and unilateral hysteresis damper.
   - **Stage 2 (`piano-strings`)**: Always-on SIMD ZOH modal string bank (~26k modes across 88 keys / 230 strings), drift-controlled SAV nonlinearity, longitudinal precursors, and phantom partials.
   - **Stage 3 (`piano-bridge`)**: Reduced-mode bidirectional bridge coupling, duplex strings, and emergent 3-pedal mechanics.
   - **Stage 4 (`piano-radiator`)**: Zero-added-latency 64-tap SIMD FIR head + partitioned NUPOLS FFT convolution.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System design, 4-stage physical model, and FTO clean-room invariants
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — M1–M8 milestones and 10-crate workspace map
- [`docs/DECISIONS_AND_FALLBACKS.md`](docs/DECISIONS_AND_FALLBACKS.md) — Deferred options (D1–D8) and recovery seams
- [`docs/SWARM_WORKFLOW.md`](docs/SWARM_WORKFLOW.md) — Multi-agent worktree, commit, and milestone push workflow
- [`.agents/AGENTS.md`](.agents/AGENTS.md) — Living project conventions and accumulated learnings

## Workspace Commands

```bash
export PATH="$HOME/.cargo/bin:$PATH"

cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo check --package piano-wasm --target wasm32-unknown-unknown
```
