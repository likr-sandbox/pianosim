# pianosim

Browser-based real-time piano physical-model synthesizer with an 88-key interactive keyboard.

**Live demo:** https://likr-sandbox.github.io/pianosim/

- Engine (`src/audio/`): hammer model + digital waveguide strings + soundboard resonance, running in an AudioWorklet.
- UI (`src/ui/`): 88-key keyboard (A0–C8) with mouse/touch (glissando), computer-keyboard mapping (A S D F … / W E T …, Z/X octave), Space sustain pedal, velocity slider.

## Development

```
npm ci
npm run dev          # local dev server
npm test             # acoustic/unit tests (vitest)
npm run build        # typecheck + production build
npm run test:e2e     # Playwright (PIANOSIM_PORT=<port> to override 5173)
```

## Deployment

Pushing to `main` runs `.github/workflows/deploy.yml`, which tests, builds and publishes `dist/` to GitHub Pages (repo Settings → Pages → Source: GitHub Actions). Vite `base` is `/pianosim/`.
