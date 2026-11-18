import { fileURLToPath } from 'node:url';
import { defineConfig, devices } from '@playwright/test';

// Default port 5188 avoids environments where 127.0.0.1:5173 black-holes SYNs.
const PORT = Number(process.env.PIANOSIM_PORT || 5188);
const ORIGIN = `http://localhost:${PORT}/pianosim/`;
const REPO_ROOT = fileURLToPath(new URL('../..', import.meta.url));
const WORKERS = Number(process.env.PIANOSIM_WORKERS || (process.env.CI ? 2 : 3));

const CHROMIUM_USE = {
  ...devices['Desktop Chrome'],
  launchOptions: {
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--autoplay-policy=no-user-gesture-required',
    ],
  },
};

const webServerCommand =
  process.env.PIANOSIM_E2E_DEV === '1'
    ? `npm run dev -- --port ${PORT} --strictPort`
    : process.env.CI === 'true' || process.env.PIANOSIM_SKIP_WEB_BUILD === '1'
      ? `npm run preview -- --port ${PORT} --strictPort`
      : `npm run build:web && npm run preview -- --port ${PORT} --strictPort`;

export default defineConfig({
  testDir: './e2e',
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: WORKERS,
  outputDir: '../../test-results',
  reporter: process.env.CI
    ? [['html', { outputFolder: '../../playwright-report', open: 'never' }], ['list']]
    : 'list',
  use: {
    baseURL: ORIGIN,
    trace: 'on-first-retry',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'functional',
      testIgnore: /perf-stability\.spec\.ts/,
      fullyParallel: true,
      use: CHROMIUM_USE,
    },
    {
      name: 'perf-stability',
      testMatch: /perf-stability\.spec\.ts/,
      dependencies: ['functional'],
      fullyParallel: false,
      use: CHROMIUM_USE,
    },
  ],
  webServer: {
    command: webServerCommand,
    cwd: REPO_ROOT,
    // Wait on Vite stdout instead of polling `url`, because unbound IPv4 loopback SYNs time out (~130s) here.
    wait: {
      stdout: /Local:\s+http/,
    },
    timeout: 45_000,
  },
});
