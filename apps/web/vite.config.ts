import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { defineConfig, transformWithEsbuild, type Plugin } from 'vite';

const WORKLET_SUFFIX = '?worklet';

/**
 * Compiles TypeScript AudioWorklet processors into standalone JS strings
 * so they load identically via Blob URL in dev, preview, and GitHub Pages.
 */
function audioWorkletPlugin(): Plugin {
  let isBuild = false;

  return {
    name: 'pianosim-audio-worklet',
    configResolved(config) {
      isBuild = config.command === 'build';
    },
    async resolveId(source, importer) {
      if (!source.endsWith(WORKLET_SUFFIX)) {
        return null;
      }
      const cleanSource = source.slice(0, -WORKLET_SUFFIX.length);
      const resolved = await this.resolve(cleanSource, importer, { skipSelf: true });
      if (!resolved) {
        return null;
      }
      return `${resolved.id}${WORKLET_SUFFIX}`;
    },
    async load(id) {
      if (!id.endsWith(WORKLET_SUFFIX)) {
        return null;
      }
      const filePath = id.slice(0, -WORKLET_SUFFIX.length);
      this.addWatchFile(filePath);
      const source = await readFile(filePath, 'utf8');
      const result = await transformWithEsbuild(source, filePath, {
        loader: 'ts',
        target: 'es2022',
        format: 'esm',
        minify: isBuild,
      });
      return `export default ${JSON.stringify(result.code)};`;
    },
  };
}

const appRoot = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.env.PIANOSIM_PORT || 5188);

export default defineConfig({
  root: appRoot,
  base: '/pianosim/',
  plugins: [audioWorkletPlugin()],
  build: {
    target: 'es2022',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    emptyOutDir: true,
  },
  server: {
    port,
    strictPort: true,
    open: false,
  },
  preview: {
    port,
    strictPort: true,
    open: false,
  },
});
