import { defineConfig } from "vitest/config";

export default defineConfig({
  base: "/pianosim/",
  build: {
    target: "esnext",
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: true,
  },
  server: {
    port: 5173,
    open: false,
  },
  test: {
    environment: "node",
    include: ["tests/**/*.{test,spec}.ts"],
  },
});
