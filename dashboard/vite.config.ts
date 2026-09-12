import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The build lands in dashboard/dist, which the FastAPI app mounts at
// /dashboard if it exists. One serving surface: the dashboard is a
// presentation layer over the platform API, so the process that owns the data
// serves it. `base` matches that mount point so assets resolve.
export default defineConfig({
  plugins: [react()],
  base: "/dashboard/",
  server: {
    port: 5173,
    // In development the API runs separately. Proxying avoids CORS entirely
    // rather than loosening the API's origin policy for a dev convenience.
    proxy: {
      "/platform": "http://localhost:8080",
      "/predict": "http://localhost:8080",
      "/ready": "http://localhost:8080",
      "/health": "http://localhost:8080",
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test-setup.ts"],
  },
});
