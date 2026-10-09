import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Send /api/... to the Python backend (uvicorn server.app:app --port 8000).
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
