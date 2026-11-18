import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// La UI no contiene secretos: la API key se escribe en tiempo de ejecución y viaja en la cabecera X-API-Key.
// En desarrollo, /api se redirige al backend FastAPI local.
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: { '/api': { target: process.env.BACKEND_URL ?? 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
