import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Vite 配置：把 /api 和 /ws 代理到后端 FastAPI（默认 8000），
// 这样浏览器访问同源，无需在后端单独配置 CORS。
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // 普通 HTTP 接口
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // WebSocket：训练进度推送
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
