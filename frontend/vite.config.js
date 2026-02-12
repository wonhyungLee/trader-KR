import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: './', // 서브패스/루트 모두 대응하도록 상대 경로 사용
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/bnf': {
        target: 'http://localhost:5001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/bnf/, '')
      }
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})
