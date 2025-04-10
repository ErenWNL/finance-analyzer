import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', // This exposes the server to your network
    proxy: {
      '/api': {
        target: 'http://localhost:5001',
        changeOrigin: true
      }
    }
  }
})