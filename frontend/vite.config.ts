import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const backendProxy = {
  '/api':    { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/thumbs': { target: 'http://127.0.0.1:8000', changeOrigin: true },
  '/static': { target: 'http://127.0.0.1:8000', changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: {
    open: false,
    port: 5173,
    proxy: backendProxy,
  },
  preview: {
    port: 5173,
    proxy: backendProxy,
  },
  resolve: {
    // Force CJS builds of @dnd-kit to avoid ESM circular-dependency TDZ errors.
    alias: {
      '@dnd-kit/core': path.resolve(__dirname, 'node_modules/@dnd-kit/core/dist/index.js'),
      '@dnd-kit/sortable': path.resolve(__dirname, 'node_modules/@dnd-kit/sortable/dist/index.js'),
      '@dnd-kit/utilities': path.resolve(__dirname, 'node_modules/@dnd-kit/utilities/dist/index.js'),
    },
  },
  build: {
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
})
