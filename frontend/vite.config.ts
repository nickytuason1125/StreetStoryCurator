import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// pywebview 6 / WebView2 silently refuses to execute <script type="module"> from
// localhost. This post-build transform rewrites to a plain deferred <script> so
// the bundle runs as a classic script in any webview.
const classicScript = {
  name: 'classic-script',
  enforce: 'post' as const,
  transformIndexHtml(html: string) {
    return html
      .replace(/<script\s+type="module"\s+crossorigin\s+src=/g, '<script defer src=')
      .replace(/<script\s+type="module"\s+src=/g, '<script defer src=')
      .replace(/\s+crossorigin(?=[\s>])/g, '')
  },
}

export default defineConfig({
  plugins: [react(), classicScript],
  server: { open: false, port: 5173 },
  build: {
    rollupOptions: {
      output: {
        format: 'iife',
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: 'assets/[name]-[hash][extname]',
      },
    },
  },
})
