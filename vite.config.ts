import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const isProd = mode === 'production';

    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      // 生产构建：代码混淆与反调试保护
      build: {
        // 启用 terser 进行深度混淆（替代默认 esbuild minify）
        minify: isProd ? 'terser' : 'esbuild',
        terserOptions: isProd ? {
          compress: {
            // 移除 console.* 调用
            drop_console: true,
            // 移除 debugger 语句
            drop_debugger: true,
            // 死代码消除
            dead_code: true,
            // 移除未使用的函数
            unused: true,
            // 内联常量表达式
            conditionals: true,
            // 优化 if 条件
            sequences: true,
            // 合并连续 var 声明
            join_vars: true,
            // 循环优化
            loops: true,
            // 表达式优化
            properties: true,
            // 警告消除
            warnings: false,
          },
          mangle: {
            // 混淆所有变量名和函数名
            toplevel: true,
            // 保留 React 相关的属性名（避免运行时错误）
            reserved: [
              'React', 'Component', 'Fragment', 'createElement',
              'useState', 'useEffect', 'useRef', 'useMemo', 'useCallback',
              'Canvas', 'useFrame', 'Three', 'THREE',
            ],
            // 混淆属性名（较激进，但能显著增加逆向难度）
            properties: {
              regex: /^_/,
            },
          },
          format: {
            // 移除所有注释
            comments: false,
            // 不输出 beautify
            beautify: false,
          },
        } : undefined,
        // 启用 CSS 代码分割与压缩
        cssCodeSplit: true,
        // chunk 大小警告阈值
        chunkSizeWarningLimit: 1000,
        rollupOptions: {
          output: {
            // 将第三方库拆分为独立 chunk，增加逆向分析复杂度
            manualChunks: {
              'vendor-react': ['react', 'react-dom'],
              'vendor-three': ['three', '@react-three/fiber', '@react-three/drei'],
            },
            // 混淆 chunk 文件名
            chunkFileNames: isProd ? 'assets/[hash].js' : 'assets/[name]-[hash].js',
            entryFileNames: isProd ? 'assets/[hash].js' : 'assets/[name]-[hash].js',
            assetFileNames: 'assets/[hash][extname]',
          },
        },
        // 生产模式完全关闭 source map
        sourcemap: !isProd,
      },
    };
});
