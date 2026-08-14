# smodeltrans

`smodeltrans` is a Tauri 2 desktop workspace built with Vue 3, TypeScript, Naive UI, and Candle. It provides local Hy-MT2 text translation, PP-OCR recognition, OCR-assisted image translation, persistent settings, and live model runtime monitoring with explicit load and unload controls.

## Development

```bash
bun install
bun run dev
```

Build the web bundle with `bun run build`, or compile the desktop shell with `CI=false bun run tauri build --no-bundle`.

## Engineering plan

The first-release image translation workflow, Tauri boundary, provider seam, non-goals, and future integration path are documented in [`docs/ENGINEERING_PLAN.md`](docs/ENGINEERING_PLAN.md).

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Vue - Official](https://marketplace.visualstudio.com/items?itemName=Vue.volar) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
