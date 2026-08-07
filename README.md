# smodeltrans

`smodeltrans` is a Tauri 2 desktop workspace for a recoverable image-translation flow, built with Vue 3, TypeScript, and Naive UI. The current first-release provider is a clearly labelled deterministic local preview; it is not OCR or a production translation model.

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
