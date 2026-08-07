# smodeltrans Engineering Plan

## Purpose and current baseline

`smodeltrans` is a greenfield desktop image-translation workspace built from the stock Tauri 2 + Vue 3 + TypeScript scaffold. The original scaffold exposed only a `greet` command and a generated greeting screen. The first release now focuses on one recoverable workflow:

> choose or drop an image -> preview -> start translation -> processing -> result

The workflow also supports cancellation, reselection, copy, save, and recovery from invalid input or a provider failure.

The product surface is intentionally honest about its current capability. The shipped provider is a deterministic local preview provider, not OCR or a production translation model. It is isolated behind `TranslationProvider` so a verified adapter can replace it without coupling the UI to a model implementation.

## Topology

### 1. Translation product workflow

**Responsibilities**

- Accept a single image from a file picker or drag-and-drop.
- Validate image type and size before previewing it.
- Present an image preview and a clear next action.
- Show `idle`, `preview`, `processing`, `result`, `cancelled`, and recoverable `error` states.
- Let users cancel an active run, choose another image, retry a retained image, copy result text, or save a text result.

**Implemented boundary**

- `src/App.vue` owns the user-visible state machine and stale-run protection.
- The first-release validation policy is PNG, JPG/JPEG, WEBP, GIF, or BMP up to 10 MB and 20 megapixels, with browser image decoding required before the file enters the workflow. This is an implementation assumption, not a production content policy.

**Stage acceptance**

- A valid image reaches preview without a page reload.
- Starting a run visibly enters processing and then produces a labelled local preview result.
- Cancellation preserves the preview and exposes retry.
- Invalid input and provider errors expose an actionable recovery path.
- Copy and save provide observable feedback and do not leave dead buttons.

**Dependencies**

- The workflow depends on `src/services/file-adapter.ts` for validation, object URL lifecycle, clipboard, and text download boundaries.
- It depends on `src/services/translation-provider.ts` for a cancellable, replaceable provider contract.


**Risks and follow-up**

- Real OCR and language behavior must be specified before replacing the local provider.
- A future product decision must define supported formats, size limits, source/target language selection, automatic language detection, and whether batch input is in scope.

### 2. Desktop UI

**Responsibilities**

- Vue 3 and TypeScript provide the reactive page and state boundary.
- Naive UI provides cards, buttons, alerts, progress, input, tags, empty state, and spacing primitives.
- The UI remains usable in a Vite browser session and in a Tauri WebView.
- The layout is responsive, keyboard-operable, and includes status announcements, visible focus treatment, and descriptive image text.

**Implemented files**

- `src/main.ts` installs Naive UI.
- `src/App.vue` renders the input and result panels, responsive layout, and all recovery actions.

**Stage acceptance**

- The input and output panels make the current state visible without relying on color alone.
- Processing uses a progress indicator and a cancellation button.
- Result text is readable, selectable, copyable, and saveable.
- Narrow windows stack the panels instead of clipping them.

**Dependencies**

- The UI depends on the application-level file and provider boundaries, not on Rust or `../PROJECT-TRNS` internals.
- `src/main.ts` must install Naive UI before the workflow components are rendered.


**Risks and follow-up**

- Before production use, test the layout with long file names, very large image dimensions, keyboard-only navigation, screen readers, and dark-mode requirements if a dark theme is added.

### 3. Tauri shell and boundary

**Responsibilities**

- Provide the desktop window and lifecycle.
- Keep native capabilities minimal and explicit.
- Preserve cancellation and error boundaries so a native failure returns to a recoverable UI state.
- Establish a safe path for image read and result write operations without exposing arbitrary shell or filesystem access.

**First-release implementation**

- `src-tauri/tauri.conf.json` provides a workspace-sized window with minimum dimensions.
- Image selection and preview use the standard Web File API and object URLs, which are available in the Tauri WebView and require no arbitrary filesystem permission.
- Result saving uses a Web Blob download. This is a working browser/Tauri-compatible path for the first release rather than an unhandled native save button.
- The existing Rust scaffold is not expanded with a broad command surface. The remaining native command/plugin scaffolding is not part of the translation workflow.

**Permission principles**

- Do not add shell, network, or unrestricted filesystem permissions for the first release.
- `src-tauri/capabilities/default.json` keeps the main window permission list empty because the current workflow does not call native commands or the opener plugin.
- Keep the restrictive `security.csp` in `src-tauri/tauri.conf.json`; never add `unsafe-eval`, and review any inline-style requirement before loosening the policy.
- Add a native file-dialog or save-dialog plugin only after its platform behavior and capability permissions are reviewed.
- If a future native command is introduced, pass validated, purpose-specific data and normalize errors before returning to Vue. Never accept an arbitrary path and write it without an explicit user-selected destination.



**Stage acceptance**

- The workflow operates in the Tauri WebView with standard file and download APIs.
- Cancellation aborts provider work and prevents stale results from replacing a newer state.
- Native capability expansion is documented and reviewed before configuration changes.

**Dependencies**

- Window configuration, CSP, and capabilities must remain aligned with the WebView-compatible file and download path.
- Future native file access must not bypass the provider/file adapter contracts or expose arbitrary filesystem operations.


**Risks and follow-up**

- Browser downloads may not match the desired native save-dialog UX on every platform. Validate a dialog plugin and narrow capability set as a separate task.
- If image paths, native decoding, or platform-specific file access become required, define and test a small Rust command boundary rather than leaking IPC details into the UI.

### 4. MCP and model integration

**Responsibilities**

- Keep the UI dependent on an application-level provider boundary.
- Translate validated application input into a future adapter call.
- Normalize success, cancellation, and recoverable failure before results reach Vue.
- Keep `../PROJECT-TRNS` as a candidate capability source, not an assumed public API.

**Implemented boundary**

- `src/services/translation-provider.ts` defines `TranslationProvider`, request/result types, cancellation behavior, and `LocalPreviewTranslationProvider`.
- The deterministic provider deliberately labels its output as a local preview and states that it is not production translation.
- No Vue code imports `../PROJECT-TRNS` internals or invents an MCP request/response schema.

**Candidate reference assets**

The sibling project has been identified as a future investigation source at:

- `../PROJECT-TRNS/src/translate.rs`
- `../PROJECT-TRNS/src/ocr_translate_text.rs`
- `../PROJECT-TRNS/src/generation.rs`
- `../PROJECT-TRNS/project-trns.toml`
- `../PROJECT-TRNS/tests/`
- `../PROJECT-TRNS/docs/`

These files demonstrate possible translation, OCR, generation, configuration, test, and documentation assets. They do not establish stable APIs for this desktop project. Before integration, inspect their public boundaries and tests, agree on ownership and transport, and add an adapter that preserves the local `TranslationProvider` contract.

**Stage acceptance**

- The plan names responsibilities and dependency direction without inventing a field-level MCP schema.
- A placeholder can be used for the first release, and the future replacement point is explicit.
- `../PROJECT-TRNS` is treated as a candidate capability source until its public boundary and tests are reviewed.

**Dependencies**

- The provider contract depends on validated application metadata today and must gain an opaque, content-bearing request only after the real OCR/model boundary is verified.
- The future adapter depends on the sibling project's public API, configuration, ownership, cancellation, and error behavior.

**Risks and follow-up**

- The current `TranslationRequest` is metadata-only by design; it is not a drop-in contract for real OCR/model input. Expand it in the future adapter phase rather than inventing MCP fields now.
- Keep the local provider available for deterministic offline workflow smoke tests while the real path is validated.


**Future replacement stages**

1. Verify the sibling project's callable boundary and supported image/language behavior.
2. Decide whether the adapter is in-process Rust, a local service, or an MCP-mediated capability.
3. Define the smallest protocol needed by the verified boundary, including cancellation and error normalization.
4. Implement the adapter behind `TranslationProvider` and keep the local provider available for offline smoke tests until the real path is proven.
5. Add integration tests for representative images, cancellation, malformed input, provider errors, and result persistence.

No field-level MCP request/response, event schema, transport choice, authentication, or production deployment contract is defined here.

### 5. Engineering plan and documentation

This document is the discoverable project-facing plan. `README.md` links here as the entry point for contributors. The implementation plan used for this pass is `.omc/plans/autopilot-impl.md`.

**Stage acceptance**

- The README links to this plan.
- The plan distinguishes shipped behavior from deferred capability.
- Every active topology component has responsibilities, dependencies, acceptance, risks, and follow-up work.
- The plan records the current greenfield scaffold evidence and candidate sibling assets.

**Dependencies**

- The plan depends on all four technical component plans being linked to their stage acceptance and risks.
- The README is the discoverability entry point and must keep its link to this document valid.

**Risks and follow-up**

- The exact target user, language behavior, production output shape, and `../PROJECT-TRNS` reuse unit remain open decisions.
- The document filename and docs index convention are currently `docs/ENGINEERING_PLAN.md` plus the README link; change them together if a later plan chooses a different entry point.

## Dependencies and sequence

1. Vue/TypeScript state and provider/file contracts.
2. Naive UI bootstrap and workflow presentation.
3. Tauri window sizing and WebView-compatible file boundaries.
4. Documentation and future integration decisions.

The UI depends on application boundaries, not the other way around. The current data flow is:

```text
Vue + Naive UI
  -> file/provider application boundaries
  -> browser/Tauri WebView file APIs and local preview provider
  -> recoverable result, cancellation, or error
```

A future model path can replace only the provider adapter:

```text
Vue UI
  -> TranslationProvider adapter
  -> verified MCP/model capability
  -> result normalization
  -> Vue result/copy/save state
```

## Explicit non-goals

- No OCR engine, model inference, production MCP client/server, or direct invocation of `../PROJECT-TRNS` internals.
- No field-level MCP request/response/event schema or complete protocol specification.
- No arbitrary filesystem writes, shell commands, network permissions, authentication, telemetry, updater, installer, or release pipeline design.
- No batch translation, multi-window workflow, source/target language selector, automatic language detection, or production content policy.
- No claim that the local preview text is a translation.

## Open decisions and validation tasks

| Decision | Current assumption | Required follow-up |
|---|---|---|
| Target user | Desktop user needing a single image translation workspace | Validate with representative users and accessibility testing |
| Image formats and size | PNG/JPG/JPEG/WEBP/GIF/BMP, max 10 MB for first release | Confirm model/OCR constraints and user expectations |
| Language behavior | No language selector; local preview only | Define source/target language and automatic detection behavior with the real provider |
| Save result shape | Text download with a `-preview.txt` suffix | Decide whether production output is text, rendered image, or both |
| Native file dialogs | Web File API and Blob download in first release | Evaluate Tauri dialog plugin and least-privilege capability set |
| PROJECT-TRNS reuse | Candidate Rust/OCR/generation assets only | Review public APIs, tests, configuration, and ownership before adapter work |
| Plan entry point | `docs/ENGINEERING_PLAN.md` linked by `README.md` | Keep the link current as topology or release scope changes |

## Verification handoff

This implementation pass was verified with `bun run build`, `cargo test --manifest-path src-tauri/Cargo.toml`, `CI=false bun run tauri build --no-bundle`, and browser smoke tests covering valid/invalid image selection, decode failure recovery, preview, processing, cancellation, retry, result copy, and result save. Formatters and linters were not run; before production model integration, add bounded pre-decode dimension checks, review whether `style-src 'unsafe-inline'` can be removed, and define explicit command permissions before adding native Tauri commands.
