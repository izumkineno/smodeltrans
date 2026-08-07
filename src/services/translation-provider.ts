import { invoke } from "@tauri-apps/api/core";

export interface TranslationRequest {
  file: File;
  targetLanguage: string;
  requestId?: string;
}

export interface TranslationResult {
  text: string;
  markdown: string;
  annotatedImageDataUrl: string;
  providerLabel: string;
  isTranslated: boolean;
  durationMs: number;
}

export interface TranslationProgress {
  requestId: string;
  progress: number;
  stage: string;
}

export interface BackendStatus {
  ready: boolean;
  device: string;
  detectorModelDir: string;
  recognizerModelDir: string;
  hyModel: string;
  fontPath: string | null;
  translatorLoaded: boolean;
  idleUnloadMinutes: number;
  message: string;
}

export interface BackendSettingsUpdate {
  detectorModelDir: string;
  recognizerModelDir: string;
  hyModel: string;
  idleUnloadMinutes: number;
}

export interface TranslationProvider {
  translate(request: TranslationRequest, signal: AbortSignal): Promise<TranslationResult>;
}

interface TauriTranslationResponse {
  text: string;
  markdown: string;
  annotatedImageDataUrl: string;
  providerLabel: string;
  isTranslated: boolean;
  durationMs: number;
}

interface BackendErrorPayload {
  code?: string;
  message?: string;
}

type InvokeFn = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

let requestCounter = 0;

function cancellationError(): Error {
  const error = new Error("Translation cancelled");
  error.name = "AbortError";
  return error;
}

function encodeBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return (window.btoa ?? globalThis.btoa)(binary);
}

function nextRequestId(): string {
  requestCounter += 1;
  return `ui-${Date.now()}-${requestCounter}`;
}

export function createTranslationRequestId(): string {
  return nextRequestId();
}

function abortable<T>(promise: Promise<T>, signal: AbortSignal, onAbort: () => void): Promise<T> {
  if (signal.aborted) {
    onAbort();
    return Promise.reject(cancellationError());
  }

  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const cleanup = () => signal.removeEventListener("abort", handleAbort);
    const handleAbort = () => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      onAbort();
      reject(cancellationError());
    };

    signal.addEventListener("abort", handleAbort, { once: true });
    void promise.then(
      (value) => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        resolve(value);
      },
      (error: unknown) => {
        if (settled) {
          return;
        }
        settled = true;
        cleanup();
        reject(error);
      },
    );
  });
}

function normalizeBackendError(error: unknown): Error {
  if (error instanceof Error) {
    return error;
  }
  const payload = (typeof error === "object" && error !== null ? error : {}) as BackendErrorPayload;
  if (payload.code === "cancelled") {
    return cancellationError();
  }
  const normalized = new Error(payload.message || "Candle 后端未能完成图片翻译");
  normalized.name = payload.code ? `BackendError:${payload.code}` : "BackendError";
  return normalized;
}

export async function getBackendStatus(invokeFn: InvokeFn = invoke): Promise<BackendStatus> {
  return invokeFn<BackendStatus>("get_backend_status");
}

export async function updateBackendSettings(
  settings: BackendSettingsUpdate,
  invokeFn: InvokeFn = invoke,
): Promise<BackendStatus> {
  return invokeFn<BackendStatus>("update_backend_settings", { request: settings });
}

export class TauriTranslationProvider implements TranslationProvider {
  constructor(private readonly invokeFn: InvokeFn = invoke) {}

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<TranslationResult> {
    if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
      throw new Error("图片翻译后端只在 Tauri 桌面端可用。");
    }
    if (signal.aborted) {
      throw cancellationError();
    }

    const bytes = new Uint8Array(await request.file.arrayBuffer());
    if (signal.aborted) {
      throw cancellationError();
    }
    const requestId = request.requestId ?? nextRequestId();
    const cancelBackendRun = () => {
      void this.invokeFn("cancel_translation", {
        request: {
          requestId,
        },
      }).catch(() => undefined);
    };
    const responsePromise = this.invokeFn<TauriTranslationResponse>("translate_image", {
      request: {
        imageBase64: encodeBase64(bytes),
        fileName: request.file.name,
        targetLanguage: request.targetLanguage,
        requestId,
      },
    }).catch((error: unknown) => {
      throw normalizeBackendError(error);
    });
    const response = await abortable(responsePromise, signal, cancelBackendRun);
    return {
      text: response.text,
      markdown: response.markdown,
      annotatedImageDataUrl: response.annotatedImageDataUrl,
      providerLabel: response.providerLabel,
      isTranslated: response.isTranslated,
      durationMs: response.durationMs,
    };
  }
}

export const translationProvider: TranslationProvider = new TauriTranslationProvider();

export function createTauriTranslationProvider(invokeFn: InvokeFn): TranslationProvider {
  return new TauriTranslationProvider(invokeFn);
}

export function isTranslationCancellation(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
