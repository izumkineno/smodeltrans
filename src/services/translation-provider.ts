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

export type DeviceKind = "cpu" | "cuda";

export interface BackendGenerationSettings {
  maxNewTokens: number;
  sampling: boolean;
  temperature: number;
  topK: number;
  topP: number;
  seed: string | null;
  repetitionPenalty: number;
  frequencyPenalty: number;
  stopTokens: number[];
  stopStrings: string[];
}

export interface BackendMemorySettings {
  enabled: boolean;
  maxTokens: number;
  maxTurns: number;
}

export interface BackendPromptSettings {
  system: string;
  user: string;
}

export interface BackendStatus {
  ready: boolean;
  device: string;
  detectorModelDir: string;
  recognizerModelDir: string;
  detectorVariant: string | null;
  recognizerVariant: string | null;
  hyModel: string;
  fontPath: string | null;
  targetLanguage: string;
  regionParallelism: number;
  translationBatchSize: number;
  translatorLoaded: boolean;
  idleUnloadMinutes: number;
  generation: BackendGenerationSettings;
  memory: BackendMemorySettings;
  prompt: BackendPromptSettings;
  message: string;
}

export type ModelTarget = "ocr" | "translator";
export type ModelAction = "load" | "unload";

export interface ModelRuntimeEvent {
  timestampMs: number;
  operation: string;
  durationMs: number;
  success: boolean;
  message: string;
}

export interface ModelRuntimeStatus {
  backend: BackendStatus;
  ocrLoaded: boolean;
  translatorLoaded: boolean;
  busy: boolean;
  idleForMs: number;
  requestCount: number;
  succeededRequests: number;
  failedRequests: number;
  averageDurationMs: number;
  lastDurationMs: number | null;
  lastOperation: string | null;
  recentEvents: ModelRuntimeEvent[];
}

export interface BackendSettingsUpdate {
  detectorModelDir: string;
  recognizerModelDir: string;
  hyModel: string;
  fontPath: string | null;
  targetLanguage: string;
  device: DeviceKind;
  regionParallelism: number;
  translationBatchSize: number;
  idleUnloadMinutes: number;
  generation: BackendGenerationSettings;
  memory: BackendMemorySettings;
  prompt: BackendPromptSettings;
}

export interface TranslationModelOption {
  name: string;
  path: string;
}

export interface OcrModelOption {
  name: string;
  detectorDir: string;
  recognizerDir: string;
  variant: string | null;
}

export interface FontModelOption {
  name: string;
  path: string | null;
}

export interface ModelCatalogOptions {
  translation: TranslationModelOption[];
  ocr: OcrModelOption[];
  fonts: FontModelOption[];
}

export interface ModelCatalogUpdate {
  translation: Array<{ name: string; path: string }>;
  ocr: Array<{ name: string; detectorDir: string; recognizerDir: string }>;
  fonts: Array<{ name: string; path: string }>;
}

export interface TranslationProvider {
  translate(request: TranslationRequest, signal: AbortSignal): Promise<TranslationResult>;
}

export interface TextTranslationRequest {
  text: string;
  targetLanguage: string;
  requestId?: string;
}

export interface TextTranslationResult {
  text: string;
  providerLabel: string;
  durationMs: number;
}

export interface TextTranslationProvider {
  translate(request: TextTranslationRequest, signal: AbortSignal): Promise<TextTranslationResult>;
}

export interface OcrRequest {
  file: File;
  requestId?: string;
}

export type OcrQuad = [[number, number], [number, number], [number, number], [number, number]];

export interface OcrCharacterBox {
  order: number;
  quad: OcrQuad;
  recognizedText: string;
}

export interface OcrRegion {
  order: number;
  quad: OcrQuad;
  recognizedText: string;
  charBoxes: OcrCharacterBox[];
}

export interface OcrResult {
  text: string;
  markdown: string;
  annotatedImageDataUrl: string;
  providerLabel: string;
  durationMs: number;
  imageWidth: number;
  imageHeight: number;
  regions: OcrRegion[];
}

export interface OcrProvider {
  recognize(request: OcrRequest, signal: AbortSignal): Promise<OcrResult>;
}

interface TauriTranslationResponse {
  text: string;
  markdown: string;
  annotatedImageDataUrl: string;
  providerLabel: string;
  isTranslated: boolean;
  durationMs: number;
}
interface TauriTextTranslationResponse {
  text: string;
  providerLabel: string;
  durationMs: number;
}

interface TauriOcrResponse {
  text: string;
  markdown: string;
  annotatedImageDataUrl: string;
  providerLabel: string;
  durationMs: number;
  imageWidth: number;
  imageHeight: number;
  regions: OcrRegion[];
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

function normalizeBackendError(error: unknown, fallbackMessage = "Candle 后端未能完成图片翻译"): Error {
  if (error instanceof Error) {
    return error;
  }
  const payload = (typeof error === "object" && error !== null ? error : {}) as BackendErrorPayload;
  if (payload.code === "cancelled") {
    return cancellationError();
  }
  const normalized = new Error(payload.message || fallbackMessage);
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

export async function getModelCatalog(
  invokeFn: InvokeFn = invoke,
): Promise<ModelCatalogOptions> {
  return invokeFn<ModelCatalogOptions>("list_model_catalog");
}

export async function saveModelCatalog(
  catalog: ModelCatalogUpdate,
  invokeFn: InvokeFn = invoke,
): Promise<BackendStatus> {
  return invokeFn<BackendStatus>("save_model_catalog", { request: catalog });
}

export async function getModelRuntimeStatus(
  invokeFn: InvokeFn = invoke,
): Promise<ModelRuntimeStatus> {
  return invokeFn<ModelRuntimeStatus>("get_model_runtime_status");
}

export async function controlModel(
  model: ModelTarget,
  action: ModelAction,
  invokeFn: InvokeFn = invoke,
): Promise<ModelRuntimeStatus> {
  return invokeFn<ModelRuntimeStatus>("control_model", {
    request: { model, action },
  });
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
export class TauriTextTranslationProvider implements TextTranslationProvider {
  constructor(private readonly invokeFn: InvokeFn = invoke) {}

  async translate(request: TextTranslationRequest, signal: AbortSignal): Promise<TextTranslationResult> {
    if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
      throw new Error("文本翻译后端只在 Tauri 桌面端可用。");
    }
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
    const responsePromise = this.invokeFn<TauriTextTranslationResponse>("translate_text", {
      request: {
        text: request.text,
        targetLanguage: request.targetLanguage,
        requestId,
      },
    }).catch((error: unknown) => {
      throw normalizeBackendError(error, "Candle 后端未能完成文本翻译");
    });
    const response = await abortable(responsePromise, signal, cancelBackendRun);
    return {
      text: response.text,
      providerLabel: response.providerLabel,
      durationMs: response.durationMs,
    };
  }
}

export class TauriOcrProvider implements OcrProvider {
  constructor(private readonly invokeFn: InvokeFn = invoke) {}

  async recognize(request: OcrRequest, signal: AbortSignal): Promise<OcrResult> {
    if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
      throw new Error("OCR 后端只在 Tauri 桌面端可用。");
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
    const responsePromise = this.invokeFn<TauriOcrResponse>("ocr_image", {
      request: {
        imageBase64: encodeBase64(bytes),
        fileName: request.file.name,
        requestId,
      },
    }).catch((error: unknown) => {
      throw normalizeBackendError(error, "Candle 后端未能完成 OCR");
    });
    const response = await abortable(responsePromise, signal, cancelBackendRun);
    return {
      text: response.text,
      markdown: response.markdown,
      annotatedImageDataUrl: response.annotatedImageDataUrl,
      providerLabel: response.providerLabel,
      durationMs: response.durationMs,
      imageWidth: response.imageWidth,
      imageHeight: response.imageHeight,
      regions: response.regions,
    };
  }
}


export const translationProvider: TranslationProvider = new TauriTranslationProvider();
export const textTranslationProvider: TextTranslationProvider = new TauriTextTranslationProvider();

export const ocrProvider: OcrProvider = new TauriOcrProvider();


export function createTauriTranslationProvider(invokeFn: InvokeFn): TranslationProvider {
  return new TauriTranslationProvider(invokeFn);
}
export function createTauriTextTranslationProvider(invokeFn: InvokeFn): TextTranslationProvider {
  return new TauriTextTranslationProvider(invokeFn);
}

export function createTauriOcrProvider(invokeFn: InvokeFn): OcrProvider {
  return new TauriOcrProvider(invokeFn);
}


export function isTranslationCancellation(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}
