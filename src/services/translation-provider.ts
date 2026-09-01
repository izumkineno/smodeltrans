import { invoke } from "@tauri-apps/api/core";

const LOG_PREFIX = " [translation-provider]";

// Helpers wrap console.* so forwarding via installConsoleForwarding captures them; keep prefix uniform
function logInfo(message: string, data?: Record<string, unknown>) {
  if (data) {
    console.info(`${LOG_PREFIX} ${message}`, data);
  } else {
    console.info(`${LOG_PREFIX} ${message}`);
  }
}

function logDebug(message: string, data?: Record<string, unknown>) {
  if (data) {
    console.debug(`${LOG_PREFIX} ${message}`, data);
  } else {
    console.debug(`${LOG_PREFIX} ${message}`);
  }
}

function logWarn(message: string, data?: Record<string, unknown>) {
  if (data) {
    console.warn(`${LOG_PREFIX} ${message}`, data);
  } else {
    console.warn(`${LOG_PREFIX} ${message}`);
  }
}

function logError(message: string, data?: Record<string, unknown>) {
  if (data) {
    console.error(`${LOG_PREFIX} ${message}`, data);
  } else {
    console.error(`${LOG_PREFIX} ${message}`);
  }
}

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
export const DEFAULT_IDLE_UNLOAD_SECONDS = 1_800;
export const IDLE_UNLOAD_SECONDS_MIN = 0;
export const IDLE_UNLOAD_SECONDS_MAX = 86_400;

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
  template: string;
  // 兼容旧持久化：`prompt`/`system`/`user` 仅用于读取旧数据
  prompt?: string;
  system?: string;
  user?: string;
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
  idleUnloadSeconds: number;
  generation: BackendGenerationSettings;
  memory: BackendMemorySettings;
  prompt: BackendPromptSettings;
  message: string;
}

export type ModelTarget = "ocr" | "translator";
export type ModelAction = "load" | "unload";


export interface ModelRuntimeStatus {
  backend: BackendStatus;
  ocrLoaded: boolean;
  translatorLoaded: boolean;
  busy: boolean;
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
  idleUnloadSeconds: number;
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
    logWarn("abortable aborted before start, invoking cancelBackendRun");
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
      logWarn("abort signal received, invoking cancelBackendRun");
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
    console.error(`${LOG_PREFIX} normalizeBackendError passthrough Error`, {
      name: error.name,
      message: error.message,
      fallbackMessage,
    });
    return error;
  }
  const payload = (typeof error === "object" && error !== null ? error : {}) as BackendErrorPayload;
  if (payload.code === "cancelled") {
    console.warn(`${LOG_PREFIX} normalizeBackendError detected cancelled payload, converting to AbortError`, {
      code: payload.code,
    });
    return cancellationError();
  }
  const normalized = new Error(payload.message || fallbackMessage);
  normalized.name = payload.code ? `BackendError:${payload.code}` : "BackendError";
  console.error(`${LOG_PREFIX} normalizeBackendError`, {
    code: payload.code ?? "unknown",
    message: normalized.message,
    name: normalized.name,
    fallbackMessage,
  });
  return normalized;
}

export async function getBackendStatus(invokeFn: InvokeFn = invoke): Promise<BackendStatus> {
  logInfo("getBackendStatus start");
  const start = Date.now();
  try {
    const status = await invokeFn<BackendStatus>("get_backend_status");
    console.info(`${LOG_PREFIX} getBackendStatus success`, {
      ready: status.ready,
      device: status.device,
      targetLanguage: status.targetLanguage,
      durationMs: Date.now() - start,
    });
    return status;
  } catch (error) {
    console.error(`${LOG_PREFIX} getBackendStatus failed`, {
      error: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - start,
    });
    throw normalizeBackendError(error, "获取后端状态失败");
  }
}

export async function updateBackendSettings(
  settings: BackendSettingsUpdate,
  invokeFn: InvokeFn = invoke,
): Promise<BackendStatus> {
  console.info(`${LOG_PREFIX} updateBackendSettings start`, {
    targetLanguage: settings.targetLanguage,
    device: settings.device,
    hyModel: settings.hyModel,
  });
  const start = Date.now();
  try {
    const status = await invokeFn<BackendStatus>("update_backend_settings", { request: settings });
    console.info(`${LOG_PREFIX} updateBackendSettings success`, {
      durationMs: Date.now() - start,
      ready: status.ready,
    });
    return status;
  } catch (error) {
    console.error(`${LOG_PREFIX} updateBackendSettings failed`, {
      error: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - start,
    });
    throw normalizeBackendError(error, "更新后端设置失败");
  }
}

export async function getModelCatalog(
  invokeFn: InvokeFn = invoke,
): Promise<ModelCatalogOptions> {
  console.info(`${LOG_PREFIX} getModelCatalog start`);
  try {
    const catalog = await invokeFn<ModelCatalogOptions>("list_model_catalog");
    console.info(`${LOG_PREFIX} getModelCatalog success`, {
      translationCount: catalog.translation.length,
      ocrCount: catalog.ocr.length,
      fontsCount: catalog.fonts.length,
    });
    return catalog;
  } catch (error) {
    console.error(`${LOG_PREFIX} getModelCatalog failed`, {
      error: error instanceof Error ? error.message : String(error),
    });
    throw normalizeBackendError(error, "获取模型目录失败");
  }
}

export async function saveModelCatalog(
  catalog: ModelCatalogUpdate,
  invokeFn: InvokeFn = invoke,
): Promise<BackendStatus> {
  console.info(`${LOG_PREFIX} saveModelCatalog start`, {
    translationCount: catalog.translation.length,
    ocrCount: catalog.ocr.length,
    fontsCount: catalog.fonts.length,
  });
  try {
    const status = await invokeFn<BackendStatus>("save_model_catalog", { request: catalog });
    console.info(`${LOG_PREFIX} saveModelCatalog success ready=${status.ready}`);
    return status;
  } catch (error) {
    console.error(`${LOG_PREFIX} saveModelCatalog failed`, {
      error: error instanceof Error ? error.message : String(error),
    });
    throw normalizeBackendError(error, "保存模型目录失败");
  }
}

export async function getModelRuntimeStatus(
  invokeFn: InvokeFn = invoke,
): Promise<ModelRuntimeStatus> {
  console.info(`${LOG_PREFIX} getModelRuntimeStatus start`);
  const start = Date.now();
  try {
    const status = await invokeFn<ModelRuntimeStatus>("get_model_runtime_status");
    console.info(`${LOG_PREFIX} getModelRuntimeStatus success`, {
      ocrLoaded: status.ocrLoaded,
      translatorLoaded: status.translatorLoaded,
      busy: status.busy,
      durationMs: Date.now() - start,
    });
    return status;
  } catch (error) {
    console.error(`${LOG_PREFIX} getModelRuntimeStatus failed`, {
      error: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - start,
    });
    throw normalizeBackendError(error, "获取模型运行时状态失败");
  }
}

export async function controlModel(
  model: ModelTarget,
  action: ModelAction,
  invokeFn: InvokeFn = invoke,
): Promise<ModelRuntimeStatus> {
  console.info(`${LOG_PREFIX} controlModel start`, { model, action });
  const start = Date.now();
  try {
    const status = await invokeFn<ModelRuntimeStatus>("control_model", {
      request: { model, action },
    });
    console.info(`${LOG_PREFIX} controlModel success`, {
      model,
      action,
      ocrLoaded: status.ocrLoaded,
      translatorLoaded: status.translatorLoaded,
      durationMs: Date.now() - start,
    });
    return status;
  } catch (error) {
    console.error(`${LOG_PREFIX} controlModel failed`, {
      model,
      action,
      error: error instanceof Error ? error.message : String(error),
      durationMs: Date.now() - start,
    });
    throw normalizeBackendError(error, "模型控制失败");
  }
}

export class TauriTranslationProvider implements TranslationProvider {
  constructor(private readonly invokeFn: InvokeFn = invoke) {}

  async translate(request: TranslationRequest, signal: AbortSignal): Promise<TranslationResult> {
    const requestId = request.requestId ?? nextRequestId();
    const start = Date.now();
    console.info(`${LOG_PREFIX} TauriTranslationProvider.translate start`, {
      requestId,
      fileName: request.file.name,
      fileSize: request.file.size,
      fileType: request.file.type,
      targetLanguage: request.targetLanguage,
    });
    if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
      logError("TauriTranslationProvider.translate not in Tauri runtime", { requestId });
      throw new Error("图片翻译后端只在 Tauri 桌面端可用。");
    }
    if (signal.aborted) {
      console.warn(`${LOG_PREFIX} TauriTranslationProvider.translate aborted before read`, {
        requestId,
        fileName: request.file.name,
        targetLanguage: request.targetLanguage,
      });
      throw cancellationError();
    }

    const bytes = new Uint8Array(await request.file.arrayBuffer());
    console.debug(`${LOG_PREFIX} TauriTranslationProvider.translate bytes loaded`, {
      requestId,
      bytesLength: bytes.length,
      fileName: request.file.name,
    });
    if (signal.aborted) {
      console.warn(`${LOG_PREFIX} TauriTranslationProvider.translate aborted after bytes loaded`, {
        requestId,
        bytesLength: bytes.length,
      });
      throw cancellationError();
    }
    const cancelBackendRun = () => {
      console.debug(`${LOG_PREFIX} TauriTranslationProvider.translate cancelBackendRun`, {
        requestId,
        targetLanguage: request.targetLanguage,
        fileName: request.file.name,
      });
      void this.invokeFn("cancel_translation", {
        request: {
          requestId,
        },
      }).catch(() => undefined);
    };
    const encoded = encodeBase64(bytes);
    console.debug(`${LOG_PREFIX} TauriTranslationProvider.translate base64 encoded`, {
      requestId,
      bytesLength: bytes.length,
      base64Length: encoded.length,
      targetLanguage: request.targetLanguage,
    });
    const responsePromise = this.invokeFn<TauriTranslationResponse>("translate_image", {
      request: {
        imageBase64: encoded,
        fileName: request.file.name,
        targetLanguage: request.targetLanguage,
        requestId,
      },
    }).catch((error: unknown) => {
      console.error(`${LOG_PREFIX} TauriTranslationProvider.translate invoke failed`, {
        requestId,
        fileName: request.file.name,
        targetLanguage: request.targetLanguage,
        error: error instanceof Error ? error.message : String(error),
      });
      throw normalizeBackendError(error);
    });
    try {
      const response = await abortable(responsePromise, signal, cancelBackendRun);
      console.info(`${LOG_PREFIX} TauriTranslationProvider.translate success`, {
        requestId,
        fileName: request.file.name,
        targetLanguage: request.targetLanguage,
        providerLabel: response.providerLabel,
        durationMs: response.durationMs,
        totalDurationMs: Date.now() - start,
        isTranslated: response.isTranslated,
        textLength: response.text.length,
      });
      console.debug(`${LOG_PREFIX} TauriTranslationProvider.translate response detail`, {
        requestId,
        markdownLength: response.markdown.length,
        hasAnnotatedImage: !!response.annotatedImageDataUrl,
      });
      return {
        text: response.text,
        markdown: response.markdown,
        annotatedImageDataUrl: response.annotatedImageDataUrl,
        providerLabel: response.providerLabel,
        isTranslated: response.isTranslated,
        durationMs: response.durationMs,
      };
    } catch (error) {
      if (isTranslationCancellation(error)) {
        console.warn(`${LOG_PREFIX} TauriTranslationProvider.translate cancelled`, {
          requestId,
          fileName: request.file.name,
          targetLanguage: request.targetLanguage,
          durationMs: Date.now() - start,
        });
      } else {
        console.error(`${LOG_PREFIX} TauriTranslationProvider.translate failed`, {
          requestId,
          fileName: request.file.name,
          targetLanguage: request.targetLanguage,
          error: error instanceof Error ? error.message : String(error),
          durationMs: Date.now() - start,
        });
      }
      throw error;
    }
  }
}
export class TauriTextTranslationProvider implements TextTranslationProvider {
  constructor(private readonly invokeFn: InvokeFn = invoke) {}

  async translate(request: TextTranslationRequest, signal: AbortSignal): Promise<TextTranslationResult> {
    const requestId = request.requestId ?? nextRequestId();
    const start = Date.now();
    console.info(`${LOG_PREFIX} TauriTextTranslationProvider.translate start`, {
      requestId,
      targetLanguage: request.targetLanguage,
      textLength: request.text.length,
      textPreview: request.text.slice(0, 80),
    });
    console.debug(`${LOG_PREFIX} TauriTextTranslationProvider.translate payload`, {
      requestId,
      textLen: request.text.length,
      targetLanguage: request.targetLanguage,
    });
    if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
      logError("TauriTextTranslationProvider.translate not in Tauri runtime", { requestId });
      throw new Error("文本翻译后端只在 Tauri 桌面端可用。");
    }
    if (signal.aborted) {
      console.warn(`${LOG_PREFIX} TauriTextTranslationProvider.translate aborted before start`, {
        requestId,
        targetLanguage: request.targetLanguage,
      });
      throw cancellationError();
    }

    const cancelBackendRun = () => {
      console.debug(`${LOG_PREFIX} TauriTextTranslationProvider.translate cancelBackendRun`, {
        requestId,
        targetLanguage: request.targetLanguage,
      });
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
      console.error(`${LOG_PREFIX} TauriTextTranslationProvider.translate invoke failed`, {
        requestId,
        targetLanguage: request.targetLanguage,
        textLength: request.text.length,
        error: error instanceof Error ? error.message : String(error),
      });
      throw normalizeBackendError(error, "Candle 后端未能完成文本翻译");
    });
    try {
      const response = await abortable(responsePromise, signal, cancelBackendRun);
      console.info(`${LOG_PREFIX} TauriTextTranslationProvider.translate success`, {
        requestId,
        targetLanguage: request.targetLanguage,
        providerLabel: response.providerLabel,
        durationMs: response.durationMs,
        totalDurationMs: Date.now() - start,
        resultTextLength: response.text.length,
      });
      return {
        text: response.text,
        providerLabel: response.providerLabel,
        durationMs: response.durationMs,
      };
    } catch (error) {
      if (isTranslationCancellation(error)) {
        console.warn(`${LOG_PREFIX} TauriTextTranslationProvider.translate cancelled`, {
          requestId,
          targetLanguage: request.targetLanguage,
          durationMs: Date.now() - start,
        });
      } else {
        console.error(`${LOG_PREFIX} TauriTextTranslationProvider.translate failed`, {
          requestId,
          targetLanguage: request.targetLanguage,
          error: error instanceof Error ? error.message : String(error),
          durationMs: Date.now() - start,
        });
      }
      throw error;
    }
  }
}

export class TauriOcrProvider implements OcrProvider {
  constructor(private readonly invokeFn: InvokeFn = invoke) {}

  async recognize(request: OcrRequest, signal: AbortSignal): Promise<OcrResult> {
    const requestId = request.requestId ?? nextRequestId();
    const start = Date.now();
    console.info(`${LOG_PREFIX} TauriOcrProvider.recognize start`, {
      requestId,
      fileName: request.file.name,
      fileSize: request.file.size,
      fileType: request.file.type,
    });
    if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
      logError("TauriOcrProvider.recognize not in Tauri runtime", { requestId });
      throw new Error("OCR 后端只在 Tauri 桌面端可用。");
    }
    if (signal.aborted) {
      console.warn(`${LOG_PREFIX} TauriOcrProvider.recognize aborted before read`, {
        requestId,
        fileName: request.file.name,
      });
      throw cancellationError();
    }

    const bytes = new Uint8Array(await request.file.arrayBuffer());
    console.debug(`${LOG_PREFIX} TauriOcrProvider.recognize bytes loaded`, {
      requestId,
      bytesLength: bytes.length,
      fileName: request.file.name,
    });
    if (signal.aborted) {
      console.warn(`${LOG_PREFIX} TauriOcrProvider.recognize aborted after bytes loaded`, {
        requestId,
        bytesLength: bytes.length,
      });
      throw cancellationError();
    }
    const cancelBackendRun = () => {
      console.debug(`${LOG_PREFIX} TauriOcrProvider.recognize cancelBackendRun`, {
        requestId,
        fileName: request.file.name,
      });
      void this.invokeFn("cancel_translation", {
        request: {
          requestId,
        },
      }).catch(() => undefined);
    };
    const encoded = encodeBase64(bytes);
    console.debug(`${LOG_PREFIX} TauriOcrProvider.recognize base64 encoded`, {
      requestId,
      bytesLength: bytes.length,
      base64Length: encoded.length,
    });
    const responsePromise = this.invokeFn<TauriOcrResponse>("ocr_image", {
      request: {
        imageBase64: encoded,
        fileName: request.file.name,
        requestId,
      },
    }).catch((error: unknown) => {
      console.error(`${LOG_PREFIX} TauriOcrProvider.recognize invoke failed`, {
        requestId,
        fileName: request.file.name,
        error: error instanceof Error ? error.message : String(error),
      });
      throw normalizeBackendError(error, "Candle 后端未能完成 OCR");
    });
    try {
      const response = await abortable(responsePromise, signal, cancelBackendRun);
      console.info(`${LOG_PREFIX} TauriOcrProvider.recognize success`, {
        requestId,
        fileName: request.file.name,
        providerLabel: response.providerLabel,
        durationMs: response.durationMs,
        totalDurationMs: Date.now() - start,
        imageWidth: response.imageWidth,
        imageHeight: response.imageHeight,
        regions: response.regions.length,
        textLength: response.text.length,
      });
      console.debug(`${LOG_PREFIX} TauriOcrProvider.recognize response detail`, {
        requestId,
        markdownLength: response.markdown.length,
        hasAnnotatedImage: !!response.annotatedImageDataUrl,
      });
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
    } catch (error) {
      if (isTranslationCancellation(error)) {
        console.warn(`${LOG_PREFIX} TauriOcrProvider.recognize cancelled`, {
          requestId,
          fileName: request.file.name,
          durationMs: Date.now() - start,
        });
      } else {
        console.error(`${LOG_PREFIX} TauriOcrProvider.recognize failed`, {
          requestId,
          fileName: request.file.name,
          error: error instanceof Error ? error.message : String(error),
          durationMs: Date.now() - start,
        });
      }
      throw error;
    }
  }
}


export const translationProvider: TranslationProvider = new TauriTranslationProvider();
export const textTranslationProvider: TextTranslationProvider = new TauriTextTranslationProvider();

export const ocrProvider: OcrProvider = new TauriOcrProvider();


export function createTauriTranslationProvider(invokeFn: InvokeFn): TranslationProvider {
  logInfo("createTauriTranslationProvider called");
  return new TauriTranslationProvider(invokeFn);
}
export function createTauriTextTranslationProvider(invokeFn: InvokeFn): TextTranslationProvider {
  logDebug("createTauriTextTranslationProvider called");
  return new TauriTextTranslationProvider(invokeFn);
}

export function createTauriOcrProvider(invokeFn: InvokeFn): OcrProvider {
  logDebug("createTauriOcrProvider called");
  return new TauriOcrProvider(invokeFn);
}


export function isTranslationCancellation(error: unknown): boolean {
  const isCancel = error instanceof Error && error.name === "AbortError";
  if (isCancel) {
    logDebug("isTranslationCancellation true", { message: (error as Error).message });
  }
  return isCancel;
}
