import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

const DL_LOG_PREFIX = " [model-download-provider]";

/**
 * ModelScope 作为默认下载源。
 * 前端通过此服务屏蔽底层的 ModelScope HTTP / SDK 细节，
 * 后端 Rust 侧对接 ModelScope 的 resolve 接口进行流式下载。
 */
export type DownloadSource = "modelscope" | "huggingface";

export interface DownloadFileSpec {
  repoId: string;
  file: string;
  dest: string;
}

export interface DownloadableModel {
  id: string;
  name: string;
  description: string;
  repoId: string;
  files: string[];
  fileSpecs?: DownloadFileSpec[];
  sizeText: string;
  kind: "translation" | "ocr" | "font";
  ocrVariant?: string;
  recommended?: boolean;
}

export type DownloadStatus =
  | "idle"
  | "downloading"
  | "completed"
  | "error"
  | "cancelled";

export interface DownloadTaskState {
  modelId: string;
  source: DownloadSource;
  status: DownloadStatus;
  progress: number; // 0-100
  downloadedBytes: number;
  totalBytes: number;
  message?: string;
}

export interface ModelDownloadProgressEvent {
  modelId: string;
  source: DownloadSource;
  progress: number;
  downloadedBytes: number;
  totalBytes: number;
  status: DownloadStatus;
  message?: string;
}

type InvokeFn = <T>(
  command: string,
  args?: Record<string, unknown>,
) => Promise<T>;

/** ModelScope 默认源的推荐清单 */
export const MODELSCOPE_DOWNLOADABLE_MODELS: DownloadableModel[] = [
  {
    id: "hy-mt2-1.8b-q4",
    name: "Hy-MT2 1.8B Q4_K_M",
    description: "多语言翻译核心，ModelScope: Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
    repoId: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
    files: ["Hy-MT2-1.8B-Q4_K_M.gguf"],
    sizeText: "~1.1 GB",
    kind: "translation",
    recommended: true,
  },
  {
    id: "hy-mt2-1.8b-q6k",
    name: "Hy-MT2 1.8B Q6_K",
    description: "多语言翻译核心，ModelScope: Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
    repoId: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
    files: ["Hy-MT2-1.8B-Q6_K.gguf"],
    sizeText: "~1.5 GB",
    kind: "translation",
  },
  {
    id: "hy-mt2-1.8b-q8",
    name: "Hy-MT2 1.8B Q8_0",
    description: "多语言翻译核心，ModelScope: Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
    repoId: "Tencent-Hunyuan/Hy-MT2-1.8B-GGUF",
    files: ["Hy-MT2-1.8B-Q8_0.gguf"],
    sizeText: "~1.9 GB",
    kind: "translation",
  },
  {
    id: "ppocr-v5-mobile",
    name: "PP-OCR v5 mobile",
    description:
      "轻量检测+识别，适合实时字幕，ModelScope: PaddlePaddle/PP-OCRv5",
    repoId: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
    files: [
      "mobile_det/model.safetensors",
      "mobile_det/config.json",
      "mobile_det/preprocessor_config.json",
      "mobile_det/inference.yml",
      "mobile_rec/model.safetensors",
      "mobile_rec/config.json",
      "mobile_rec/preprocessor_config.json",
      "mobile_rec/inference.yml",
    ],
    fileSpecs: [
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
        file: "model.safetensors",
        dest: "mobile_det/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
        file: "config.json",
        dest: "mobile_det/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
        file: "preprocessor_config.json",
        dest: "mobile_det/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
        file: "inference.yml",
        dest: "mobile_det/inference.yml",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
        file: "model.safetensors",
        dest: "mobile_rec/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
        file: "config.json",
        dest: "mobile_rec/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
        file: "preprocessor_config.json",
        dest: "mobile_rec/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors",
        file: "inference.yml",
        dest: "mobile_rec/inference.yml",
      },
    ],
    sizeText: "~18 MB",
    kind: "ocr",
    ocrVariant: "v5-mobile",
  },
  {
    id: "ppocr-v5-server",
    name: "PP-OCR v5 server",
    description: "高精度检测+识别，ModelScope: PaddlePaddle/PP-OCRv5",
    repoId: "PaddlePaddle/PP-OCRv5_server_det_safetensors",
    files: [
      "server_det/model.safetensors",
      "server_det/config.json",
      "server_det/preprocessor_config.json",
      "server_det/inference.yml",
      "server_rec/model.safetensors",
      "server_rec/config.json",
      "server_rec/preprocessor_config.json",
      "server_rec/inference.yml",
    ],
    fileSpecs: [
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_det_safetensors",
        file: "model.safetensors",
        dest: "server_det/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_det_safetensors",
        file: "config.json",
        dest: "server_det/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_det_safetensors",
        file: "preprocessor_config.json",
        dest: "server_det/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_det_safetensors",
        file: "inference.yml",
        dest: "server_det/inference.yml",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
        file: "model.safetensors",
        dest: "server_rec/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
        file: "config.json",
        dest: "server_rec/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
        file: "preprocessor_config.json",
        dest: "server_rec/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv5_server_rec_safetensors",
        file: "inference.yml",
        dest: "server_rec/inference.yml",
      },
    ],
    sizeText: "~55 MB",
    kind: "ocr",
    ocrVariant: "v5-server",
  },
  {
    id: "ppocr-v6-tiny",
    name: "PP-OCR v6 tiny",
    description: "超轻量，速度最快，ModelScope: PaddlePaddle/PP-OCRv6",
    repoId: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
    files: [
      "tiny_det/model.safetensors",
      "tiny_det/config.json",
      "tiny_det/preprocessor_config.json",
      "tiny_det/inference.yml",
      "tiny_det/configuration.json",
      "tiny_rec/model.safetensors",
      "tiny_rec/config.json",
      "tiny_rec/preprocessor_config.json",
      "tiny_rec/inference.yml",
    ],
    fileSpecs: [
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
        file: "model.safetensors",
        dest: "tiny_det/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
        file: "config.json",
        dest: "tiny_det/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
        file: "preprocessor_config.json",
        dest: "tiny_det/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
        file: "inference.yml",
        dest: "tiny_det/inference.yml",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_det_safetensors",
        file: "configuration.json",
        dest: "tiny_det/configuration.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
        file: "model.safetensors",
        dest: "tiny_rec/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
        file: "config.json",
        dest: "tiny_rec/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
        file: "preprocessor_config.json",
        dest: "tiny_rec/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_tiny_rec_safetensors",
        file: "inference.yml",
        dest: "tiny_rec/inference.yml",
      },
    ],
    sizeText: "~8 MB",
    kind: "ocr",
    ocrVariant: "v6-tiny",
  },
  {
    id: "ppocr-v6-small",
    name: "PP-OCR v6 small",
    description: "均衡精度与速度，ModelScope: PaddlePaddle/PP-OCRv6",
    repoId: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
    files: [
      "small_det/model.safetensors",
      "small_det/config.json",
      "small_det/preprocessor_config.json",
      "small_det/inference.yml",
      "small_det/configuration.json",
      "small_rec/model.safetensors",
      "small_rec/config.json",
      "small_rec/preprocessor_config.json",
      "small_rec/inference.yml",
    ],
    fileSpecs: [
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
        file: "model.safetensors",
        dest: "small_det/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
        file: "config.json",
        dest: "small_det/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
        file: "preprocessor_config.json",
        dest: "small_det/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
        file: "inference.yml",
        dest: "small_det/inference.yml",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_det_safetensors",
        file: "configuration.json",
        dest: "small_det/configuration.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
        file: "model.safetensors",
        dest: "small_rec/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
        file: "config.json",
        dest: "small_rec/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
        file: "preprocessor_config.json",
        dest: "small_rec/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_small_rec_safetensors",
        file: "inference.yml",
        dest: "small_rec/inference.yml",
      },
    ],
    sizeText: "~22 MB",
    kind: "ocr",
    ocrVariant: "v6-small",
    recommended: true,
  },
  {
    id: "ppocr-v6-medium",
    name: "PP-OCR v6 medium",
    description: "高精度，适合离线批量，ModelScope: PaddlePaddle/PP-OCRv6",
    repoId: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
    files: [
      "medium_det/model.safetensors",
      "medium_det/config.json",
      "medium_det/preprocessor_config.json",
      "medium_det/inference.yml",
      "medium_det/configuration.json",
      "medium_rec/model.safetensors",
      "medium_rec/config.json",
      "medium_rec/preprocessor_config.json",
      "medium_rec/inference.yml",
    ],
    fileSpecs: [
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
        file: "model.safetensors",
        dest: "medium_det/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
        file: "config.json",
        dest: "medium_det/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
        file: "preprocessor_config.json",
        dest: "medium_det/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
        file: "inference.yml",
        dest: "medium_det/inference.yml",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_det_safetensors",
        file: "configuration.json",
        dest: "medium_det/configuration.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
        file: "model.safetensors",
        dest: "medium_rec/model.safetensors",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
        file: "config.json",
        dest: "medium_rec/config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
        file: "preprocessor_config.json",
        dest: "medium_rec/preprocessor_config.json",
      },
      {
        repoId: "PaddlePaddle/PP-OCRv6_medium_rec_safetensors",
        file: "inference.yml",
        dest: "medium_rec/inference.yml",
      },
    ],
    sizeText: "~158 MB",
    kind: "ocr",
    ocrVariant: "v6-medium",
  },
];
export const HUGGINGFACE_DOWNLOADABLE_MODELS: DownloadableModel[] = [
  {
    id: "hy-mt2-1.8b-q4",
    name: "Hy-MT2 1.8B Q4_K_M",
    description: "HF: mlx-community/Hy-MT2-1.8B-GGUF",
    repoId: "mlx-community/Hy-MT2-1.8B-GGUF",
    files: ["Hy-MT2-1.8B-Q4_K_M.gguf"],
    sizeText: "~1.1 GB",
    kind: "translation",
    recommended: true,
  },
];
export function listDownloadableModels(
  source: DownloadSource,
): DownloadableModel[] {
  console.info(`${DL_LOG_PREFIX} listDownloadableModels start`, { source });
  const models = source === "modelscope"
    ? MODELSCOPE_DOWNLOADABLE_MODELS
    : HUGGINGFACE_DOWNLOADABLE_MODELS;
  console.info(`${DL_LOG_PREFIX} listDownloadableModels success`, { source, count: models.length });
  console.debug(`${DL_LOG_PREFIX} listDownloadableModels detail`, { modelIds: models.map(m=>m.id) });
  return models;
}

export interface DownloadFamily {
  id: string;
  name: string;
  description: string;
  kind: "translation" | "ocr" | "mixed";
  models: DownloadableModel[];
}

export function listDownloadFamilies(source: DownloadSource): DownloadFamily[] {
  console.info(`${DL_LOG_PREFIX} listDownloadFamilies start`, { source });
  const models = listDownloadableModels(source);
  const translation = models.filter((model) => model.kind === "translation");
  const ocr = models.filter((model) => model.kind === "ocr");
  const families: DownloadFamily[] = [];
  if (translation.length) {
    families.push({
      id: "translation",
      name: "Hy-MT2 翻译模型族",
      description: "ModelScope LLM-Research/Hy-MT2 系列，GGUF 量化",
      kind: "translation",
      models: translation,
    });
  }
  if (ocr.length) {
    families.push({
      id: "ppocr",
      name: "PP-OCR 识别模型族",
      description: "PaddleOCR V5/V6 检测+识别，适配实时与批量",
      kind: "ocr",
      models: ocr,
    });
  }
  console.info(`${DL_LOG_PREFIX} listDownloadFamilies success`, { source, families: families.length, translationCount: translation.length, ocrCount: ocr.length });
  return families;
}

function isDesktopRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function startModelDownload(
  modelId: string,
  source: DownloadSource = "modelscope",
  invokeFn: InvokeFn = invoke,
): Promise<DownloadTaskState> {
  console.info(`${DL_LOG_PREFIX} startModelDownload start`, { modelId, source });
  if (!isDesktopRuntime()) {
    console.warn(`${DL_LOG_PREFIX} startModelDownload not desktop, returning simulated state`, { modelId, source });
    return {
      modelId,
      source,
      status: "downloading",
      progress: 0,
      downloadedBytes: 0,
      totalBytes: 100,
    };
  }
  const start = Date.now();
  try {
    const task = await invokeFn<DownloadTaskState>("start_model_download", {
      request: { modelId, source },
    });
    console.info(`${DL_LOG_PREFIX} startModelDownload success`, { modelId, source, status: task.status, progress: task.progress, durationMs: Date.now() - start });
    console.debug(`${DL_LOG_PREFIX} startModelDownload task`, { downloadedBytes: task.downloadedBytes, totalBytes: task.totalBytes });
    return task;
  } catch (error) {
    console.error(`${DL_LOG_PREFIX} startModelDownload invoke failed, falling back to simulated`, { modelId, source, error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
    // 后端尚未实现新命令时，回退为前端模拟，避免界面阻塞
    return {
      modelId,
      source,
      status: "downloading",
      progress: 0,
      downloadedBytes: 0,
      totalBytes: 100,
    };
  }
}

export async function cancelModelDownload(
  modelId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${DL_LOG_PREFIX} cancelModelDownload start`, { modelId });
  if (!isDesktopRuntime()) {
    console.warn(`${DL_LOG_PREFIX} cancelModelDownload not desktop, skip`, { modelId });
    return;
  }
  const start = Date.now();
  try {
    await invokeFn("cancel_model_download", { request: { modelId } });
    console.info(`${DL_LOG_PREFIX} cancelModelDownload success`, { modelId, durationMs: Date.now() - start });
  } catch (error) {
    console.warn(`${DL_LOG_PREFIX} cancelModelDownload failed or not implemented`, { modelId, error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
  }
}

export async function getDownloadTask(
  modelId: string,
  invokeFn: InvokeFn = invoke,
): Promise<DownloadTaskState | null> {
  console.debug(`${DL_LOG_PREFIX} getDownloadTask start`, { modelId });
  if (!isDesktopRuntime()) {
    console.debug(`${DL_LOG_PREFIX} getDownloadTask not desktop`, { modelId });
    return null;
  }
  try {
    const task = await invokeFn<DownloadTaskState | null>(
      "get_model_download_status",
      {
        request: { modelId },
      },
    );
    console.debug(`${DL_LOG_PREFIX} getDownloadTask result`, { modelId, status: task?.status, progress: task?.progress });
    return task;
  } catch (error) {
    console.warn(`${DL_LOG_PREFIX} getDownloadTask failed`, { modelId, error: error instanceof Error ? error.message : String(error) });
    return null;
  }
}

export function listenDownloadProgress(
  handler: (event: ModelDownloadProgressEvent) => void,
): Promise<UnlistenFn> {
  console.info(`${DL_LOG_PREFIX} listenDownloadProgress registering`);
  if (!isDesktopRuntime()) {
    console.warn(`${DL_LOG_PREFIX} listenDownloadProgress not desktop, returning no-op`);
    // 浏览器预览不监听真实事件
    return Promise.resolve(() => {});
  }
  return listen<ModelDownloadProgressEvent>(
    "model-download-progress",
    (event) => {
      const payload = event.payload;
      if (payload.status === "completed") {
        console.info(`${DL_LOG_PREFIX} download progress completed`, { modelId: payload.modelId, progress: payload.progress, downloadedBytes: payload.downloadedBytes, totalBytes: payload.totalBytes });
      } else if (payload.status === "error" || payload.status === "cancelled") {
        console.warn(`${DL_LOG_PREFIX} download progress status`, { modelId: payload.modelId, status: payload.status, message: payload.message, progress: payload.progress });
      } else {
        console.debug(`${DL_LOG_PREFIX} download progress`, { modelId: payload.modelId, status: payload.status, progress: payload.progress, downloadedBytes: payload.downloadedBytes, totalBytes: payload.totalBytes });
      }
      handler(event.payload);
    },
  );
}

export interface DownloadedModelInfo {
  modelId: string;
  downloaded: boolean;
  baseDir: string;
}

export async function listDownloadedModels(
  invokeFn: InvokeFn = invoke,
): Promise<DownloadedModelInfo[]> {
  console.info(`${DL_LOG_PREFIX} listDownloadedModels start`);
  if (!isDesktopRuntime()) {
    console.debug(`${DL_LOG_PREFIX} listDownloadedModels not desktop`);
    return [];
  }
  const start = Date.now();
  try {
    const models = await invokeFn<DownloadedModelInfo[]>("list_downloaded_models");
    console.info(`${DL_LOG_PREFIX} listDownloadedModels success`, { count: models.length, durationMs: Date.now() - start });
    console.debug(`${DL_LOG_PREFIX} listDownloadedModels detail`, { modelIds: models.map(m=>m.modelId) });
    return models;
  } catch (error) {
    console.warn(`${DL_LOG_PREFIX} listDownloadedModels failed`, { error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
    return [];
  }
}

export async function activateDownloadedModel(
  modelId: string,
  invokeFn: InvokeFn = invoke,
): Promise<unknown> {
  console.info(`${DL_LOG_PREFIX} activateDownloadedModel start`, { modelId });
  if (!isDesktopRuntime()) {
    console.warn(`${DL_LOG_PREFIX} activateDownloadedModel not desktop`, { modelId });
    throw new Error("桌面端功能");
  }
  const start = Date.now();
  try {
    const result = await invokeFn("activate_downloaded_model", {
      request: { modelId },
    });
    console.info(`${DL_LOG_PREFIX} activateDownloadedModel success`, { modelId, durationMs: Date.now() - start });
    return result;
  } catch (error) {
    console.error(`${DL_LOG_PREFIX} activateDownloadedModel failed`, { modelId, error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
    throw error;
  }
}

export async function deleteDownloadedModel(
  modelId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${DL_LOG_PREFIX} deleteDownloadedModel start`, { modelId });
  if (!isDesktopRuntime()) {
    console.warn(`${DL_LOG_PREFIX} deleteDownloadedModel not desktop`, { modelId });
    throw new Error("桌面端功能");
  }
  const start = Date.now();
  try {
    await invokeFn("delete_downloaded_model", {
      request: { modelId },
    });
    console.info(`${DL_LOG_PREFIX} deleteDownloadedModel success`, { modelId, durationMs: Date.now() - start });
  } catch (error) {
    console.error(`${DL_LOG_PREFIX} deleteDownloadedModel failed`, { modelId, error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
    throw error;
  }
}

/** 工具：ModelScope resolve URL（供后端或文档展示） */
export function modelscopeResolveUrl(repoId: string, file: string): string {
  return `https://www.modelscope.cn/models/${repoId}/resolve/master/${file}`;
}
