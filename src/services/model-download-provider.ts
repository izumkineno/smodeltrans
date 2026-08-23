import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

/**
 * ModelScope 作为默认下载源。
 * 前端通过此服务屏蔽底层的 ModelScope HTTP / SDK 细节，
 * 后端 Rust 侧对接 ModelScope 的 resolve 接口进行流式下载。
 */
export type DownloadSource = "modelscope" | "huggingface";

export interface DownloadableModel {
  id: string;
  name: string;
  description: string;
  repoId: string;
  files: string[];
  sizeText: string;
  kind: "translation" | "ocr" | "font";
  ocrVariant?: string;
  recommended?: boolean;
}

export type DownloadStatus = "idle" | "downloading" | "completed" | "error" | "cancelled";

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

type InvokeFn = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

/** ModelScope 默认源的推荐清单 */
export const MODELSCOPE_DOWNLOADABLE_MODELS: DownloadableModel[] = [
  {
    id: "hy-mt2-1.8b-q4",
    name: "Hy-MT2 1.8B Q4_K_M",
    description: "多语言翻译核心，ModelScope: LLM-Research/Hy-MT2",
    repoId: "LLM-Research/Hy-MT2-1.8B",
    files: ["Hy-MT2-1.8B-Q4_K_M.gguf"],
    sizeText: "~1.1 GB",
    kind: "translation",
    recommended: true,
  },
  {
    id: "ppocr-v5-mobile",
    name: "PP-OCR v5 mobile",
    description: "轻量检测+识别，适合实时字幕",
    repoId: "damo/PPOCR-v5-mobile",
    files: ["det.onnx", "rec.onnx", "inference.yml"],
    sizeText: "~18 MB",
    kind: "ocr",
    ocrVariant: "v5-mobile",
    recommended: true,
  },
  {
    id: "ppocr-v5-server",
    name: "PP-OCR v5 server",
    description: "高精度检测+识别，适合批量图片",
    repoId: "damo/PPOCR-v5-server",
    files: ["det.onnx", "rec.onnx", "inference.yml"],
    sizeText: "~55 MB",
    kind: "ocr",
    ocrVariant: "v5-server",
  },
  {
    id: "ppocr-v6-tiny",
    name: "PP-OCR v6 tiny",
    description: "超轻量，速度最快",
    repoId: "damo/PPOCR-v6-tiny",
    files: ["det.onnx", "rec.onnx"],
    sizeText: "~8 MB",
    kind: "ocr",
    ocrVariant: "v6-tiny",
  },
  {
    id: "ppocr-v6-small",
    name: "PP-OCR v6 small",
    description: "均衡精度与速度",
    repoId: "damo/PPOCR-v6-small",
    files: ["det.onnx", "rec.onnx"],
    sizeText: "~22 MB",
    kind: "ocr",
    ocrVariant: "v6-small",
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

export function listDownloadableModels(source: DownloadSource): DownloadableModel[] {
  return source === "modelscope" ? MODELSCOPE_DOWNLOADABLE_MODELS : HUGGINGFACE_DOWNLOADABLE_MODELS;
}

/** 浏览器预览时无 Tauri 后端，直接走前端模拟进度 */
function isDesktopRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

export async function startModelDownload(
  modelId: string,
  source: DownloadSource = "modelscope",
  invokeFn: InvokeFn = invoke,
): Promise<DownloadTaskState> {
  if (!isDesktopRuntime()) {
    // 浏览器预览：直接返回模拟的 downloading 状态，由前端定时器驱动
    return {
      modelId,
      source,
      status: "downloading",
      progress: 0,
      downloadedBytes: 0,
      totalBytes: 100,
    };
  }
  try {
    return await invokeFn<DownloadTaskState>("start_model_download", {
      request: { modelId, source },
    });
  } catch {
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
  if (!isDesktopRuntime()) return;
  try {
    await invokeFn("cancel_model_download", { request: { modelId } });
  } catch {
    // 忽略未实现
  }
}

export async function getDownloadTask(
  modelId: string,
  invokeFn: InvokeFn = invoke,
): Promise<DownloadTaskState | null> {
  if (!isDesktopRuntime()) return null;
  try {
    return await invokeFn<DownloadTaskState | null>("get_model_download_status", {
      request: { modelId },
    });
  } catch {
    return null;
  }
}

export function listenDownloadProgress(
  handler: (event: ModelDownloadProgressEvent) => void,
): Promise<UnlistenFn> {
  if (!isDesktopRuntime()) {
    // 浏览器预览不监听真实事件
    return Promise.resolve(() => {});
  }
  return listen<ModelDownloadProgressEvent>("model-download-progress", (event) => {
    handler(event.payload);
  });
}

/** 工具：ModelScope resolve URL（供后端或文档展示） */
export function modelscopeResolveUrl(repoId: string, file: string): string {
  return `https://www.modelscope.cn/models/${repoId}/resolve/master/${file}`;
}
