import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

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
  return source === "modelscope"
    ? MODELSCOPE_DOWNLOADABLE_MODELS
    : HUGGINGFACE_DOWNLOADABLE_MODELS;
}

export interface DownloadFamily {
  id: string;
  name: string;
  description: string;
  kind: "translation" | "ocr" | "mixed";
  models: DownloadableModel[];
}

export function listDownloadFamilies(source: DownloadSource): DownloadFamily[] {
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
    return await invokeFn<DownloadTaskState | null>(
      "get_model_download_status",
      {
        request: { modelId },
      },
    );
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
  return listen<ModelDownloadProgressEvent>(
    "model-download-progress",
    (event) => {
      handler(event.payload);
    },
  );
}

/** 工具：ModelScope resolve URL（供后端或文档展示） */
export function modelscopeResolveUrl(repoId: string, file: string): string {
  return `https://www.modelscope.cn/models/${repoId}/resolve/master/${file}`;
}
