<script setup lang="ts">
import { computed, h, onActivated, onBeforeUnmount, onDeactivated, ref } from "vue";
import type { VNodeChild } from "vue";
import { listen } from "@tauri-apps/api/event";
import type { UnlistenFn } from "@tauri-apps/api/event";
import { readImage } from "@tauri-apps/plugin-clipboard-manager";
import {
  NAlert,
  NButton,
  NCard,
  NEmpty,
  NInput,
  NProgress,
  NSpin,
  NSpace,
  NTag,
  useMessage,
} from "naive-ui";
import type { ImageRenderToolbarProps } from "naive-ui";
import ImagePreviewFrame from "./ImagePreviewFrame.vue";
import {
  copyTranslationText,
  createImagePreview,
  MAX_IMAGE_BYTES,
  MAX_IMAGE_PIXELS,
  releaseImagePreview,
  saveImageDataUrl,
  saveTranslationText,
  SUPPORTED_IMAGE_EXTENSIONS,
  validateImageFile,
  validateImagePreview,
} from "../services/file-adapter";
import {
  createTranslationRequestId,
  isTranslationCancellation,
  translationProvider,
} from "../services/translation-provider";
import type { TranslationProgress } from "../services/translation-provider";
import { targetLanguage } from "../services/workspace-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";

type WorkflowState = "idle" | "preview" | "processing" | "result" | "cancelled" | "error";
type TagType = "default" | "success" | "warning" | "error" | "info";

const workflowState = ref<WorkflowState>("idle");
const selectedFile = ref<File | null>(null);
const previewUrl = ref<string | null>(null);
const resultText = ref<string | null>(null);
const annotatedResultUrl = ref<string | null>(null);
const resultIsTranslated = ref(false);
const providerLabel = ref("");
const translationDurationMs = ref<number | null>(null);
const errorMessage = ref("");
const actionFeedback = ref("");
const statusMessage = ref("请选择一张图片开始。");
const isDragActive = ref(false);
const processingProgress = ref(0);
const fileInput = ref<HTMLInputElement | null>(null);
const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const activeController = ref<AbortController | null>(null);

let progressUnlisten: UnlistenFn | undefined;
let selectionVersion = 0;

const toast = useMessage();

function notify(type: WorkspaceToastType, message: string) {
  showWorkspaceToast(toast, type, message);
}

function setStatusToast(type: WorkspaceToastType, message: string) {
  statusMessage.value = message;
  notify(type, message);
}

function setErrorState(message: string) {
  errorMessage.value = message;
  workflowState.value = "error";
  statusMessage.value = message;
  notify("error", message);
}

function setActionFeedback(type: WorkspaceToastType, message: string) {
  actionFeedback.value = message;
  notify(type, message);
}

const workflowStatus = computed(() => {
  switch (workflowState.value) {
    case "preview":
      return "准备翻译";
    case "processing":
      return "翻译中";
    case "result":
      return "结果就绪";
    case "cancelled":
      return "已取消翻译";
    case "error":
      return "需要处理";
    default:
      return "等待上传图片";
  }
});

const workflowTagType = computed<TagType>(() => {
  switch (workflowState.value) {
    case "preview":
      return "info";
    case "processing":
      return "warning";
    case "result":
      return "success";
    case "cancelled":
      return "default";
    case "error":
      return "error";
    default:
      return "default";
  }
});

const canStartTranslation = computed(
  () =>
    selectedFile.value !== null &&
    (workflowState.value === "preview" ||
      workflowState.value === "cancelled" ||
      workflowState.value === "error"),
);

const startButtonLabel = computed(() =>
  workflowState.value === "cancelled" || workflowState.value === "error"
    ? "再次尝试"
    : "开始翻译",
);

const fileSizeLabel = computed(() => {
  const file = selectedFile.value;
  if (!file) {
    return "";
  }
  if (file.size < 1024) {
    return `${file.size} B`;
  }
  if (file.size < 1024 * 1024) {
    return `${(file.size / 1024).toFixed(1)} KB`;
  }
  return `${(file.size / (1024 * 1024)).toFixed(1)} MB`;
});

const translationDurationLabel = computed(() => {
  const durationMs = translationDurationMs.value;
  if (durationMs === null || !Number.isFinite(durationMs) || durationMs < 0) {
    return "";
  }
  return durationMs < 1000
    ? `${Math.max(1, Math.round(durationMs))} ms`
    : `${(durationMs / 1000).toFixed(2)} 秒`;
});

function clearProgressListener() {
  progressUnlisten?.();
  progressUnlisten = undefined;
}

async function selectFile(file: File | undefined) {
  if (!file) {
    return;
  }

  const currentSelection = ++selectionVersion;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();

  const validationMessage = validateImageFile(file);
  if (validationMessage) {
    releaseImagePreview(previewUrl.value);
    selectedFile.value = null;
    previewUrl.value = null;
    resultText.value = null;
    annotatedResultUrl.value = null;
    resultIsTranslated.value = false;
    translationDurationMs.value = null;
    providerLabel.value = "";
    actionFeedback.value = "";
    setErrorState(validationMessage);
    return;
  }

  const nextPreviewUrl = createImagePreview(file);
  const previewValidationMessage = await validateImagePreview(nextPreviewUrl);
  if (currentSelection !== selectionVersion) {
    releaseImagePreview(nextPreviewUrl);
    return;
  }
  if (previewValidationMessage) {
    releaseImagePreview(nextPreviewUrl);
    releaseImagePreview(previewUrl.value);
    selectedFile.value = null;
    previewUrl.value = null;
    resultText.value = null;
    annotatedResultUrl.value = null;
    resultIsTranslated.value = false;
    translationDurationMs.value = null;
    providerLabel.value = "";
    actionFeedback.value = "";
    setErrorState(previewValidationMessage);
    return;
  }

  releaseImagePreview(previewUrl.value);
  selectedFile.value = file;
  previewUrl.value = nextPreviewUrl;
  resultText.value = null;
  providerLabel.value = "";
  annotatedResultUrl.value = null;
  resultIsTranslated.value = false;
  translationDurationMs.value = null;
  errorMessage.value = "";
  actionFeedback.value = "";
  workflowState.value = "preview";
  setStatusToast("success", `${file.name} 已准备好翻译。`);
}

function handlePreviewError() {
  selectionVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  processingProgress.value = 0;
  releaseImagePreview(previewUrl.value);
  selectedFile.value = null;
  previewUrl.value = null;
  resultText.value = null;
  providerLabel.value = "";
  annotatedResultUrl.value = null;
  resultIsTranslated.value = false;
  translationDurationMs.value = null;
  actionFeedback.value = "";
  setErrorState("无法显示这张图片，请选择其他文件。");
}

function handleFileInput(event: Event) {
  const input = event.currentTarget as HTMLInputElement;
  void selectFile(input.files?.[0]);
  input.value = "";
}

function handleDragOver(event: DragEvent) {
  event.preventDefault();
  isDragActive.value = true;
}

function handleDragLeave() {
  isDragActive.value = false;
}

function handleDrop(event: DragEvent) {
  event.preventDefault();
  isDragActive.value = false;
  void selectFile(event.dataTransfer?.files[0]);
}

function openFilePicker() {
  fileInput.value?.click();
}

async function startTranslation() {
  const file = selectedFile.value;
  if (!file || !canStartTranslation.value) {
    return;
  }

  const controller = new AbortController();
  activeController.value = controller;
  resultText.value = null;
  annotatedResultUrl.value = null;
  resultIsTranslated.value = false;
  translationDurationMs.value = null;
  providerLabel.value = "";
  errorMessage.value = "";
  actionFeedback.value = "";
  workflowState.value = "processing";
  statusMessage.value = "翻译正在进行中。";
  clearProgressListener();
  processingProgress.value = 0;
  const requestId = createTranslationRequestId();

  try {
    if (isDesktopRuntime) {
      progressUnlisten = await listen<TranslationProgress>("translation-progress", (event) => {
        if (
          event.payload.requestId !== requestId ||
          activeController.value !== controller ||
          workflowState.value !== "processing"
        ) {
          return;
        }
        processingProgress.value = Math.min(100, Math.max(0, Math.round(event.payload.progress)));
        statusMessage.value = event.payload.stage;
      });
    }

    const translation = await translationProvider.translate(
      {
        file,
        targetLanguage: targetLanguage.value,
        requestId,
      },
      controller.signal,
    );

    if (controller.signal.aborted || activeController.value !== controller) {
      if (activeController.value === controller && controller.signal.aborted) {
        workflowState.value = "cancelled";
        statusMessage.value = "翻译已取消，图片预览仍可用。";
      }
      return;
    }

    resultText.value = translation.text;
    annotatedResultUrl.value = translation.annotatedImageDataUrl;
    providerLabel.value = translation.providerLabel;
    resultIsTranslated.value = translation.isTranslated;
    translationDurationMs.value = translation.durationMs;
    processingProgress.value = 100;
    workflowState.value = "result";
    setStatusToast(
      translation.isTranslated ? "success" : "warning",
      translation.isTranslated
        ? "翻译结果已准备好。"
        : "PP-OCRv5 识别结果已准备好；Hy 翻译需要 CUDA。",
    );
  } catch (error) {
    if (controller.signal.aborted || isTranslationCancellation(error)) {
      if (activeController.value === controller) {
        workflowState.value = "cancelled";
        statusMessage.value = "翻译已取消，图片预览仍可用。";
      }
      return;
    }
    if (activeController.value !== controller) {
      return;
    }
    setErrorState(
      error instanceof Error
        ? error.message
        : "Candle 图片翻译后端未能完成，请检查模型配置后重试。",
    );
  } finally {
    if (activeController.value === controller) {
      activeController.value = null;
      clearProgressListener();
      if (workflowState.value !== "result") {
        processingProgress.value = 0;
      }
    }
  }
}

function cancelTranslation() {
  if (!activeController.value) {
    return;
  }
  activeController.value.abort();
  clearProgressListener();
  processingProgress.value = 0;
  workflowState.value = "cancelled";
  setStatusToast("warning", "翻译已取消，图片预览仍可用。");
}

function resetWorkflow() {
  selectionVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  releaseImagePreview(previewUrl.value);
  selectedFile.value = null;
  previewUrl.value = null;
  resultText.value = null;
  providerLabel.value = "";
  annotatedResultUrl.value = null;
  resultIsTranslated.value = false;
  translationDurationMs.value = null;
  errorMessage.value = "";
  actionFeedback.value = "";
  processingProgress.value = 0;
  workflowState.value = "idle";
  statusMessage.value = "请选择一张图片开始。";
  notify("info", "OCR 翻译工作区已重置。");
}

async function copyResult() {
  if (!resultText.value) {
    return;
  }
  try {
    await copyTranslationText(resultText.value);
    setActionFeedback("success", "结果已复制到剪贴板。");
  } catch {
    setActionFeedback("error", "剪贴板不可用，请手动选择文本进行复制。");
  }
}

function saveResult() {
  if (!resultText.value || !selectedFile.value) {
    return;
  }
  try {
    saveTranslationText(resultText.value, selectedFile.value.name);
    setActionFeedback("success", "结果已保存为文本文件。");
  } catch {
    setActionFeedback("error", "结果无法保存，请从结果面板重试。");
  }
}

function saveAnnotatedImage() {
  if (!annotatedResultUrl.value || !selectedFile.value) {
    return;
  }
  try {
    saveImageDataUrl(annotatedResultUrl.value, selectedFile.value.name);
    setActionFeedback("success", "标注图片已保存。");
  } catch {
    setActionFeedback("error", "标注图片无法保存，请重试。");
  }
}

function renderImageToolbar({ nodes }: ImageRenderToolbarProps): VNodeChild {
  const zoomOutNode = h("span", { "data-image-preview-zoom": "out" }, [nodes.zoomOut]);
  const zoomInNode = h("span", { "data-image-preview-zoom": "in" }, [nodes.zoomIn]);
  return [
    nodes.prev,
    nodes.next,
    nodes.rotateCounterclockwise,
    nodes.rotateClockwise,
    nodes.resizeToOriginalSize,
    zoomOutNode,
    zoomInNode,
    nodes.download,
    nodes.close,
  ];
}

function handleImagePreviewWheel(event: WheelEvent) {
  if (event.deltaY === 0) {
    return;
  }
  const target = event.target;
  if (!(target instanceof Element)) {
    return;
  }
  const container = target.closest(".n-image-preview-container");
  if (!container) {
    return;
  }
  const direction = event.deltaY < 0 ? "in" : "out";
  const zoomControl = container.querySelector<HTMLElement>(
    `[data-image-preview-zoom="${direction}"]`,
  );
  const zoomButton =
    zoomControl?.querySelector<HTMLElement>(".n-base-icon, button, [role='button']") ??
    zoomControl;
  if (!zoomButton) {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  zoomButton.click();
}

function addImagePreviewWheelListener() {
  window.addEventListener("wheel", handleImagePreviewWheel, {
    capture: true,
    passive: false,
  });
}

function removeImagePreviewWheelListener() {
  window.removeEventListener("wheel", handleImagePreviewWheel, { capture: true });
}

function clipboardFileFromEvent(event: ClipboardEvent): File | null {
  const clipboardData = event.clipboardData;
  if (!clipboardData) {
    return null;
  }
  return (
    Array.from(clipboardData.files).find((item) => item.type.startsWith("image/")) ??
    Array.from(clipboardData.items)
      .find((item) => item.kind === "file" && item.type.startsWith("image/"))
      ?.getAsFile() ??
    null
  );
}

async function readTauriClipboardImage(): Promise<File> {
  const image = await readImage();
  const [rgba, size] = await Promise.all([image.rgba(), image.size()]);
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("无法创建剪贴板图片画布。");
  }
  context.putImageData(
    new ImageData(new Uint8ClampedArray(rgba), size.width, size.height),
    0,
    0,
  );
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((value) => {
      if (value) {
        resolve(value);
      } else {
        reject(new Error("无法编码剪贴板图片。"));
      }
    }, "image/png");
  });
  return new File([blob], `clipboard-${Date.now()}.png`, { type: "image/png" });
}

async function handlePaste(event: ClipboardEvent) {
  const webFile = clipboardFileFromEvent(event);
  if (webFile) {
    event.preventDefault();
    await selectFile(webFile);
    return;
  }
  if (!isDesktopRuntime) {
    return;
  }
  const clipboardTypes = Array.from(event.clipboardData?.types ?? []);
  const mayContainImage =
    clipboardTypes.length === 0 ||
    clipboardTypes.includes("Files") ||
    clipboardTypes.some((type) => type.startsWith("image/"));
  if (!mayContainImage) {
    return;
  }
  event.preventDefault();
  try {
    await selectFile(await readTauriClipboardImage());
  } catch {
    setActionFeedback("warning", "剪贴板中没有可读取的图片。");
  }
}

onActivated(() => {
  window.addEventListener("paste", handlePaste);
  addImagePreviewWheelListener();
});

onDeactivated(() => {
  window.removeEventListener("paste", handlePaste);
  removeImagePreviewWheelListener();
});

onBeforeUnmount(() => {
  selectionVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  window.removeEventListener("paste", handlePaste);
  removeImagePreviewWheelListener();
  releaseImagePreview(previewUrl.value);
});
</script>

<template>
  <div class="ocr-translation-page">
    <p class="sr-only" aria-live="polite">{{ statusMessage }}</p>

    <section class="workflow-grid" aria-label="OCR 翻译流程">
      <n-card class="panel input-panel" :bordered="false">
        <div class="panel-heading">
          <div class="panel-title">
            <span class="section-number">01</span>
            <div>
              <p class="panel-kicker">输入 / 图片</p>
              <h2>选择图片</h2>
              <p class="panel-copy">将图片拖放到这里，或从设备中浏览。</p>
            </div>
          </div>
          <n-tag class="state-tag" :type="workflowTagType" round size="small">{{ workflowStatus }}</n-tag>
        </div>

        <div
          v-if="!selectedFile"
          class="drop-zone"
          :class="{ 'drop-zone-active': isDragActive }"
          role="button"
          tabindex="0"
          aria-label="选择图片文件"
          @click="openFilePicker"
          @keydown.enter.prevent="openFilePicker"
          @keydown.space.prevent="openFilePicker"
          @dragover="handleDragOver"
          @dragleave="handleDragLeave"
          @drop="handleDrop"
        >
          <input
            id="image-file"
            ref="fileInput"
            class="file-input"
            type="file"
            :accept="SUPPORTED_IMAGE_EXTENSIONS.map((extension) => `.${extension}`).join(',')"
            aria-label="选择图片文件"
            @change="handleFileInput"
          />
          <div class="drop-zone-label">
            <span class="drop-icon" aria-hidden="true">
              <svg viewBox="0 0 32 32" fill="none">
                <rect x="4.5" y="5.5" width="23" height="21" rx="3" stroke="currentColor" stroke-width="1.5" />
                <circle cx="11" cy="12" r="2" stroke="currentColor" stroke-width="1.5" />
                <path d="m7 23 6.2-6 4.3 4 3-3 4.5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </span>
            <span class="drop-zone-kicker">拖放图片以预览</span>
            <span class="drop-zone-copy">或从设备选择文件，或按 Ctrl+V 粘贴图片</span>
            <n-button text type="primary" size="small" class="drop-zone-action" @click.stop="openFilePicker">
              浏览文件 <span aria-hidden="true">↗</span>
            </n-button>
          </div>
        </div>

        <div v-else class="preview-layout">
          <ImagePreviewFrame
            title="输入预览"
            state-label="已加载"
            :src="previewUrl ?? undefined"
            :preview-src="previewUrl ?? undefined"
            :alt="`预览：${selectedFile.name}`"
            variant="input"
            :render-toolbar="renderImageToolbar"
            @error="handlePreviewError"
          />

          <div class="preview-details">
            <div class="file-identity">
              <span class="file-type-mark" aria-hidden="true">IMG</span>
              <div class="file-identity-copy">
                <p class="detail-label">已选文件</p>
                <h3>{{ selectedFile.name }}</h3>
                <p class="file-meta">
                  {{ fileSizeLabel }} <span aria-hidden="true">·</span>
                  {{ selectedFile.type || "图片文件" }}
                </p>
              </div>
            </div>

            <div class="button-row">
              <n-button secondary @click="resetWorkflow">选择其他</n-button>
              <n-button
                v-if="workflowState !== 'processing'"
                type="primary"
                :disabled="!canStartTranslation"
                @click="startTranslation"
              >
                {{ startButtonLabel }}
              </n-button>
              <n-button v-else tertiary type="warning" @click="cancelTranslation">取消翻译</n-button>
            </div>

            <n-alert
              v-if="workflowState === 'cancelled'"
              class="inline-alert"
              type="warning"
              title="翻译已取消"
              :show-icon="true"
            >
              预览仍在此处，可以随时重新开始。
            </n-alert>
            <n-alert
              v-if="workflowState === 'error'"
              class="inline-alert"
              type="error"
              title="翻译需要处理"
              :show-icon="true"
            >
              {{ errorMessage }}
            </n-alert>
          </div>
        </div>

        <n-alert
          v-if="workflowState === 'error' && !selectedFile"
          class="inline-alert"
          type="error"
          title="图片无法使用"
          :show-icon="true"
        >
          {{ errorMessage }}
        </n-alert>

        <p class="input-helper">
          支持：{{ SUPPORTED_IMAGE_EXTENSIONS.map((extension) => extension.toUpperCase()).join(" · ") }}
          <span aria-hidden="true"> / </span>
          最大 {{ MAX_IMAGE_BYTES / (1024 * 1024) }} MB · {{ MAX_IMAGE_PIXELS / 1000000 }} MP
        </p>
      </n-card>

      <n-card class="panel output-panel" :bordered="false">
        <div class="panel-heading">
          <div class="panel-title">
            <span class="section-number">02</span>
            <div>
              <p class="panel-kicker">输出 / 文本</p>
              <h2>翻译结果</h2>
              <p class="panel-copy">复制结果，或将其保存为文本文件。</p>
            </div>
          </div>
          <n-tag v-if="providerLabel" type="success" round size="small">{{ providerLabel }}</n-tag>
        </div>

        <div v-if="workflowState === 'processing'" class="processing-state" aria-busy="true">
          <div class="processing-visual" aria-hidden="true">
            <n-spin size="medium" />
          </div>
          <div class="processing-copy">
            <p class="detail-label">正在处理图片</p>
            <p>Candle OCR 与 Hy-MT2 正在准备结果。</p>
          </div>
          <n-progress
            type="line"
            :percentage="processingProgress"
            :show-indicator="false"
            aria-label="翻译进度"
            :aria-valuetext="`${statusMessage}，${processingProgress}% 完成`"
            processing
          />
          <div class="progress-meta">
            <span id="progress-status">{{ statusMessage }}</span>
            <strong>{{ processingProgress }}%</strong>
          </div>
        </div>

        <div v-else-if="workflowState === 'result' && resultText" class="result-state">
          <ImagePreviewFrame
            v-if="annotatedResultUrl"
            title="OCR 标注预览"
            variant="result"
            :src="annotatedResultUrl"
            :preview-src="annotatedResultUrl"
            :alt="resultIsTranslated ? 'OCR 标注后的翻译图片' : 'PP-OCRv5 标注识别图片'"
            :render-toolbar="renderImageToolbar"
          >
            <template #actions>
              <n-button text size="small" @click="saveAnnotatedImage">保存 PNG</n-button>
            </template>
          </ImagePreviewFrame>
          <div class="result-toolbar">
            <span>{{ resultIsTranslated ? "翻译输出" : "OCR 输出" }}</span>
            <div class="result-toolbar-meta">
              <span>{{ resultIsTranslated ? "译文" : "原文" }}</span>
              <span
                v-if="translationDurationLabel"
                class="result-duration"
                :aria-label="resultIsTranslated ? '翻译用时' : '处理用时'"
              >
                {{ resultIsTranslated ? "翻译用时" : "处理用时" }}
                {{ translationDurationLabel }}
              </span>
            </div>
          </div>

          <p v-if="!resultIsTranslated" class="result-mode-note">
            当前使用 CPU，仅展示 PP-OCRv5 识别文本；切换 CUDA 后可启用 Hy-MT2 翻译。
          </p>
          <n-input
            class="result-input"
            type="textarea"
            :value="resultText"
            readonly
            :autosize="{ minRows: 9, maxRows: 16 }"
            :aria-label="resultIsTranslated ? '翻译结果文本' : 'PP-OCRv5 识别文本'"
          />
          <div class="result-actions">
            <n-space :size="10" wrap>
              <n-button secondary @click="copyResult">复制文本</n-button>
              <n-button type="primary" @click="saveResult">保存 .txt</n-button>
            </n-space>
            <p v-if="actionFeedback" class="action-feedback" aria-live="polite">{{ actionFeedback }}</p>
          </div>
        </div>

        <div v-else-if="workflowState === 'error'" class="output-message">
          <n-alert class="inline-alert" type="error" title="本次翻译未完成" :show-icon="true">
            {{ errorMessage }}
          </n-alert>
          <n-button v-if="selectedFile" type="primary" @click="startTranslation">再次尝试</n-button>
          <n-button v-else secondary @click="openFilePicker">选择图片</n-button>
        </div>

        <div v-else-if="workflowState === 'cancelled'" class="output-message">
          <n-alert class="inline-alert" type="warning" title="流程已取消" :show-icon="true">
            没有丢失任何内容，图片预览已准备好，可以再次尝试。
          </n-alert>
          <n-button type="primary" @click="startTranslation">再次尝试</n-button>
        </div>

        <div v-else class="empty-output">
          <n-empty
            size="small"
            :description="
              workflowState === 'preview'
                ? '开始翻译以查看首版预览结果。'
                : '翻译结果将在此处显示。'
            "
          />
        </div>
      </n-card>
    </section>

    <footer class="workspace-footer">
      <span>Candle 本地推理</span>
      <span class="footer-separator" aria-hidden="true"></span>
      <span>PP-OCRv5 · Hy-MT2 · Tauri</span>
      <span class="footer-spacer"></span>
      <span>PNG · JPG · WEBP · GIF · BMP</span>
    </footer>
  </div>
</template>
