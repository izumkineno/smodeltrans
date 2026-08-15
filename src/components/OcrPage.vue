<script setup lang="ts">
import { computed, h, onActivated, onBeforeUnmount, onDeactivated, ref } from "vue";
import type { VNodeChild } from "vue";
import { readImage } from "@tauri-apps/plugin-clipboard-manager";
import { listen } from "@tauri-apps/api/event";
import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  NAlert,
  NButton,
  NCard,
  NEmpty,
  NInput,
  NProgress,
  NSpin,
  NSpace,
  NTabPane,
  NTabs,
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
  ocrProvider,
  type OcrRegion,
  type OcrResult,
  type TranslationProgress,
} from "../services/translation-provider";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";

type OcrWorkflowState = "idle" | "preview" | "processing" | "result" | "cancelled" | "error";
type TagType = "default" | "success" | "warning" | "error" | "info";

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const workflowState = ref<OcrWorkflowState>("idle");
const selectedFile = ref<File | null>(null);
const previewUrl = ref<string | null>(null);
const annotatedResultUrl = ref<string | null>(null);
const resultText = ref<string | null>(null);
const resultMarkdown = ref<string | null>(null);
const providerLabel = ref("");
const durationMs = ref<number | null>(null);
const resultRegions = ref<OcrRegion[]>([]);
const resultImageWidth = ref(0);
const resultImageHeight = ref(0);
const selectedImageText = ref("");
const outputMode = ref<"text" | "markdown">("text");
const isDragActive = ref(false);
const processingProgress = ref(0);
const statusMessage = ref("请选择一张图片开始 OCR。");
const errorMessage = ref("");
const actionFeedback = ref("");
const fileInput = ref<HTMLInputElement | null>(null);
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
      return "准备识别";
    case "processing":
      return "识别中";
    case "result":
      return "结果就绪";
    case "cancelled":
      return "已取消";
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

const canStartOcr = computed(
  () =>
    isDesktopRuntime &&
    selectedFile.value !== null &&
    (workflowState.value === "preview" ||
      workflowState.value === "cancelled" ||
      workflowState.value === "error"),
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

const durationLabel = computed(() => {
  if (durationMs.value === null || !Number.isFinite(durationMs.value) || durationMs.value < 0) {
    return "";
  }
  return durationMs.value < 1000
    ? `${Math.max(1, Math.round(durationMs.value))} ms`
    : `${(durationMs.value / 1000).toFixed(2)} 秒`;
});

const activeOutput = computed(() =>
  outputMode.value === "markdown" ? resultMarkdown.value ?? "" : resultText.value ?? "",
);
const imageSelectionHint = computed(() => {
  if (!selectedImageText.value) {
    return "点击图片打开大图，拖拽文字进行选择，按 Ctrl+C 可直接复制。";
  }
  const characterCount = Array.from(selectedImageText.value.replace(/\n/g, "")).length;
  return `已选择 ${characterCount} 个字符，按 Ctrl+C 复制。`;
});

function clearProgressListener() {
  progressUnlisten?.();
  progressUnlisten = undefined;
}

function resetResult() {
  annotatedResultUrl.value = null;
  resultText.value = null;
  resultMarkdown.value = null;
  providerLabel.value = "";
  durationMs.value = null;
  resultRegions.value = [];
  resultImageWidth.value = 0;
  resultImageHeight.value = 0;
  selectedImageText.value = "";
  outputMode.value = "text";
  actionFeedback.value = "";
  errorMessage.value = "";
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
    resetResult();
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
    resetResult();
    setErrorState(previewValidationMessage);
    return;
  }

  releaseImagePreview(previewUrl.value);
  selectedFile.value = file;
  previewUrl.value = nextPreviewUrl;
  resetResult();
  workflowState.value = "preview";
  setStatusToast("success", `${file.name} 已准备好进行 OCR。`);
}

function handlePreviewError() {
  selectionVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  releaseImagePreview(previewUrl.value);
  selectedFile.value = null;
  previewUrl.value = null;
  resetResult();
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

async function startOcr() {
  const file = selectedFile.value;
  if (!file || !canStartOcr.value) {
    if (file && !isDesktopRuntime) {
      setErrorState("OCR 后端只在 Tauri 桌面端可用。");
    }
    return;
  }

  const controller = new AbortController();
  activeController.value = controller;
  resetResult();
  workflowState.value = "processing";
  processingProgress.value = 0;
  statusMessage.value = "正在准备 OCR。";
  clearProgressListener();
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

    const result: OcrResult = await ocrProvider.recognize(
      { file, requestId },
      controller.signal,
    );
    if (controller.signal.aborted || activeController.value !== controller) {
      return;
    }

    resultText.value = result.text;
    resultMarkdown.value = result.markdown;
    annotatedResultUrl.value = result.annotatedImageDataUrl;
    providerLabel.value = result.providerLabel;
    durationMs.value = result.durationMs;
    resultRegions.value = result.regions;
    resultImageWidth.value = result.imageWidth;
    resultImageHeight.value = result.imageHeight;
    processingProgress.value = 100;
    workflowState.value = "result";
    setStatusToast("success", "OCR 识别结果已准备好。");
  } catch (error) {
    if (controller.signal.aborted || isTranslationCancellation(error)) {
      if (activeController.value === controller) {
        workflowState.value = "cancelled";
        statusMessage.value = "OCR 已取消，图片预览仍可用。";
      }
      return;
    }
    if (activeController.value !== controller) {
      return;
    }
    setErrorState(error instanceof Error ? error.message : "OCR 未完成，请检查模型配置后重试。");
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

function cancelOcr() {
  if (!activeController.value) {
    return;
  }
  activeController.value.abort();
  clearProgressListener();
  processingProgress.value = 0;
  workflowState.value = "cancelled";
  setStatusToast("warning", "OCR 已取消，图片预览仍可用。");
}

function resetWorkflow() {
  selectionVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  releaseImagePreview(previewUrl.value);
  selectedFile.value = null;
  previewUrl.value = null;
  resetResult();
  processingProgress.value = 0;
  workflowState.value = "idle";
  statusMessage.value = "请选择一张图片开始 OCR。";
  notify("info", "OCR 工作区已重置。");
}

async function copyResult() {
  if (!activeOutput.value) {
    return;
  }
  try {
    await copyTranslationText(activeOutput.value);
    setActionFeedback("success", outputMode.value === "markdown" ? "Markdown 已复制到剪贴板。" : "识别文本已复制到剪贴板。");
  } catch {
    setActionFeedback("error", "剪贴板不可用，请手动选择文本进行复制。");
  }
}

function saveResult() {
  if (!activeOutput.value || !selectedFile.value) {
    return;
  }
  try {
    saveTranslationText(activeOutput.value, selectedFile.value.name);
    setActionFeedback("success", "OCR 结果已保存为文本文件。");
  } catch {
    setActionFeedback("error", "OCR 结果无法保存，请重试。");
  }
}

function saveAnnotatedImage() {
  if (!annotatedResultUrl.value || !selectedFile.value) {
    return;
  }
  try {
    saveImageDataUrl(annotatedResultUrl.value, selectedFile.value.name);
    setActionFeedback("success", "OCR 标注图片已保存。");
  } catch {
    setActionFeedback("error", "OCR 标注图片无法保存，请重试。");
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
    canvas.toBlob((value) => (value ? resolve(value) : reject(new Error("无法编码剪贴板图片。"))), "image/png");
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
  const types = Array.from(event.clipboardData?.types ?? []);
  const mayContainImage =
    types.length === 0 || types.includes("Files") || types.some((type) => type.startsWith("image/"));
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
});

onDeactivated(() => {
  window.removeEventListener("paste", handlePaste);
});

onBeforeUnmount(() => {
  selectionVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  window.removeEventListener("paste", handlePaste);
  releaseImagePreview(previewUrl.value);
});
</script>

<template>
  <section class="ocr-page" aria-label="OCR 识别工作区">
    <p class="one-shot-context">适合只从图片提取文字，不进行翻译。</p>
    <p class="sr-only" aria-live="polite">{{ statusMessage }}</p>
    <div class="workflow-grid ocr-page-grid">
      <n-card class="panel ocr-page-card ocr-input-card" :bordered="false">
        <div class="panel-heading ocr-card-heading">
          <div class="panel-title ocr-card-title">
            <span class="section-number ocr-section-number">01</span>
            <div>
              <p class="panel-kicker ocr-panel-kicker">输入 / 图片</p>
              <h2>选择图片</h2>
              <p class="panel-copy ocr-panel-copy">识别图片中的文字，不执行模型翻译。</p>
            </div>
          </div>
          <n-tag class="ocr-state-tag" :type="workflowTagType" round size="small">{{ workflowStatus }}</n-tag>
        </div>

        <div
          v-if="!selectedFile"
          class="drop-zone ocr-drop-zone"
          :class="{ 'ocr-drop-zone-active': isDragActive, 'drop-zone-active': isDragActive }"
          role="button"
          tabindex="0"
          aria-label="选择 OCR 图片文件"
          @click="openFilePicker"
          @keydown.enter.prevent="openFilePicker"
          @keydown.space.prevent="openFilePicker"
          @dragover="handleDragOver"
          @dragleave="handleDragLeave"
          @drop="handleDrop"
        >
          <input
            id="ocr-image-file"
            ref="fileInput"
            class="file-input ocr-file-input"
            type="file"
            :accept="SUPPORTED_IMAGE_EXTENSIONS.map((extension) => `.${extension}`).join(',')"
            aria-label="选择 OCR 图片文件"
            @change="handleFileInput"
          />
          <div class="drop-zone-label ocr-drop-label">
            <span class="drop-icon ocr-drop-icon" aria-hidden="true">
              <svg viewBox="0 0 32 32" fill="none">
                <rect x="4.5" y="5.5" width="23" height="21" rx="3" stroke="currentColor" stroke-width="1.5" />
                <circle cx="11" cy="12" r="2" stroke="currentColor" stroke-width="1.5" />
                <path d="m7 23 6.2-6 4.3 4 3-3 4.5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
            </span>
            <span class="drop-zone-kicker ocr-drop-kicker">拖放图片以识别</span>
            <span class="drop-zone-copy ocr-drop-copy">或从设备选择文件，或按 Ctrl+V 粘贴图片</span>
            <n-button text type="primary" size="small" class="drop-zone-action ocr-drop-action" @click.stop="openFilePicker">
              浏览文件 <span aria-hidden="true">↗</span>
            </n-button>
          </div>
        </div>

        <div v-else class="preview-layout ocr-preview-layout">
          <ImagePreviewFrame
            title="输入预览"
            state-label="已加载"
            :src="previewUrl ?? undefined"
            :preview-src="previewUrl ?? undefined"
            :alt="`OCR 预览：${selectedFile.name}`"
            variant="input"
            :render-toolbar="renderImageToolbar"
            @error="handlePreviewError"
          />
          <div class="preview-details ocr-preview-details">
            <div class="ocr-file-identity">
              <span class="ocr-file-type-mark" aria-hidden="true">IMG</span>
              <div>
                <p class="ocr-detail-label">已选文件</p>
                <h3>{{ selectedFile.name }}</h3>
                <p class="ocr-file-meta">{{ fileSizeLabel }} <span aria-hidden="true">·</span> {{ selectedFile.type || '图片文件' }}</p>
              </div>
            </div>
            <div class="button-row ocr-button-row">
              <n-button secondary @click="resetWorkflow">选择其他</n-button>
              <n-button
                v-if="workflowState !== 'processing'"
                type="primary"
                :disabled="!canStartOcr"
                @click="startOcr"
              >
                {{ workflowState === 'cancelled' || workflowState === 'error' ? '再次识别' : '开始 OCR' }}
              </n-button>
              <n-button v-else tertiary type="warning" @click="cancelOcr">取消 OCR</n-button>
            </div>
            <n-alert v-if="!isDesktopRuntime" class="inline-alert ocr-inline-alert" type="info" :show-icon="false">
              浏览器预览仅展示界面；OCR 需要在 Tauri 桌面端运行。
            </n-alert>
            <n-alert v-if="workflowState === 'error'" class="inline-alert ocr-inline-alert" type="error" title="图片无法识别" :show-icon="true">
              {{ errorMessage }}
            </n-alert>
          </div>
        </div>

        <n-alert v-if="workflowState === 'error' && !selectedFile" class="inline-alert ocr-inline-alert" type="error" title="图片无法使用" :show-icon="true">
          {{ errorMessage }}
        </n-alert>
        <p class="input-helper ocr-input-helper">
          支持：{{ SUPPORTED_IMAGE_EXTENSIONS.map((extension) => extension.toUpperCase()).join(' · ') }}
          <span aria-hidden="true"> / </span>
          最大 {{ MAX_IMAGE_BYTES / (1024 * 1024) }} MB · {{ MAX_IMAGE_PIXELS / 1000000 }} MP
        </p>
      </n-card>

      <n-card class="panel ocr-page-card ocr-output-card" :bordered="false">
        <div class="panel-heading ocr-card-heading">
          <div class="panel-title ocr-card-title">
            <span class="section-number ocr-section-number">02</span>
            <div>
              <p class="panel-kicker ocr-panel-kicker">输出 / 识别文本</p>
              <h2>OCR 结果</h2>
              <p class="panel-copy ocr-panel-copy">查看识别文本和结构化 Markdown。</p>
            </div>
          </div>
          <n-tag v-if="providerLabel" type="success" round size="small">{{ providerLabel }}</n-tag>
        </div>

        <div v-if="workflowState === 'processing'" class="processing-state ocr-processing-state" aria-busy="true">
          <div class="ocr-processing-visual" aria-hidden="true"><n-spin size="medium" /></div>
          <p class="ocr-detail-label">正在识别图片</p>
          <p class="ocr-status-copy">{{ statusMessage }}</p>
          <n-progress
            type="line"
            :percentage="processingProgress"
            :show-indicator="false"
            processing
            aria-label="OCR 进度"
            :aria-valuetext="`${statusMessage}，${processingProgress}% 完成`"
          />
          <div class="ocr-progress-meta"><span>{{ statusMessage }}</span><strong>{{ processingProgress }}%</strong></div>
        </div>

        <div v-else-if="workflowState === 'result' && resultText !== null" class="result-state ocr-result-state">
          <div
            v-if="annotatedResultUrl && resultImageWidth > 0 && resultImageHeight > 0"
            class="ocr-selectable-result"
          >
            <ImagePreviewFrame
              title="OCR 标注图片"
              state-label="打开大图后可选字"
              variant="result"
              :src="annotatedResultUrl"
              :preview-src="annotatedResultUrl"
              alt="可在放大预览中选择文字的 PP-OCR 标注识别图片"
              :render-toolbar="renderImageToolbar"
              :image-width="resultImageWidth"
              :image-height="resultImageHeight"
              :regions="resultRegions"
              @selection-change="selectedImageText = $event"
            >
              <template #actions>
                <n-button text size="small" @click="saveAnnotatedImage">保存 PNG</n-button>
              </template>
            </ImagePreviewFrame>
            <p class="ocr-selection-hint" aria-live="polite">{{ imageSelectionHint }}</p>
          </div>
          <div class="result-toolbar ocr-result-toolbar">
            <span>识别输出</span>
            <div class="ocr-result-toolbar-meta">
              <span>{{ providerLabel || 'PP-OCR' }}</span>
              <span v-if="durationLabel">处理用时 {{ durationLabel }}</span>
            </div>
          </div>
          <n-tabs v-model:value="outputMode" type="line" size="small" animated>
            <n-tab-pane name="text" tab="识别文本">
              <n-input
                class="ocr-result-input"
                type="textarea"
                :value="resultText"
                readonly
                :autosize="{ minRows: 8, maxRows: 16 }"
                aria-label="OCR 识别文本"
              />
            </n-tab-pane>
            <n-tab-pane name="markdown" tab="Markdown">
              <n-input
                class="ocr-result-input"
                type="textarea"
                :value="resultMarkdown ?? ''"
                readonly
                :autosize="{ minRows: 8, maxRows: 16 }"
                aria-label="OCR Markdown 输出"
              />
            </n-tab-pane>
          </n-tabs>
          <div class="ocr-result-actions">
            <n-space :size="10" wrap>
              <n-button secondary @click="copyResult">复制内容</n-button>
              <n-button type="primary" @click="saveResult">保存文本</n-button>
            </n-space>
            <p v-if="actionFeedback" class="ocr-action-feedback" aria-live="polite">{{ actionFeedback }}</p>
          </div>
        </div>

        <div v-else-if="workflowState === 'cancelled'" class="output-message ocr-output-message">
          <n-alert class="inline-alert ocr-inline-alert" type="warning" title="OCR 已取消" :show-icon="true">
            图片预览仍然保留，可以再次开始识别。
          </n-alert>
          <n-button type="primary" :disabled="!canStartOcr" @click="startOcr">再次识别</n-button>
        </div>

        <div v-else-if="workflowState === 'error'" class="output-message ocr-output-message">
          <n-alert class="inline-alert ocr-inline-alert" type="error" title="OCR 未完成" :show-icon="true">
            {{ errorMessage }}
          </n-alert>
          <n-button type="primary" :disabled="!canStartOcr" @click="startOcr">再次识别</n-button>
        </div>

        <div v-else class="empty-output ocr-empty-output">
          <n-empty size="small" description="OCR 结果将在此处显示。" />
        </div>
      </n-card>
    </div>

    <footer class="ocr-page-footer">
      <span>PP-OCR 本地识别</span>
      <span class="ocr-footer-separator" aria-hidden="true"></span>
      <span>Candle · Tauri</span>
      <span class="ocr-footer-spacer"></span>
      <span>{{ statusMessage }}</span>
    </footer>
  </section>
</template>

<style scoped>
.one-shot-context {
  margin: 0 0 12px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.ocr-page {
  width: 100%;
  margin-top: 24px;
}

.ocr-page-grid {
  display: grid;
  width: 100%;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.ocr-page-card {
  min-height: max(520px, calc(100dvh - 360px));
}

.ocr-page-card :deep(.n-card__content) {
  display: flex;
  min-height: inherit;
  flex-direction: column;
  padding: 20px;
}

.ocr-card-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.ocr-card-title {
  display: flex;
  min-width: 0;
  gap: 12px;
}

.ocr-section-number {
  display: grid;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--green);
  font-size: 12px;
  font-weight: 600;
}

.ocr-panel-kicker,
.ocr-detail-label {
  margin: 0 0 4px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}

.ocr-card-heading h2 {
  margin: 0;
  color: var(--text);
  font-size: 16px;
  font-weight: 650;
  line-height: 1.35;
}

.ocr-panel-copy {
  max-width: 340px;
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.ocr-drop-zone {
  position: relative;
  display: flex;
  min-height: clamp(260px, 38dvh, 420px);
  flex: 1;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  background: var(--surface-soft);
  cursor: pointer;
  transition: border-color 180ms ease, background-color 180ms ease, box-shadow 180ms ease;
}

.ocr-drop-zone:hover,
.ocr-drop-zone-active {
  border-color: var(--green);
  background: #ecf5ff;
  box-shadow: inset 0 0 0 1px rgba(64, 158, 255, 0.12);
}

.ocr-drop-label {
  display: flex;
  max-width: 280px;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 24px;
  cursor: pointer;
  text-align: center;
}

.ocr-drop-icon {
  display: grid;
  width: 48px;
  height: 48px;
  margin-bottom: 6px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--green);
  background: #f5f7fa;
}

.ocr-drop-icon svg {
  width: 27px;
  height: 27px;
}

.ocr-drop-kicker {
  color: var(--text);
  font-size: 14px;
  font-weight: 500;
}

.ocr-drop-copy,
.ocr-input-helper {
  color: var(--text-muted);
  font-size: 12px;
}

.ocr-drop-action {
  margin-top: 8px;
}

.ocr-file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  clip-path: inset(50%);
}

.ocr-preview-layout {
  display: grid;
  flex: 1;
  grid-template-rows: minmax(230px, 1fr) auto;
  gap: 16px;
}

.ocr-preview-details {
  display: grid;
  gap: 14px;
}

.ocr-file-identity {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 12px;
}

.ocr-file-type-mark {
  display: grid;
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--green);
  font-size: 10px;
  font-weight: 700;
}

.ocr-file-identity h3 {
  max-width: 100%;
  margin: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 14px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.ocr-file-meta {
  margin: 4px 0 0;
  color: var(--text-muted);
  font-size: 12px;
}

.ocr-button-row {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.ocr-inline-alert {
  margin-top: 0;
}

.ocr-input-helper {
  margin: 14px 0 0;
  line-height: 1.5;
}

.ocr-processing-state,
.ocr-output-message,
.ocr-empty-output {
  display: flex;
  min-height: 360px;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.ocr-processing-state {
  align-items: stretch;
}

.ocr-processing-visual {
  display: grid;
  place-items: center;
  min-height: 56px;
}

.ocr-processing-state .ocr-detail-label,
.ocr-processing-state .ocr-status-copy {
  text-align: center;
}

.ocr-status-copy {
  margin: 0;
  color: var(--text-soft);
  font-size: 13px;
}

.ocr-progress-meta,
.ocr-result-toolbar,
.ocr-result-toolbar-meta {
  display: flex;
  align-items: center;
}

.ocr-progress-meta {
  justify-content: space-between;
  gap: 12px;
  color: var(--text-muted);
  font-size: 12px;
}

.ocr-progress-meta strong {
  color: var(--green);
  font-variant-numeric: tabular-nums;
}

.ocr-result-state {
  display: flex;
  min-height: 360px;
  flex: 1;
  flex-direction: column;
}

.ocr-selectable-result {
  display: grid;
  gap: 7px;
}

.ocr-selection-hint {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.ocr-result-toolbar {
  justify-content: space-between;
  gap: 12px;
  margin: 12px 0 8px;
  padding: 0 1px 8px;
  border-bottom: 1px solid var(--divider);
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 600;
}

.ocr-result-toolbar-meta {
  gap: 12px;
  color: var(--text-muted);
  font-weight: 400;
}

.ocr-result-input {
  width: 100%;
  margin-top: 8px;
}

.ocr-result-input :deep(textarea) {
  min-height: 170px !important;
  font-family: inherit;
  font-size: 13px;
  line-height: 1.55;
}

.ocr-result-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 14px;
}

.ocr-action-feedback {
  margin: 0;
  color: var(--green);
  font-size: 12px;
  line-height: 1.5;
  text-align: right;
}

.ocr-output-message {
  align-items: flex-start;
}

.ocr-output-message .ocr-inline-alert {
  width: 100%;
}

.ocr-page-footer {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.ocr-footer-separator {
  width: 4px;
  height: 4px;
  flex: 0 0 4px;
  border-radius: 50%;
  background: #c0c4cc;
}

.ocr-footer-spacer {
  flex: 1;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
  clip-path: inset(50%);
}

@media (max-width: 900px) {
  .ocr-page-grid {
    grid-template-columns: 1fr;
  }

  .ocr-page-card {
    min-height: 0;
  }

  .ocr-drop-zone,
  .ocr-processing-state,
  .ocr-result-state,
  .ocr-output-message,
  .ocr-empty-output {
    min-height: clamp(220px, 34dvh, 320px);
  }
}

@media (max-width: 560px) {
  .ocr-card-heading {
    flex-direction: column;
    gap: 10px;
  }

  .ocr-state-tag {
    align-self: flex-start;
  }

  .ocr-button-row,
  .ocr-result-actions,
  .ocr-page-footer {
    align-items: flex-start;
    flex-direction: column;
  }

  .ocr-action-feedback {
    text-align: left;
  }

  .ocr-footer-spacer {
    display: none;
  }
}
</style>
