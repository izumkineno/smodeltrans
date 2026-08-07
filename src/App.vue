<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, ref, type CSSProperties } from "vue";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { readImage } from "@tauri-apps/plugin-clipboard-manager";
import { open as openNativeDialog } from "@tauri-apps/plugin-dialog";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  lightTheme,
  NAlert,
  NButton,
  NCard,
  NConfigProvider,
  NEmpty,
  NGlobalStyle,
  NIcon,
  NInput,
  NInputNumber,
  NLayout,
  NLayoutContent,
  NLayoutSider,
  NMenu,
  NProgress,
  NSpin,
  NSpace,
  NTag,
  type ImageRenderToolbarProps,
  type GlobalThemeOverrides,
  type MenuOption,
  zhCN,
} from "naive-ui";
import {
  copyTranslationText,
  createImagePreview,
  MAX_IMAGE_BYTES,
  MAX_IMAGE_PIXELS,
  releaseImagePreview,
  saveTranslationText,
  saveImageDataUrl,
  SUPPORTED_IMAGE_EXTENSIONS,
  validateImageFile,
  validateImagePreview,
} from "./services/file-adapter";
import ImagePreviewFrame from "./components/ImagePreviewFrame.vue";
import {
  createTranslationRequestId,
  getBackendStatus,
  isTranslationCancellation,
  translationProvider,
  updateBackendSettings,
  type BackendSettingsUpdate,
  type BackendStatus,
  type TranslationProgress,
} from "./services/translation-provider";

type WorkflowState = "idle" | "preview" | "processing" | "result" | "cancelled" | "error";
type TagType = "default" | "success" | "warning" | "error" | "info";

const themePalette = {
  appBg: "#f5f7fa",
  surface: "#ffffff",
  surfaceRaised: "#ffffff",
  surfaceSoft: "#f8fafc",
  border: "#dcdfe6",
  borderStrong: "#c0c4cc",
  text: "#303133",
  textSoft: "#606266",
  textMuted: "#909399",
  placeholder: "#a8abb2",
  divider: "#ebeef5",
  input: "#ffffff",
  progressRail: "#e4e7ed",
  primary: "#409eff",
  primaryHover: "#66b1ff",
  primaryPressed: "#3a8ee6",
  primarySuppl: "#79bbff",
  success: "#67c23a",
  successHover: "#85ce61",
  successPressed: "#5daf34",
  successSuppl: "#b3e19d",
  warning: "#e6a23c",
  error: "#f56c6c",
} as const;

const themeOverrides: GlobalThemeOverrides = {
  common: {
    baseColor: "#ffffff",
    primaryColor: themePalette.primary,
    primaryColorHover: themePalette.primaryHover,
    primaryColorPressed: themePalette.primaryPressed,
    primaryColorSuppl: themePalette.primarySuppl,
    successColor: themePalette.success,
    successColorHover: themePalette.successHover,
    successColorPressed: themePalette.successPressed,
    successColorSuppl: themePalette.successSuppl,
    warningColor: themePalette.warning,
    errorColor: themePalette.error,
    textColorBase: themePalette.text,
    textColor1: themePalette.text,
    textColor2: themePalette.textSoft,
    textColor3: themePalette.textMuted,
    placeholderColor: themePalette.placeholder,
    dividerColor: themePalette.divider,
    borderColor: themePalette.border,
    cardColor: themePalette.surface,
    modalColor: themePalette.surfaceRaised,
    popoverColor: themePalette.surfaceRaised,
    bodyColor: themePalette.appBg,
    inputColor: themePalette.input,
    progressRailColor: themePalette.progressRail,
    railColor: themePalette.progressRail,
    fontFamily:
      '"Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", ui-sans-serif, system-ui, sans-serif',
    fontFamilyMono: 'Consolas, "Cascadia Code", ui-monospace, monospace',
    borderRadius: "4px",
    borderRadiusSmall: "3px",
  },
  Button: {
    heightMedium: "36px",
    borderRadiusMedium: "4px",
    fontSizeMedium: "14px",
    fontWeight: "500",
    fontWeightStrong: "600",
  },
  Card: {
    color: themePalette.surface,
    colorEmbedded: themePalette.surfaceSoft,
    borderColor: themePalette.divider,
    borderRadius: "4px",
    paddingMedium: "16px",
    boxShadow: "0 2px 12px rgba(0, 0, 0, 0.04)",
  },
  Empty: {
    fontSizeSmall: "12px",
    iconSizeSmall: "16px",
    textColor: themePalette.textMuted,
    iconColor: themePalette.textMuted,
    extraTextColor: themePalette.textMuted,
  },
  Menu: {
    color: "#0000",
    borderRadius: "4px",
    fontSize: "14px",
    itemHeight: "40px",
    itemTextColor: themePalette.textSoft,
    itemTextColorHover: themePalette.text,
    itemTextColorActive: themePalette.primary,
    itemTextColorActiveHover: themePalette.primary,
    itemIconColor: themePalette.textMuted,
    itemIconColorHover: themePalette.text,
    itemIconColorActive: themePalette.primary,
    itemIconColorActiveHover: themePalette.primary,
    itemColorHover: "#f5f7fa",
    itemColorActive: "#ecf5ff",
    itemColorActiveHover: "#ecf5ff",
  },
  Tag: {
    fontSizeSmall: "12px",
    fontSizeMedium: "12px",
  },
  Input: {
    fontSizeSmall: "12px",
    fontSizeMedium: "14px",
  },
  Alert: {
    fontSize: "14px",
    borderRadius: "4px",
  },
};

const appThemeStyle = {
  "--app-bg": themePalette.appBg,
  "--surface": themePalette.surface,
  "--surface-raised": themePalette.surfaceRaised,
  "--surface-soft": themePalette.surfaceSoft,
  "--border": themePalette.border,
  "--border-strong": themePalette.borderStrong,
  "--divider": themePalette.divider,
  "--text": themePalette.text,
  "--text-soft": themePalette.textSoft,
  "--text-muted": themePalette.textMuted,
  "--green": themePalette.primary,
  "--green-soft": themePalette.primarySuppl,
} as CSSProperties;

const activeMenu = ref("translate");

function renderTranslationIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("path", {
            d: "M4 5.5h7M4 10h12M4 14.5h7",
            stroke: "currentColor",
            "stroke-width": "1.4",
            "stroke-linecap": "round",
          }),
          h("path", {
            d: "m13 4 3 1.5-3 1.5M11 13l-3 1.5 3 1.5",
            stroke: "currentColor",
            "stroke-width": "1.4",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
        ]),
    },
  );
}

function renderSettingsIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("path", {
            d: "M8.7 2.2h2.6l.5 1.8a6.7 6.7 0 0 1 1.3.8l1.8-.5 1.3 2.2-1.3 1.3c.1.5.2.9.2 1.4s-.1.9-.2 1.4l1.3 1.3-1.3 2.2-1.8-.5a6.7 6.7 0 0 1-1.3.8l-.5 1.8H8.7l-.5-1.8a6.7 6.7 0 0 1-1.3-.8l-1.8.5-1.3-2.2 1.3-1.3a6.7 6.7 0 0 1-.2-1.4c0-.5.1-.9.2-1.4L3.8 6.5l1.3-2.2 1.8.5a6.7 6.7 0 0 1 1.3-.8l.5-1.8Z",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linejoin": "round",
          }),
          h("circle", {
            cx: "10",
            cy: "10",
            r: "2.2",
            stroke: "currentColor",
            "stroke-width": "1.2",
          }),
        ]),
    },
  );
}

const menuOptions: MenuOption[] = [
  {
    label: "翻译",
    key: "translate",
    icon: renderTranslationIcon,
  },
  {
    label: "设置",
    key: "settings",
    icon: renderSettingsIcon,
  },
];

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
const targetLanguage = ref("Chinese");
const modelDetectorPath = ref("");
const modelRecognizerPath = ref("");
const modelHyPath = ref("");
const idleUnloadMinutes = ref(30);
const settingsMessage = ref("");
const settingsLoading = ref(false);
const isDragActive = ref(false);
const processingProgress = ref(0);
const fileInput = ref<HTMLInputElement | null>(null);
const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const appWindow = isDesktopRuntime ? getCurrentWindow() : null;
const isWindowMaximized = ref(false);
const activeController = ref<AbortController | null>(null);

let progressUnlisten: UnlistenFn | undefined;
const backendStatus = ref<BackendStatus | null>(null);
let selectionVersion = 0;

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

const settingsTagType = computed<TagType>(() => {
  if (!backendStatus.value) {
    return "default";
  }
  return backendStatus.value.ready ? "success" : "warning";
});

const settingsStatusLabel = computed(() => {
  if (!backendStatus.value) {
    return isDesktopRuntime ? "读取中" : "浏览器预览";
  }
  return backendStatus.value.ready ? "模型已就绪" : "需要检查";
});

const canStartTranslation = computed(
  () =>
    selectedFile.value !== null &&
    (workflowState.value === "preview" ||
      workflowState.value === "cancelled" ||
      workflowState.value === "error"),
);

const startButtonLabel = computed(() => {
  if (workflowState.value === "cancelled" || workflowState.value === "error") {
    return "再次尝试";
  }
  return "开始翻译";
});

const fileSizeLabel = computed(() => {
  if (!selectedFile.value) {
    return "";
  }

  if (selectedFile.value.size < 1024) {
    return `${selectedFile.value.size} B`;
  }

  if (selectedFile.value.size < 1024 * 1024) {
    return `${(selectedFile.value.size / 1024).toFixed(1)} KB`;
  }

  return `${(selectedFile.value.size / (1024 * 1024)).toFixed(1)} MB`;
});

const translationDurationLabel = computed(() => {
  const durationMs = translationDurationMs.value;
  if (durationMs === null || !Number.isFinite(durationMs) || durationMs < 0) {
    return "";
  }
  if (durationMs < 1000) {
    return `${Math.max(1, Math.round(durationMs))} ms`;
  }
  return `${(durationMs / 1000).toFixed(2)} 秒`;
});

async function syncWindowState() {
  if (!appWindow) {
    return;
  }

  try {
    isWindowMaximized.value = await appWindow.isMaximized();
  } catch {
    // Window state is optional in the browser preview.
  }
}

function loadPersistedSettings() {
  if (typeof window === "undefined") {
    return;
  }
  try {
    const persistedLanguage = window.localStorage.getItem("smodeltrans.targetLanguage");
    if (persistedLanguage?.trim()) {
      targetLanguage.value = persistedLanguage.trim();
    }
  } catch {
    settingsMessage.value = "无法读取本地设置，将使用默认目标语言。";
  }
}

function applyBackendStatus(status: BackendStatus) {
  backendStatus.value = status;
  modelDetectorPath.value = status.detectorModelDir;
  modelRecognizerPath.value = status.recognizerModelDir;
  modelHyPath.value = status.hyModel;
  idleUnloadMinutes.value = status.idleUnloadMinutes;
}

async function refreshBackendStatus() {
  if (!isDesktopRuntime) {
    settingsMessage.value = "设置状态仅在 Tauri 桌面端可读取。";
    return;
  }
  settingsLoading.value = true;
  try {
    const status = await getBackendStatus();
    applyBackendStatus(status);
    settingsMessage.value = status.message;
  } catch (error) {
    settingsMessage.value =
      error instanceof Error ? error.message : "无法读取后端模型状态。";
  } finally {
    settingsLoading.value = false;
  }
}

type ModelPathField = "detector" | "recognizer" | "hy";

function modelPathFor(field: ModelPathField): string {
  if (field === "detector") {
    return modelDetectorPath.value;
  }
  if (field === "recognizer") {
    return modelRecognizerPath.value;
  }
  return modelHyPath.value;
}

async function chooseModelPath(field: ModelPathField) {
  if (!isDesktopRuntime) {
    settingsMessage.value = "模型路径选择仅在 Tauri 桌面端可用。";
    return;
  }
  try {
    const currentPath = modelPathFor(field);
    const selected =
      field === "hy"
        ? await openNativeDialog({
            title: "选择 Hy-MT2 GGUF 模型",
            defaultPath: currentPath || undefined,
            multiple: false,
            filters: [{ name: "GGUF 模型", extensions: ["gguf"] }],
          })
        : await openNativeDialog({
            title: field === "detector" ? "选择 PP-OCRv5 detector 文件夹" : "选择 PP-OCRv5 recognizer 文件夹",
            defaultPath: currentPath || undefined,
            directory: true,
            multiple: false,
          });
    if (typeof selected !== "string" || !selected.trim()) {
      return;
    }
    if (field === "detector") {
      modelDetectorPath.value = selected;
    } else if (field === "recognizer") {
      modelRecognizerPath.value = selected;
    } else {
      modelHyPath.value = selected;
    }
    settingsMessage.value = "路径已选择，点击保存模型设置后生效。";
  } catch (error) {
    settingsMessage.value =
      error instanceof Error ? error.message : "无法打开模型路径选择器。";
  }
}

async function saveModelSettings() {
  if (!isDesktopRuntime) {
    settingsMessage.value = "模型设置仅在 Tauri 桌面端可保存。";
    return;
  }
  const idleMinutes = idleUnloadMinutes.value ?? 0;
  if (
    !Number.isInteger(idleMinutes) ||
    idleMinutes < 0 ||
    idleMinutes > 24 * 60
  ) {
    settingsMessage.value = "模型空闲释放时间必须为 0 到 1440 分钟。";
    return;
  }
  const settings: BackendSettingsUpdate = {
    detectorModelDir: modelDetectorPath.value.trim(),
    recognizerModelDir: modelRecognizerPath.value.trim(),
    hyModel: modelHyPath.value.trim(),
    idleUnloadMinutes: idleMinutes,
  };
  if (!settings.detectorModelDir || !settings.recognizerModelDir || !settings.hyModel) {
    settingsMessage.value = "请选择完整的 PP-OCRv5 与 Hy-MT2 模型路径。";
    return;
  }
  settingsLoading.value = true;
  try {
    const status = await updateBackendSettings(settings);
    applyBackendStatus(status);
    settingsMessage.value =
      idleMinutes === 0
        ? "模型设置已保存，自动释放已关闭。"
        : `模型设置已保存，空闲 ${idleMinutes} 分钟后释放显存模型。`;
  } catch (error) {
    settingsMessage.value =
      error instanceof Error ? error.message : "无法保存模型设置。";
  } finally {
    settingsLoading.value = false;
  }
}

function saveSettings() {
  const nextLanguage = targetLanguage.value.trim();
  if (nextLanguage.length < 1 || nextLanguage.length > 64) {
    settingsMessage.value = "目标语言长度必须为 1 到 64 个字符。";
    return;
  }
  targetLanguage.value = nextLanguage;
  try {
    window.localStorage.setItem("smodeltrans.targetLanguage", nextLanguage);
    settingsMessage.value = "翻译设置已保存。";
  } catch {
    settingsMessage.value = "无法写入本地设置，请检查应用存储权限。";
  }
}

function minimizeWindow() {
  if (!appWindow) {
    return;
  }

  void appWindow.minimize().catch(() => undefined);
}

function toggleWindowMaximize() {
  if (!appWindow) {
    return;
  }

  void appWindow
    .toggleMaximize()
    .then(syncWindowState)
    .catch(() => undefined);
}

function closeWindow() {
  if (!appWindow) {
    return;
  }

  void appWindow.close().catch(() => undefined);
}

function handleTitlebarMouseDown(event: MouseEvent) {
  if (!appWindow || event.button !== 0) {
    return;
  }

  const target = event.target as HTMLElement | null;
  if (target?.closest("button")) {
    return;
  }

  void appWindow.startDragging().catch(() => undefined);
}

function handleTitlebarDoubleClick(event: MouseEvent) {
  const target = event.target as HTMLElement | null;
  if (target?.closest("button")) {
    return;
  }

  toggleWindowMaximize();
}


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
    errorMessage.value = validationMessage;
    actionFeedback.value = "";
    workflowState.value = "error";
    statusMessage.value = validationMessage;
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
    errorMessage.value = previewValidationMessage;
    actionFeedback.value = "";
    workflowState.value = "error";
    statusMessage.value = previewValidationMessage;
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
  statusMessage.value = `${file.name} 已准备好翻译。`;
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
  errorMessage.value = "无法显示这张图片，请选择其他文件。";
  actionFeedback.value = "";
  workflowState.value = "error";
  statusMessage.value = errorMessage.value;
}

function handleFileInput(event: Event) {
  const input = event.currentTarget as HTMLInputElement;
  selectFile(input.files?.[0]);
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
  selectFile(event.dataTransfer?.files[0]);
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
      progressUnlisten = await listen<TranslationProgress>(
        "translation-progress",
        (event) => {
          if (
            event.payload.requestId !== requestId ||
            activeController.value !== controller ||
            workflowState.value !== "processing"
          ) {
            return;
          }
          processingProgress.value = Math.min(
            100,
            Math.max(0, Math.round(event.payload.progress)),
          );
          statusMessage.value = event.payload.stage;
        },
      );
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
    statusMessage.value = translation.isTranslated
      ? "翻译结果已准备好。"
      : "PP-OCRv5 识别结果已准备好；Hy 翻译需要 CUDA。";
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

    workflowState.value = "error";
    errorMessage.value =
      error instanceof Error
        ? error.message
        : "Candle 图片翻译后端未能完成，请检查模型配置后重试。";
    statusMessage.value = errorMessage.value;
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
  statusMessage.value = "翻译已取消，图片预览仍可用。";
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
}

async function copyResult() {
  if (!resultText.value) {
    return;
  }

  try {
    await copyTranslationText(resultText.value);
    actionFeedback.value = "结果已复制到剪贴板。";
  } catch {
    actionFeedback.value = "剪贴板不可用，请手动选择文本进行复制。";
  }
}

function saveResult() {
  if (!resultText.value || !selectedFile.value) {
    return;
  }

  try {
    saveTranslationText(resultText.value, selectedFile.value.name);
    actionFeedback.value = "结果已保存为文本文件。";
  } catch {
    actionFeedback.value = "结果无法保存，请从结果面板重试。";
  }
}

function saveAnnotatedImage() {
  if (!annotatedResultUrl.value || !selectedFile.value) {
    return;
  }

  try {
    saveImageDataUrl(annotatedResultUrl.value, selectedFile.value.name);
    actionFeedback.value = "标注图片已保存。";
  } catch {
    actionFeedback.value = "标注图片无法保存，请重试。";
  }
}


function renderImageToolbar({ nodes }: ImageRenderToolbarProps) {
  const zoomOutNode = h(
    "span",
    { "data-image-preview-zoom": "out" },
    [nodes.zoomOut],
  );
  const zoomInNode = h(
    "span",
    { "data-image-preview-zoom": "in" },
    [nodes.zoomIn],
  );

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
    actionFeedback.value = "剪贴板中没有可读取的图片。";
  }
}
onMounted(() => {
  loadPersistedSettings();
  void syncWindowState();
  void refreshBackendStatus();
  window.addEventListener("paste", handlePaste);
  addImagePreviewWheelListener();
});
onBeforeUnmount(() => {
  selectionVersion += 1;
  activeController.value?.abort();
  clearProgressListener();
  window.removeEventListener("paste", handlePaste);
  removeImagePreviewWheelListener();
  releaseImagePreview(previewUrl.value);
});
</script>

<template>
  <n-config-provider :locale="zhCN" :theme="lightTheme" :theme-overrides="themeOverrides">
    <n-global-style />
    <a class="skip-link" href="#main-content">跳转到主要内容</a>

    <div class="app-shell" :style="appThemeStyle">
    <header
      class="titlebar"
      data-tauri-drag-region
      @mousedown="handleTitlebarMouseDown"
      @dblclick="handleTitlebarDoubleClick"
    >
      <div class="titlebar-brand" data-tauri-drag-region>
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M5 5h5v5H5zM14 5h5v5h-5zM5 14h5v5H5z" fill="currentColor" />
            <path d="M14 14h5M16.5 11.5V19M14 17h5" stroke="currentColor" stroke-width="1.7" />
          </svg>
        </span>
        <span class="titlebar-name">smodeltrans</span>
        <span class="titlebar-divider" aria-hidden="true">/</span>
        <span class="titlebar-context">图片翻译</span>
      </div>

      <div class="window-controls" aria-label="窗口控制">
        <n-button
          class="window-control"
          quaternary
          circle
          size="small"
          aria-label="最小化窗口"
          title="最小化"
          @click.stop="minimizeWindow"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M3 8h10" stroke="currentColor" stroke-width="1.2" />
          </svg>
        </n-button>
        <n-button
          class="window-control"
          quaternary
          circle
          size="small"
          :aria-label="isWindowMaximized ? '恢复窗口' : '最大化窗口'"
          :title="isWindowMaximized ? '恢复' : '最大化'"
          @click.stop="toggleWindowMaximize"
        >
          <svg v-if="!isWindowMaximized" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect x="3.25" y="3.25" width="9.5" height="9.5" stroke="currentColor" stroke-width="1.2" />
          </svg>
          <svg v-else viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M5.5 5.5h6v6h-6z" stroke="currentColor" stroke-width="1.2" />
            <path d="M4.5 10.5h-1v-7h7v1" stroke="currentColor" stroke-width="1.2" />
          </svg>
        </n-button>
        <n-button
          class="window-control window-control-close"
          quaternary
          circle
          size="small"
          aria-label="关闭窗口"
          title="关闭"
          @click.stop="closeWindow"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.2" />
          </svg>
        </n-button>
      </div>
    </header>

    <n-layout has-sider class="workspace-shell">
      <n-layout-sider class="sidebar" bordered :width="208" :native-scrollbar="false">
        <div class="sidebar-header">
          <p class="sidebar-kicker">工作区</p>
          <p class="sidebar-title">SmodelTrans</p>
          <p class="sidebar-subtitle">本地图片工作台</p>
        </div>

        <nav class="sidebar-nav" aria-label="主导航">
          <p class="nav-heading">工作台</p>
          <n-menu v-model:value="activeMenu" :icon-size="16" :options="menuOptions" />
        </nav>

        <div class="sidebar-bottom">
          <n-card class="provider-card" :bordered="false" size="small">
            <span class="provider-indicator" aria-hidden="true"></span>
            <div>
              <strong>本地模型</strong>
              <span>{{ settingsStatusLabel }}</span>
            </div>
          </n-card>
          <p class="sidebar-build">ver. 0.1.0</p>
        </div>
      </n-layout-sider>

      <n-layout class="workspace-main">
        <n-layout-content id="main-content" class="content">
          <section
            class="workspace-header"
            :aria-labelledby="activeMenu === 'translate' ? 'page-title' : 'settings-title'"
          >
            <h1 :id="activeMenu === 'translate' ? 'page-title' : 'settings-title'">
              {{ activeMenu === "translate" ? "图片翻译" : "设置" }}
            </h1>
            <n-tag
              class="state-tag"
              :type="activeMenu === 'translate' ? workflowTagType : settingsTagType"
              round
              size="small"
              :aria-label="activeMenu === 'translate' ? '流程状态' : '后端状态'"
            >
              {{ activeMenu === "translate" ? workflowStatus : settingsStatusLabel }}
            </n-tag>
          </section>

          <template v-if="activeMenu === 'translate'">

        <p class="sr-only" aria-live="polite">{{ statusMessage }}</p>

        <section class="workflow-grid" aria-label="图片翻译流程">
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
                  <n-button v-else tertiary type="warning" @click="cancelTranslation">
                    取消翻译
                  </n-button>
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
              <n-alert
                class="inline-alert"
                type="error"
                title="本次翻译未完成"
                :show-icon="true"
              >
                {{ errorMessage }}
              </n-alert>
              <n-button v-if="selectedFile" type="primary" @click="startTranslation">再次尝试</n-button>
              <n-button v-else secondary @click="openFilePicker">选择图片</n-button>
            </div>

            <div v-else-if="workflowState === 'cancelled'" class="output-message">
              <n-alert
                class="inline-alert"
                type="warning"
                title="流程已取消"
                :show-icon="true"
              >
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
          </template>
          <section v-else class="settings-page" aria-labelledby="settings-title">
            <div class="settings-grid">
              <n-card class="settings-card" :bordered="false">
                <div class="settings-card-heading">
                  <div>
                    <p class="panel-kicker">运行状态</p>
                    <h2>模型服务</h2>
                  </div>
                  <n-tag :type="settingsTagType" round size="small">{{ settingsStatusLabel }}</n-tag>
                </div>
                <p class="settings-card-copy">
                  这里显示后端实际读取到的设备与模型状态，不会伪造就绪结果。
                </p>
                <div class="settings-metrics">
                  <div>
                    <span>设备</span>
                    <strong>{{ backendStatus?.device ?? "未读取" }}</strong>
                  </div>
                  <div>
                    <span>翻译器</span>
                    <strong>{{ backendStatus?.translatorLoaded ? "已加载" : "按需加载" }}</strong>
                  </div>
                </div>
                <n-alert v-if="settingsMessage" class="settings-alert" type="info" :show-icon="false">
                  {{ settingsMessage }}
                </n-alert>
              </n-card>

              <n-card class="settings-card" :bordered="false">
                <div class="settings-card-heading">
                  <div>
                    <p class="panel-kicker">翻译参数</p>
                    <h2>目标语言</h2>
                  </div>
                </div>
                <p class="settings-card-copy">目标语言会保存到本地，并用于下一次翻译请求。</p>
                <label class="settings-field">
                  <span>语言名称</span>
                  <n-input
                    v-model:value="targetLanguage"
                    maxlength="64"
                    placeholder="例如：Chinese"
                    aria-label="目标语言"
                  />
                </label>
                <n-button type="primary" @click="saveSettings">保存设置</n-button>
              </n-card>

              <n-card class="settings-card settings-card-wide" :bordered="false">
                <div class="settings-card-heading">
                  <div>
                    <p class="panel-kicker">模型资源</p>
                    <h2>本地模型路径</h2>
                  </div>
                </div>
                <p class="settings-card-copy">
                  选择后端实际使用的 PP-OCRv5 文件夹与 Hy-MT2 GGUF 文件；保存后立即应用，下一次翻译会按新路径加载。
                </p>
                <dl class="settings-path-list">
                  <div>
                    <dt>PP-OCRv5 detector</dt>
                    <dd>
                      <span class="settings-path-value">{{ modelDetectorPath || "未读取" }}</span>
                      <n-button secondary size="small" @click="chooseModelPath('detector')">选择文件夹</n-button>
                    </dd>
                  </div>
                  <div>
                    <dt>PP-OCRv5 recognizer</dt>
                    <dd>
                      <span class="settings-path-value">{{ modelRecognizerPath || "未读取" }}</span>
                      <n-button secondary size="small" @click="chooseModelPath('recognizer')">选择文件夹</n-button>
                    </dd>
                  </div>
                  <div>
                    <dt>Hy-MT2</dt>
                    <dd>
                      <span class="settings-path-value">{{ modelHyPath || "未读取" }}</span>
                      <n-button secondary size="small" @click="chooseModelPath('hy')">选择 GGUF</n-button>
                    </dd>
                  </div>
                </dl>
                <div class="settings-card-actions">
                  <n-button secondary :loading="settingsLoading" @click="refreshBackendStatus">刷新状态</n-button>
                  <n-button type="primary" :loading="settingsLoading" @click="saveModelSettings">
                    保存模型设置
                  </n-button>
                </div>
              </n-card>

              <n-card class="settings-card settings-card-wide" :bordered="false">
                <div class="settings-card-heading">
                  <div>
                    <p class="panel-kicker">显存管理</p>
                    <h2>空闲模型释放</h2>
                  </div>
                  <n-tag type="info" round size="small">CUDA</n-tag>
                </div>
                <p class="settings-card-copy">
                  翻译完成后保持模型在显存中的时间。设置为 0 表示不自动释放；释放后下一次翻译会按需重新加载。
                </p>
                <label class="settings-field settings-number-field">
                  <span>空闲释放时间（分钟）</span>
                  <n-input-number
                    v-model:value="idleUnloadMinutes"
                    :min="0"
                    :max="1440"
                    :step="5"
                    aria-label="模型空闲释放时间"
                  />
                </label>
              </n-card>
            </div>
          </section>
        </n-layout-content>
      </n-layout>
      </n-layout>
  </div>
  </n-config-provider>
</template>

<style>
:root {
  color: #303133;
  background: #f5f7fa;
  color-scheme: light;
  font-family:
    "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", ui-sans-serif, system-ui, sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  --app-bg: #f5f7fa;
  --surface: #ffffff;
  --surface-raised: #ffffff;
  --surface-soft: #f8fafc;
  --border: #dcdfe6;
  --border-strong: #c0c4cc;
  --divider: #ebeef5;
  --text: #303133;
  --text-soft: #606266;
  --text-muted: #909399;
  --primary: #409eff;
  --primary-soft: #79bbff;
  --green: #409eff;
  --green-soft: #79bbff;
  --font-size-meta: 12px;
  --font-size-body: 14px;
  --font-size-heading: 16px;
}

* {
  box-sizing: border-box;
}

html,
body,
#app {
  min-width: 320px;
  min-height: 100%;
  margin: 0;
}

body {
  height: 100vh;
  overflow: hidden;
  background: var(--app-bg);
}

button,
input,
textarea {
  font: inherit;
}

button {
  cursor: pointer;
}

button:focus-visible,
[role="button"]:focus-visible,
a:focus-visible {
  outline: 2px solid var(--green);
}

.drop-zone:focus {
  outline: 2px solid var(--green);
  outline-offset: 3px;
}

::selection {
  color: #ffffff;
  background: var(--green);
}

.skip-link {
  position: fixed;
  z-index: 50;
  top: 12px;
  left: 12px;
  padding: 8px 12px;
  border-radius: 4px;
  color: #ffffff;
  background: var(--green);
  font-size: 14px;
  font-weight: 600;
  transform: translateY(-160%);
  transition: transform 180ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

.app-shell {
  display: flex;
  height: 100dvh;
  min-height: 560px;
  flex-direction: column;
  overflow: hidden;
  background: var(--app-bg);
}

.titlebar {
  display: flex;
  height: 44px;
  flex: 0 0 44px;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--divider);
  color: var(--text-soft);
  background: var(--surface);
  user-select: none;
  -webkit-app-region: drag;
}

.titlebar-brand {
  display: flex;
  min-width: 0;
  height: 100%;
  align-items: center;
  gap: 9px;
  padding: 0 16px;
  -webkit-app-region: drag;
}

.brand-mark {
  display: grid;
  width: 21px;
  height: 21px;
  flex: 0 0 21px;
  place-items: center;
  border: 1px solid rgba(64, 158, 255, 0.38);
  border-radius: 4px;
  color: var(--green);
  background: #ecf5ff;
}

.brand-mark svg {
  width: 15px;
  height: 15px;
}

.titlebar-name,
.titlebar-context,
.titlebar-divider {
  white-space: nowrap;
}

.titlebar-name {
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
}

.titlebar-divider {
  color: #c0c4cc;
  font-family: inherit;
  font-size: 12px;
}

.titlebar-context {
  overflow: hidden;
  color: var(--text-muted);
  font-family: inherit;
  font-size: 12px;
  letter-spacing: 0;
  text-overflow: ellipsis;
}

.window-controls {
  display: flex;
  height: 100%;
  -webkit-app-region: no-drag;
}

.titlebar .n-button.window-control {
  display: grid;
  width: 44px;
  min-width: 44px;
  height: 44px;
  min-height: 44px;
  padding: 0;
  place-items: center;
  border: 0;
  border-radius: 0;
  color: var(--text-muted);
  background: transparent;
  -webkit-app-region: no-drag;
}

.titlebar .n-button.window-control svg {
  width: 16px;
  height: 16px;
}

.titlebar .n-button.window-control:hover {
  color: var(--text);
  background: #f5f7fa;
}

.titlebar .n-button.window-control-close:hover {
  color: #f56c6c;
  background: #fef0f0;
}

.workspace-shell {
  display: flex;
  min-height: 0;
  flex: 1;
  background: var(--app-bg);
}

.workspace-main {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  background: var(--app-bg);
}

.workspace-main > .n-layout-scroll-container {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
}

.workspace-main > .n-layout-scroll-container > .content {
  flex: 1;
  min-height: 0;
}

.sidebar {
  display: flex;
  min-height: 0;
  flex: 0 0 208px;
  flex-direction: column;
  border-right: 1px solid var(--divider);
  background: var(--surface);
  overflow: hidden;
}

.sidebar .n-layout-sider__content {
  display: flex;
  min-height: 100%;
  flex-direction: column;
}

.sidebar > .n-scrollbar,
.sidebar > .n-scrollbar > .n-scrollbar-container,
.sidebar > .n-scrollbar > .n-scrollbar-container > .n-scrollbar-content {
  display: flex;
  height: 100%;
  min-height: 100%;
  flex-direction: column;
}

.sidebar > .n-scrollbar {
  min-height: 0;
  flex: 1;
}

.sidebar-header {
  padding: 24px 20px 22px;
}

.sidebar-kicker,
.nav-heading,
.sidebar-build {
  margin: 0;
  color: var(--text-muted);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0;
}

.sidebar-title {
  margin: 9px 0 0;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0;
}

.sidebar-subtitle {
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 12px;
}

.sidebar-nav {
  padding: 0 10px;
}

.sidebar-nav .n-menu {
  width: 100%;
  background: transparent;
}

.nav-heading {
  padding: 0 9px 8px;
}


.sidebar-bottom {
  margin-top: auto;
  padding: 16px 14px 18px;
}

.provider-card {
  flex: 0 0 auto;
  border: 1px solid var(--divider);
  border-radius: 4px;
  background: var(--surface);
  box-shadow: none;
}

.provider-card .n-card-content {
  display: flex;
  align-items: center;
  gap: 9px;
  padding: 11px;
}

.provider-card strong,
.provider-card span {
  display: block;
}

.provider-card strong {
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 650;
}

.provider-card div span {
  margin-top: 3px;
  color: var(--text-muted);
  font-size: 12px;
}

.provider-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  flex: 0 0 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 3px rgba(64, 158, 255, 0.14);
}

.sidebar-build {
  margin: 14px 2px 0;
  color: var(--text-muted);
  font-size: 12px;
  letter-spacing: 0;
}

.content {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: auto;
  padding: 24px clamp(24px, 3vw, 48px) 20px;
}

.workspace-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin: 0 auto;
  width: 100%;
  max-width: none;
}

.settings-page {
  width: 100%;
  max-width: none;
  margin: 24px auto 0;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.settings-card {
  min-width: 0;
  border: 1px solid var(--divider);
  border-radius: 6px;
  background: var(--surface);
}

.settings-card-wide {
  grid-column: 1 / -1;
}

.settings-card .n-card-content {
  display: flex;
  min-height: 190px;
  flex-direction: column;
  gap: 14px;
}

.settings-card-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.settings-card-heading h2 {
  margin: 6px 0 0;
  font-size: 16px;
}

.settings-card-copy {
  max-width: 620px;
  margin: 0;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.settings-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.settings-metrics div {
  padding: 12px;
  border: 1px solid var(--divider);
  border-radius: 4px;
  background: var(--surface-soft);
}

.settings-metrics span,
.settings-metrics strong {
  display: block;
}

.settings-metrics span {
  color: var(--text-muted);
  font-size: 12px;
}

.settings-metrics strong {
  margin-top: 6px;
  color: var(--text);
  font-size: 14px;
  font-weight: 650;
}

.settings-field {
  display: grid;
  gap: 7px;
  max-width: 360px;
  color: var(--text-soft);
  font-size: 13px;
  font-weight: 600;
}

.settings-alert {
  margin-top: auto;
}

.settings-path-list {
  display: grid;
  gap: 10px;
  margin: 0;
}

.settings-path-list > div {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--divider);
}

.settings-path-list > div:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.settings-path-list dt {
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 650;
}

.settings-path-list dd {
  min-width: 0;
  margin: 0;
  color: var(--text-muted);
  font-family: var(--font-mono, Consolas, monospace);
  font-size: 12px;
  overflow-wrap: anywhere;
}

.settings-path-list dd {
  display: flex;
  align-items: center;
  gap: 12px;
}

.settings-path-value {
  min-width: 0;
  flex: 1;
  overflow-wrap: anywhere;
}

.settings-card-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: auto;
}

.settings-number-field .n-input-number {
  width: 180px;
}

.breadcrumb,
.panel-kicker,
.detail-label,
.preview-frame-bar,
.result-toolbar,
.input-helper,
.workspace-footer,
.progress-meta {
  font-family: inherit;
  letter-spacing: 0;
  text-transform: none;
}

.breadcrumb {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.breadcrumb span {
  padding: 0 6px;
  color: #c0c4cc;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1,
h2,
h3 {
  color: var(--text);
}

h1 {
  margin-bottom: 0;
  margin-top: 0;
  font-family: inherit;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.5;
}


.workflow-grid {
  display: grid;
  width: 100%;
  max-width: none;
  margin: 24px auto 0;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.panel {
  min-height: max(360px, calc(100dvh - 360px));
}

.panel .n-card-content {
  display: flex;
  min-height: max(360px, calc(100dvh - 360px));
  flex-direction: column;
}

.panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.panel-title {
  display: flex;
  min-width: 0;
  gap: 12px;
}

.section-number {
  display: grid;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--green);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
}

.panel-kicker,
.detail-label {
  margin-bottom: 4px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}

.panel-heading h2 {
  margin-bottom: 0;
  font-size: 16px;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.35;
}

.panel-copy {
  max-width: 340px;
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.drop-zone {
  position: relative;
  display: flex;
  min-height: clamp(220px, 34dvh, 380px);
  flex: 1;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  background: var(--surface-soft);
  cursor: pointer;
  transition:
    border-color 180ms ease,
    background-color 180ms ease,
    box-shadow 180ms ease;
}

.drop-zone:hover,
.drop-zone-active {
  border-color: var(--green);
  background: #ecf5ff;
  box-shadow: inset 0 0 0 1px rgba(64, 158, 255, 0.12);
}

.drop-zone-label {
  display: flex;
  max-width: 260px;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 24px;
  cursor: pointer;
  text-align: center;
}

.drop-icon {
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

.drop-icon svg {
  width: 27px;
  height: 27px;
}

.drop-zone-kicker {
  color: var(--text);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0;
}

.drop-zone-copy {
  color: var(--text-muted);
  font-size: 12px;
}
.drop-zone-action {
  margin-top: 8px;
}

.file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  clip-path: inset(50%);
}

.preview-layout {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 14px;
}

.image-preview-frame {
  display: flex;
  min-height: 260px;
  flex: 1;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface-soft);
}
.result-preview-frame {
  display: flex;
  min-height: 150px;
  max-height: 240px;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface-soft);
}

.result-preview-canvas {
  display: grid;
  min-height: 0;
  flex: 1;
  place-items: center;
  padding: 8px;
  overflow: auto;
}

.result-image {
  display: block;
  width: 100%;
}

.input-image {
  display: block;
  width: auto;
  max-width: 100%;
  align-self: center;
}

.input-image img {
  display: block;
  width: auto;
  max-width: 100%;
  max-height: 290px;
  object-fit: contain;
  cursor: zoom-in;
}

.result-image img {
  display: block;
  width: auto;
  max-width: 100%;
  max-height: 190px;
  object-fit: contain;
  cursor: zoom-in;
}


.preview-frame-bar,
.result-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 32px;
  padding: 0 12px;
  border-bottom: 1px solid var(--divider);
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.preview-frame-state {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--green-soft);
}

.preview-frame-state .provider-indicator {
  width: 5px;
  height: 5px;
  flex-basis: 5px;
  box-shadow: none;
}

.preview-frame-actions {
  display: inline-flex;
  align-items: center;
  gap: 2px;
}



.preview-canvas {
  display: grid;
  min-height: 0;
  flex: 1;
  place-items: center;
  padding: 12px;
  background:
    linear-gradient(45deg, rgba(48, 49, 51, 0.035) 25%, transparent 25%),
    linear-gradient(-45deg, rgba(48, 49, 51, 0.035) 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, rgba(48, 49, 51, 0.035) 75%),
    linear-gradient(-45deg, transparent 75%, rgba(48, 49, 51, 0.035) 75%),
    #f5f7fa;
  background-position:
    0 0,
    0 8px,
    8px -8px,
    -8px 0;
  background-size: 16px 16px;
}

.preview-canvas img {
  display: block;
  max-width: 100%;
  max-height: 290px;
  object-fit: contain;
}

.preview-details {
  display: flex;
  flex-direction: column;
}

.file-identity {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.file-type-mark {
  display: grid;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  place-items: center;
  border: 1px solid #b3d8ff;
  border-radius: 4px;
  color: var(--green);
  background: #ecf5ff;
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
}

.file-identity-copy {
  min-width: 0;
}

.file-identity-copy .detail-label {
  margin-bottom: 3px;
}

.preview-details h3 {
  overflow: hidden;
  margin-bottom: 0;
  color: var(--text-soft);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  margin: 4px 0 0;
  color: var(--text-muted);
  font-family: inherit;
  font-size: 12px;
}

.button-row,
.result-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 9px;
}

.button-row {
  margin-top: 14px;
}

.inline-alert {
  margin-top: 14px;
  border-radius: 7px;
}

.input-helper {
  margin: 12px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.processing-state,
.result-state,
.output-message,
.empty-output {
  min-height: 280px;
  flex: 1;
}

.processing-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 14px;
}

.processing-visual {
  position: relative;
  display: grid;
  width: 52px;
  height: 52px;
  margin-bottom: 2px;
  place-items: center;
}

.processing-copy .detail-label {
  margin-bottom: 0;
}

.processing-copy p:last-child {
  max-width: 330px;
  margin: 8px 0 0;
  color: var(--text-soft);
  font-size: 14px;
  line-height: 1.5;
}

.progress-meta {
  display: flex;
  justify-content: space-between;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.progress-meta strong {
  color: var(--green-soft);
  font-weight: 700;
}

.result-state {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-toolbar {
  min-height: auto;
  padding: 0 1px 8px;
  border-bottom: 1px solid var(--divider);
}

.result-toolbar-meta {
  display: inline-flex;
  align-items: center;
  gap: 12px;
}

.result-duration {
  color: var(--text-soft);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.result-mode-note {
  margin: -2px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.result-input textarea {
  min-height: 210px !important;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
}

.result-actions {
  justify-content: space-between;
  gap: 10px;
}

.settings-card .n-button {
  align-self: flex-start;
}


.action-feedback {
  margin: 0;
  color: var(--green);
  font-size: 12px;
  line-height: 1.5;
  text-align: right;
}

.output-message {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  gap: 14px;
}

.output-message .inline-alert {
  width: 100%;
  margin-top: 0;
}

.empty-output {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 20px;
  border: 1px dashed var(--border);
  border-radius: 4px;
  color: var(--text-muted);
  text-align: center;
  background: var(--surface-soft);
}



.app-shell .n-button {
  cursor: pointer;
}

.app-shell .n-button--disabled {
  cursor: not-allowed;
}

.workspace-footer {
  display: flex;
  width: 100%;
  max-width: none;
  align-items: center;
  gap: 10px;
  margin: 16px auto 0;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
  line-height: 1.5;
}

.footer-separator {
  width: 4px;
  height: 4px;
  flex: 0 0 4px;
  border-radius: 50%;
  background: #c0c4cc;
}

.footer-spacer {
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


@media (max-width: 1040px) {

  .content {
    padding-right: 24px;
    padding-left: 24px;
  }

  .panel .n-card-content {
    padding: 16px;
  }
}

@media (max-width: 900px) {
  .settings-grid {
    grid-template-columns: 1fr;
  }

  .settings-card-wide {
    grid-column: auto;
  }
  .workflow-grid {
    grid-template-columns: 1fr;
  }

  .panel,
  .panel .n-card-content {
    min-height: 0;
  }

  .drop-zone,
  .processing-state,
  .result-state,
  .output-message,
  .empty-output {
    min-height: clamp(220px, 34dvh, 320px);
  }

}

@media (max-width: 720px) {
  .settings-page {
    margin-top: 16px;
  }

  .settings-path-list > div {
    grid-template-columns: 1fr;
    gap: 4px;
  }
  .app-shell {
    min-height: 100dvh;
  }


  .sidebar {
    display: none;
  }

  .content {
    padding: 22px 16px 16px;
  }

  .workspace-header {
    align-items: center;
    flex-wrap: wrap;
  }

}

@media (max-width: 560px) {
  .titlebar-brand {
    padding-left: 12px;
  }

  .titlebar-context,
  .titlebar-divider {
    display: none;
  }

  .window-control {
    width: 42px;
  }

  .content {
    padding-right: 16px;
    padding-left: 16px;
  }

  h1 {
    font-size: 16px;
  }


  .panel .n-card-content {
    padding: 15px;
  }

  .panel-heading {
    flex-direction: column;
    gap: 10px;
  }

  .state-tag {
    align-self: flex-start;
  }

  .drop-zone {
    min-height: clamp(200px, 32dvh, 280px);
  }

  .image-preview-frame {
    min-height: 220px;
  }

  .result-actions {
    align-items: flex-start;
    flex-direction: column;
  }

  .action-feedback {
    text-align: left;
  }

  .workspace-footer {
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .footer-spacer {
    display: none;
  }
}


@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>