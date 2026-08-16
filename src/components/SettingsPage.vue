<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { open as openNativeDialog } from "@tauri-apps/plugin-dialog";
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NInputNumber,
  NModal,
  NSelect,
  NSwitch,
  NTag,
  useMessage,
} from "naive-ui";
import {
  getModelCatalog,
  saveModelCatalog,
  updateBackendSettings,
} from "../services/translation-provider";
import type {
  BackendSettingsUpdate,
  BackendStatus,
  DeviceKind,
  ModelCatalogOptions,
  ModelCatalogUpdate,
} from "../services/translation-provider";
import {
  applySharedBackendStatus,
  backendStatus,
  fetchSharedBackendStatus,
  loadPersistedTargetLanguage,
  savePersistedTargetLanguage,
  targetLanguage,
} from "../services/workspace-settings";
import { setThemeMode, themeMode } from "../services/theme-settings";
import type { ThemeMode } from "../services/theme-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";

type TagType = "default" | "success" | "warning" | "error" | "info";
type ModelDialogMode = "translation" | "ocr" | "font" | null;

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const modelDetectorPath = ref("");
const modelRecognizerPath = ref("");
const modelHyPath = ref("");
const modelFontPath = ref<string | null>(null);
const device = ref<DeviceKind>("cuda");
const regionParallelism = ref(16);
const translationBatchSize = ref(4);
const idleUnloadMinutes = ref(30);
const generationMaxNewTokens = ref(4096);
const generationSampling = ref(true);
const generationTemperature = ref(0.7);
const generationTopK = ref(20);
const generationTopP = ref(0.6);
const generationSeed = ref("");
const generationRepetitionPenalty = ref(1.05);
const generationFrequencyPenalty = ref(0);
const generationStopTokensText = ref("");
const generationStopStringsText = ref("");
const memoryEnabled = ref(false);
const memoryMaxTokens = ref(4096);
const memoryMaxTurns = ref(16);
const systemPrompt = ref("");
const userPrompt = ref("");
const settingsMessage = ref("");
const settingsMessageType = ref<WorkspaceToastType>("info");
const settingsLoading = ref(false);
const dialogSaving = ref(false);
const catalogLoaded = ref(false);
const deviceOptions: Array<{ label: string; value: DeviceKind }> = [
  { label: "CUDA", value: "cuda" },
  { label: "CPU（仅用于状态检查；Hy 翻译需要 CUDA）", value: "cpu" },
];
const themeModeOptions: Array<{ label: string; value: ThemeMode }> = [
  { label: "自动（跟随系统）", value: "system" },
  { label: "浅色（手动）", value: "light" },
  { label: "深色（手动）", value: "dark" },
];
const themeModeLabels: Record<ThemeMode, string> = {
  system: "自动（跟随系统）",
  light: "浅色（手动）",
  dark: "深色（手动）",
};
const toast = useMessage();

function setSettingsFeedback(
  type: WorkspaceToastType,
  message: string,
  notify = true,
) {
  settingsMessage.value = message;
  settingsMessageType.value = type;
  if (notify) {
    showWorkspaceToast(toast, type, message);
  }
}

function handleThemeModeChange(nextMode: ThemeMode | null): void {
  if (!nextMode) {
    return;
  }
  const persistError = setThemeMode(nextMode);
  if (persistError) {
    setSettingsFeedback("error", persistError);
    return;
  }
  setSettingsFeedback("success", `界面主题已切换为${themeModeLabels[nextMode]}。`, false);
}

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

function applyBackendStatus(status: BackendStatus) {
  applySharedBackendStatus(status);
  modelDetectorPath.value = status.detectorModelDir;
  modelRecognizerPath.value = status.recognizerModelDir;
  modelHyPath.value = status.hyModel;
  modelFontPath.value = status.fontPath;
  device.value = status.device === "cpu" ? "cpu" : "cuda";
  translationBatchSize.value = status.translationBatchSize;
  idleUnloadMinutes.value = status.idleUnloadMinutes;
  generationMaxNewTokens.value = status.generation.maxNewTokens;
  generationSampling.value = status.generation.sampling;
  generationTemperature.value = status.generation.temperature;
  generationTopK.value = status.generation.topK;
  generationTopP.value = status.generation.topP;
  generationSeed.value = status.generation.seed ?? "";
  generationRepetitionPenalty.value = status.generation.repetitionPenalty;
  generationFrequencyPenalty.value = status.generation.frequencyPenalty;
  generationStopTokensText.value = status.generation.stopTokens.join(", ");
  generationStopStringsText.value = status.generation.stopStrings.join("\n");
  memoryEnabled.value = status.memory.enabled;
  memoryMaxTokens.value = status.memory.maxTokens;
  memoryMaxTurns.value = status.memory.maxTurns;
  systemPrompt.value = status.prompt.system;
  userPrompt.value = status.prompt.user;
}

async function refreshBackendStatus(notify = true) {
  if (!isDesktopRuntime) {
    setSettingsFeedback("warning", "设置状态仅在 Tauri 桌面端可读取。", notify);
    return;
  }
  settingsLoading.value = true;
  try {
    const status = await fetchSharedBackendStatus();
    applyBackendStatus(status);
    void loadModelCatalog();
    setSettingsFeedback(status.ready ? "success" : "warning", status.message, notify);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法读取后端模型状态。", notify);
  } finally {
    settingsLoading.value = false;
  }
}

const modelCatalog = ref<ModelCatalogOptions>({ translation: [], ocr: [], fonts: [] });
const modelDialogMode = ref<ModelDialogMode>(null);
// Writable computed: the modal's v-model:show must be able to close it
// (top-right X and mask clicks assign `false`).
const modelDialogOpen = computed({
  get: () => modelDialogMode.value !== null,
  set: (open: boolean) => {
    if (!open) {
      modelDialogMode.value = null;
    }
  },
});
const modelDialogTitle = computed(() => {
  if (modelDialogMode.value === "translation") {
    return "配置翻译模型路径";
  }
  if (modelDialogMode.value === "ocr") {
    return "配置 OCR 模型路径";
  }
  return "配置标注字体路径";
});
const dialogName = ref("");
const dialogTranslationPath = ref("");
const dialogOcrDetectorPath = ref("");
const dialogOcrRecognizerPath = ref("");
const dialogFontPath = ref("");

type OcrModelType =
  | "v5-server"
  | "v5-mobile"
  | "v6-tiny"
  | "v6-small"
  | "v6-medium";
const OCR_MODEL_TYPES: Array<{ label: string; value: OcrModelType }> = [
  { label: "PP-OCR v5 server（高精度，较慢）", value: "v5-server" },
  { label: "PP-OCR v5 mobile（快速，适合实时翻译）", value: "v5-mobile" },
  { label: "PP-OCR v6 tiny（轻量，最快）", value: "v6-tiny" },
  { label: "PP-OCR v6 small（均衡）", value: "v6-small" },
  { label: "PP-OCR v6 medium（高精度）", value: "v6-medium" },
];
const dialogOcrType = ref<OcrModelType>("v5-mobile");
const UNCONFIGURED_OCR_PREFIX = "__unset__:";

function pathBaseName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const separator = Math.max(trimmed.lastIndexOf("\\"), trimmed.lastIndexOf("/"));
  return separator >= 0 ? trimmed.slice(separator + 1) : trimmed;
}

// Translation dropdown only ever shows the one supported type, Hy-MT2.
const translationModelPath = computed(
  () => modelCatalog.value.translation[0]?.path ?? modelHyPath.value ?? "",
);
const translationModelOptionsWithCurrent = computed(() =>
  translationModelPath.value
    ? [{ label: "Hy-MT2", value: translationModelPath.value }]
    : [],
);
const selectedTranslationValue = computed(() => modelHyPath.value || null);

function resolveOcrEntry(type: OcrModelType): {
  detectorDir: string;
  recognizerDir: string;
} | null {
  const entry = modelCatalog.value.ocr.find((option) => option.variant === type);
  if (entry) {
    return { detectorDir: entry.detectorDir, recognizerDir: entry.recognizerDir };
  }
  if (
    modelDetectorPath.value &&
    modelRecognizerPath.value &&
    backendStatus.value?.detectorVariant === type
  ) {
    return { detectorDir: modelDetectorPath.value, recognizerDir: modelRecognizerPath.value };
  }
  return null;
}

// OCR dropdown only ever shows the two supported variants, resolved to paths.
// Unconfigured variants stay visible but disabled, with a clear hint.
const ocrModelOptions = computed(() =>
  OCR_MODEL_TYPES.map(({ label, value }) => {
    const resolved = resolveOcrEntry(value);
    return resolved
      ? { label, value: `${resolved.detectorDir}|${resolved.recognizerDir}` }
      : {
          label: `${label}（未配置路径）`,
          value: `${UNCONFIGURED_OCR_PREFIX}${value}`,
          disabled: true,
        };
  }),
);

// 配置翻译模型弹窗里的模型下拉与页面下拉保持一致：只显示类型「Hy-MT2」，
// 其取值跟随弹窗当前路径（选择新文件后仍显示 Hy-MT2，不细分文件名）。
const translationDialogPath = computed(
  () =>
    dialogTranslationPath.value ||
    modelCatalog.value.translation[0]?.path ||
    modelHyPath.value ||
    "",
);
const translationDialogOptions = computed(() =>
  translationDialogPath.value
    ? [{ label: "Hy-MT2", value: translationDialogPath.value }]
    : [],
);

function selectDialogTranslationModel(value: string | null): void {
  dialogTranslationPath.value = value ?? "";
}
const selectedOcrValue = computed(() =>
  modelDetectorPath.value && modelRecognizerPath.value
    ? `${modelDetectorPath.value}|${modelRecognizerPath.value}`
    : null,
);

const SYSTEM_FONT_VALUE = "__system__";
const fontModelOptions = computed(() => [
  { label: "系统自动匹配", value: SYSTEM_FONT_VALUE },
  ...modelCatalog.value.fonts.map((option) => ({
    label: `${option.name}（${option.path ? pathBaseName(option.path) : ""}）`,
    value: option.path ?? SYSTEM_FONT_VALUE,
  })),
]);
const selectedFontValue = computed(() => modelFontPath.value ?? SYSTEM_FONT_VALUE);

async function loadModelCatalog(): Promise<void> {
  if (!isDesktopRuntime) {
    catalogLoaded.value = true;
    return;
  }
  try {
    modelCatalog.value = await getModelCatalog();
  } catch {
    modelCatalog.value = { translation: [], ocr: [], fonts: [] };
  } finally {
    catalogLoaded.value = true;
  }
}

function selectTranslationModel(path: string): void {
  modelHyPath.value = path;
  setSettingsFeedback("info", "已选择翻译模型 Hy-MT2，点击“保存设置”生效。");
}

function selectOcrModel(value: string): void {
  if (value.startsWith(UNCONFIGURED_OCR_PREFIX)) {
    setSettingsFeedback("error", "该模型类型尚未配置路径，请点击“配置路径…”设置。");
    return;
  }
  const separator = value.indexOf("|");
  if (separator < 0) {
    return;
  }
  modelDetectorPath.value = value.slice(0, separator);
  modelRecognizerPath.value = value.slice(separator + 1);
  setSettingsFeedback("info", "已选择 OCR 模型，点击“保存设置”生效。");
}

function selectFontModel(value: string): void {
  modelFontPath.value = value === SYSTEM_FONT_VALUE ? null : value;
  setSettingsFeedback("info", "已选择标注字体，点击“保存设置”生效。");
}

function openModelDialog(mode: Exclude<ModelDialogMode, null>): void {
  dialogName.value = "";
  dialogTranslationPath.value = "";
  dialogOcrDetectorPath.value = "";
  dialogOcrRecognizerPath.value = "";
  dialogFontPath.value = "";
  dialogOcrType.value =
    backendStatus.value?.detectorVariant === "v5-server" ? "v5-server" : "v5-mobile";
  modelDialogMode.value = mode;
}

function closeModelDialog(): void {
  modelDialogMode.value = null;
}

async function pickDialogTranslationPath(): Promise<void> {
  const selected = await openNativeDialog({
    title: "选择 Hy-MT2 GGUF 模型",
    defaultPath: dialogTranslationPath.value || undefined,
    multiple: false,
    filters: [{ name: "GGUF 模型", extensions: ["gguf"] }],
  });
  if (typeof selected === "string" && selected.trim()) {
    dialogTranslationPath.value = selected;
  }
}

async function pickDialogOcrDetectorPath(): Promise<void> {
  const selected = await openNativeDialog({
    title: "选择 PP-OCR detector 文件夹",
    defaultPath: dialogOcrDetectorPath.value || undefined,
    directory: true,
    multiple: false,
  });
  if (typeof selected === "string" && selected.trim()) {
    dialogOcrDetectorPath.value = selected;
  }
}

async function pickDialogOcrRecognizerPath(): Promise<void> {
  const selected = await openNativeDialog({
    title: "选择 PP-OCR recognizer 文件夹",
    defaultPath: dialogOcrRecognizerPath.value || undefined,
    directory: true,
    multiple: false,
  });
  if (typeof selected === "string" && selected.trim()) {
    dialogOcrRecognizerPath.value = selected;
  }
}

async function pickDialogFontPath(): Promise<void> {
  const selected = await openNativeDialog({
    title: "选择标注字体",
    defaultPath: dialogFontPath.value || undefined,
    multiple: false,
    filters: [{ name: "字体文件", extensions: ["ttf", "otf"] }],
  });
  if (typeof selected === "string" && selected.trim()) {
    dialogFontPath.value = selected;
  }
}

async function saveModelDialog(): Promise<void> {
  if (dialogSaving.value) {
    return;
  }
  const mode = modelDialogMode.value;
  const next: ModelCatalogUpdate = {
    translation: modelCatalog.value.translation.map((entry) => ({
      name: entry.name,
      path: entry.path,
    })),
    ocr: modelCatalog.value.ocr.map((entry) => ({
      name: entry.name,
      detectorDir: entry.detectorDir,
      recognizerDir: entry.recognizerDir,
    })),
    fonts: modelCatalog.value.fonts
      .filter((entry) => entry.path !== null)
      .map((entry) => ({ name: entry.name, path: entry.path as string })),
  };
  let entryName = "";
  if (mode === "translation") {
    entryName = "Hy-MT2";
    const path = dialogTranslationPath.value.trim();
    if (!path) {
      setSettingsFeedback("error", "请选择 GGUF 模型文件。");
      return;
    }
    next.translation.push({ name: entryName, path });
  } else if (mode === "ocr") {
    entryName = dialogOcrType.value;
    const detectorDir = dialogOcrDetectorPath.value.trim();
    const recognizerDir = dialogOcrRecognizerPath.value.trim();
    if (!detectorDir || !recognizerDir) {
      setSettingsFeedback("error", "请选择 detector 与 recognizer 两个文件夹。");
      return;
    }
    next.ocr.push({ name: entryName, detectorDir, recognizerDir });
  } else if (mode === "font") {
    entryName = dialogName.value.trim();
    if (!entryName) {
      setSettingsFeedback("error", "请输入字体名称。");
      return;
    }
    const path = dialogFontPath.value.trim();
    if (!path) {
      setSettingsFeedback("error", "请选择字体文件。");
      return;
    }
    next.fonts.push({ name: entryName, path });
  } else {
    return;
  }
  dialogSaving.value = true;
  try {
    await saveModelCatalog(next);
    await loadModelCatalog();
    if (mode === "translation") {
      selectTranslationModel(dialogTranslationPath.value);
    } else if (mode === "ocr") {
      selectOcrModel(`${dialogOcrDetectorPath.value}|${dialogOcrRecognizerPath.value}`);
    } else {
      selectFontModel(dialogFontPath.value);
    }
    closeModelDialog();
    setSettingsFeedback("success", `已配置「${entryName}」并选中，点击“保存设置”生效。`);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存模型条目。");
  } finally {
    dialogSaving.value = false;
  }
}

function requireInteger(value: number, min: number, max: number, label: string): number | null {
  if (!Number.isInteger(value) || value < min || value > max) {
    setSettingsFeedback("error", `${label}必须为 ${min} 到 ${max} 的整数。`);
    return null;
  }
  return value;
}

function requireFiniteRange(
  value: number,
  minExclusive: number | null,
  minInclusive: number | null,
  maxInclusive: number | null,
  label: string,
): number | null {
  const minOk = minExclusive === null || value > minExclusive;
  const inclusiveOk = minInclusive === null || value >= minInclusive;
  const maxOk = maxInclusive === null || value <= maxInclusive;
  if (!Number.isFinite(value) || !minOk || !inclusiveOk || !maxOk) {
    setSettingsFeedback("error", `${label}超出允许范围。`);
    return null;
  }
  return value;
}

function parseStopTokens(): number[] | null {
  const text = generationStopTokensText.value.trim();
  if (!text) {
    return [];
  }
  const tokens: number[] = [];
  const seen = new Set<number>();
  for (const part of text.split(/[\s,]+/)) {
    if (!/^\d+$/.test(part)) {
      setSettingsFeedback("error", "停止 token 必须是 0 到 1000000 的整数。");
      return null;
    }
    const token = Number(part);
    if (!Number.isInteger(token) || token < 0 || token > 1000000) {
      setSettingsFeedback("error", "停止 token 必须是 0 到 1000000 的整数。");
      return null;
    }
    if (!seen.has(token)) {
      seen.add(token);
      tokens.push(token);
    }
  }
  if (tokens.length > 32) {
    setSettingsFeedback("error", "停止 token 最多 32 个。");
    return null;
  }
  return tokens;
}

function parseStopStrings(): string[] | null {
  const stopStrings: string[] = [];
  const seen = new Set<string>();
  for (const line of generationStopStringsText.value.split(/\r?\n/)) {
    const value = line.trim();
    if (!value) {
      continue;
    }
    if (Array.from(value).length > 128) {
      setSettingsFeedback("error", "每个停止字符串最多 128 个字符。");
      return null;
    }
    if (!seen.has(value)) {
      seen.add(value);
      stopStrings.push(value);
    }
  }
  if (stopStrings.length > 16) {
    setSettingsFeedback("error", "停止字符串最多 16 条。");
    return null;
  }
  return stopStrings;
}

async function saveModelSettings() {
  if (!isDesktopRuntime) {
    if (saveSettings(false)) {
      setSettingsFeedback("warning", "浏览器预览只保存翻译设置；模型参数仅在 Tauri 桌面端可保存。");
    } else {
      showWorkspaceToast(toast, "error", settingsMessage.value);
    }
    return;
  }

  const nextLanguage = targetLanguage.value.trim();
  if (Array.from(nextLanguage).length < 1 || Array.from(nextLanguage).length > 64) {
    setSettingsFeedback("error", "目标语言长度必须为 1 到 64 个字符。");
    return;
  }
  const detectorModelDir = modelDetectorPath.value.trim();
  const recognizerModelDir = modelRecognizerPath.value.trim();
  const hyModel = modelHyPath.value.trim();
  if (!detectorModelDir || !recognizerModelDir || !hyModel) {
    setSettingsFeedback("error", "请选择完整的 PP-OCR 与 Hy-MT2 模型路径。");
    return;
  }
  const idleMinutes = requireInteger(idleUnloadMinutes.value ?? 0, 0, 1440, "模型空闲释放时间");
  const ocrParallelism = requireInteger(regionParallelism.value ?? 0, 1, 16, "OCR 并发");
  const batchSize = requireInteger(translationBatchSize.value ?? 0, 1, 4, "Hy 批大小");
  const maxNewTokens = requireInteger(generationMaxNewTokens.value ?? 0, 1, 4096, "最大生成 token");
  const topK = requireInteger(generationTopK.value ?? 0, 0, 1024, "top-k");
  const memoryTokens = requireInteger(memoryMaxTokens.value ?? 0, 1, 262144, "记忆 token 预算");
  const memoryTurns = requireInteger(memoryMaxTurns.value ?? 0, 1, 1024, "记忆轮数");
  if (
    idleMinutes === null ||
    ocrParallelism === null ||
    batchSize === null ||
    maxNewTokens === null ||
    topK === null ||
    memoryTokens === null ||
    memoryTurns === null
  ) {
    return;
  }
  if (generationSampling.value && topK === 0) {
    setSettingsFeedback("error", "开启 sampling 时 top-k 必须大于 0。");
    return;
  }
  const temperature = requireFiniteRange(generationTemperature.value, 0, null, null, "temperature");
  const topP = requireFiniteRange(generationTopP.value, 0, null, 1, "top-p");
  const repetitionPenalty = requireFiniteRange(
    generationRepetitionPenalty.value,
    0,
    null,
    null,
    "repetition penalty",
  );
  const frequencyPenalty = requireFiniteRange(
    generationFrequencyPenalty.value,
    null,
    0,
    null,
    "frequency penalty",
  );
  if (
    temperature === null ||
    topP === null ||
    repetitionPenalty === null ||
    frequencyPenalty === null
  ) {
    return;
  }
  const seedText = generationSeed.value.trim();
  if (seedText && !/^[1-9]\d*$/.test(seedText)) {
    setSettingsFeedback("error", "seed 必须为空，或为正整数十进制字符串。");
    return;
  }
  const stopTokens = parseStopTokens();
  const stopStrings = parseStopStrings();
  if (stopTokens === null || stopStrings === null) {
    return;
  }
  const trimmedSystemPrompt = systemPrompt.value.trim();
  if (Array.from(trimmedSystemPrompt).length > 4096) {
    setSettingsFeedback("error", "system 预设提示词最多 4096 个字符。");
    return;
  }
  const trimmedUserPrompt = userPrompt.value.trim();
  if (Array.from(trimmedUserPrompt).length > 4096) {
    setSettingsFeedback("error", "user 预设提示词最多 4096 个字符。");
    return;
  }

  const settings: BackendSettingsUpdate = {
    detectorModelDir,
    recognizerModelDir,
    hyModel,
    fontPath: modelFontPath.value?.trim() || null,
    targetLanguage: nextLanguage,
    device: device.value,
    regionParallelism: ocrParallelism,
    translationBatchSize: batchSize,
    idleUnloadMinutes: idleMinutes,
    generation: {
      maxNewTokens,
      sampling: generationSampling.value,
      temperature,
      topK,
      topP,
      seed: seedText || null,
      repetitionPenalty,
      frequencyPenalty,
      stopTokens,
      stopStrings,
    },
    memory: {
      enabled: memoryEnabled.value,
      maxTokens: memoryTokens,
      maxTurns: memoryTurns,
    },
    prompt: {
      system: trimmedSystemPrompt,
      user: trimmedUserPrompt,
    },
  };

  settingsLoading.value = true;
  try {
    const status = await updateBackendSettings(settings);
    applyBackendStatus(status);
    setSettingsFeedback(
      "success",
      idleMinutes === 0
        ? "设置已保存，下一次翻译会使用新的参数。自动释放已关闭。"
        : "设置已保存，下一次翻译会使用新的参数。",
    );
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存模型设置。");
  } finally {
    settingsLoading.value = false;
  }
}

function saveSettings(notify = true): boolean {
  const nextLanguage = targetLanguage.value.trim();
  if (Array.from(nextLanguage).length < 1 || Array.from(nextLanguage).length > 64) {
    setSettingsFeedback("error", "目标语言长度必须为 1 到 64 个字符。", notify);
    return false;
  }
  targetLanguage.value = nextLanguage;
  const persistError = savePersistedTargetLanguage();
  if (persistError) {
    setSettingsFeedback("error", persistError, notify);
    return false;
  }
  setSettingsFeedback("success", "翻译设置已保存。", notify);
  return true;
}

onMounted(() => {
  const persistedSettingsError = loadPersistedTargetLanguage();
  if (persistedSettingsError) {
    setSettingsFeedback("warning", persistedSettingsError, false);
  }
  if (backendStatus.value) {
    applyBackendStatus(backendStatus.value);
  }
  void loadModelCatalog();
  void refreshBackendStatus(false);
});

</script>

<template>
  <section class="settings-page" aria-labelledby="settings-title">
    <div class="settings-grid">


      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">01 / 模型准备</p>
            <h2>本地模型</h2>
          </div>
        </div>
        <p class="settings-card-copy">
          选择 Hy-MT2 与 PP-OCR 模型，为实时翻译做好准备；标注字体可选。配置路径后，点击"保存设置"生效。
        </p>
        <dl class="settings-path-list">
          <div>
            <dt>翻译模型（Hy-MT2）</dt>
            <dd class="settings-model-select-row">
              <n-select
                :value="selectedTranslationValue"
                :options="translationModelOptionsWithCurrent"
                :placeholder="catalogLoaded ? '选择模型' : '加载模型列表…'"
                size="small"
                class="settings-model-select"
                @update:value="selectTranslationModel"
                aria-label="翻译模型"
              />
              <n-button secondary size="small" @click="openModelDialog('translation')">配置路径…</n-button>
              <span class="settings-model-help">{{ modelHyPath || "未选择翻译模型" }}</span>
            </dd>
          </div>
          <div>
            <dt>OCR 模型（PP-OCR）</dt>
            <dd class="settings-model-select-row">
              <n-select
                :value="selectedOcrValue"
                :options="ocrModelOptions"
                :placeholder="catalogLoaded ? '选择模型' : '加载模型列表…'"
                size="small"
                class="settings-model-select"
                @update:value="selectOcrModel"
                aria-label="OCR 模型"
              />
              <n-button secondary size="small" @click="openModelDialog('ocr')">配置路径…</n-button>
              <span class="settings-model-help">
                det: {{ modelDetectorPath || "未选择" }} · rec: {{ modelRecognizerPath || "未选择" }}
              </span>
            </dd>
          </div>
          <div>
            <dt>标注字体</dt>
            <dd class="settings-model-select-row">
              <n-select
                :value="selectedFontValue"
                :options="fontModelOptions"
                :placeholder="catalogLoaded ? '选择字体' : '加载字体列表…'"
                size="small"
                class="settings-model-select"
                @update:value="selectFontModel"
                aria-label="标注字体"
              />
              <n-button secondary size="small" @click="openModelDialog('font')">配置路径…</n-button>
              <span class="settings-model-help">{{ modelFontPath || "系统自动匹配" }}</span>
            </dd>
          </div>
        </dl>
      </n-card>

      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">界面外观</p>
            <h2>颜色主题</h2>
          </div>
        </div>
        <p class="settings-card-copy">
          选择手动主题，或使用自动模式跟随 Windows 系统的浅色与深色设置；切换会立即生效并自动保存。
        </p>
        <label class="settings-field settings-field-wide">
          <span>主题模式</span>
          <n-select
            :value="themeMode"
            :options="themeModeOptions"
            aria-label="主题模式"
            @update:value="handleThemeModeChange"
          />
          <span class="settings-help">自动模式会在系统主题变化时实时切换。</span>
        </label>
      </n-card>

      <n-modal
        v-model:show="modelDialogOpen"
        preset="card"
        :title="modelDialogTitle"
        :mask-closable="true"
        style="width: 520px; max-width: calc(100vw - 48px)"
        class="model-path-dialog"
      >
        <div class="model-dialog-fields">
          <template v-if="modelDialogMode === 'ocr'">
            <n-select
              v-model:value="dialogOcrType"
              :options="OCR_MODEL_TYPES"
              aria-label="OCR 模型类型"
            />
            <div class="model-dialog-path-row">
              <span class="model-dialog-path-value" :title="dialogOcrDetectorPath">{{ dialogOcrDetectorPath || "未选择" }}</span>
              <n-button secondary size="small" @click="pickDialogOcrDetectorPath">选择 detector 文件夹</n-button>
            </div>
            <div class="model-dialog-path-row">
              <span class="model-dialog-path-value" :title="dialogOcrRecognizerPath">{{ dialogOcrRecognizerPath || "未选择" }}</span>
              <n-button secondary size="small" @click="pickDialogOcrRecognizerPath">选择 recognizer 文件夹</n-button>
            </div>
          </template>
          <template v-else-if="modelDialogMode === 'translation'">
            <n-select
              :value="translationDialogPath || null"
              :options="translationDialogOptions"
              placeholder="选择已发现或注册的 GGUF 模型"
              clearable
              @update:value="selectDialogTranslationModel"
              aria-label="翻译模型文件"
            />
            <div class="model-dialog-path-row">
              <span class="model-dialog-path-value" :title="dialogTranslationPath">{{ dialogTranslationPath || "未选择" }}</span>
              <n-button secondary size="small" @click="pickDialogTranslationPath">选择 GGUF 文件</n-button>
            </div>
          </template>
          <template v-else>
            <n-input
              v-model:value="dialogName"
              maxlength="64"
              placeholder="字体名称（用于下拉框显示）"
            />
            <div class="model-dialog-path-row">
              <span class="model-dialog-path-value" :title="dialogFontPath">{{ dialogFontPath || "未选择" }}</span>
              <n-button secondary size="small" @click="pickDialogFontPath">选择字体文件</n-button>
            </div>
          </template>
        </div>
        <template #footer>
          <div class="model-dialog-footer">
            <n-button secondary size="small" :disabled="dialogSaving" @click="closeModelDialog">取消</n-button>
            <n-button type="primary" size="small" :loading="dialogSaving" @click="saveModelDialog">注册并选择</n-button>
          </div>
        </template>
      </n-modal>

      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">02 / 翻译准备</p>
            <h2>目标语言与提示词</h2>
          </div>
        </div>
        <p class="settings-card-copy">先设置目标语言；system 和 user 预设提示词会作为下一次翻译请求的默认模型上下文。</p>
        <div class="settings-field-grid">
          <label class="settings-field">
            <span>目标语言</span>
            <n-input
              v-model:value="targetLanguage"
              maxlength="64"
              placeholder="例如：Chinese"
              aria-label="目标语言"
            />
          </label>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>System 预设提示词</span>
            <n-input
              v-model:value="systemPrompt"
              type="textarea"
              maxlength="4096"
              placeholder="可选：例如 Return concise JSON."
              :autosize="{ minRows: 3, maxRows: 6 }"
              aria-label="Hy system prompt"
            />
          </label>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>User 预设提示词</span>
            <n-input
              v-model:value="userPrompt"
              type="textarea"
              maxlength="4096"
              placeholder="可选：例如 Preserve product names and translate only visible text."
              :autosize="{ minRows: 3, maxRows: 6 }"
              aria-label="Hy user preset prompt"
            />
          </label>
        </div>
      </n-card>
      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">运行资源</p>
            <h2>设备、批处理与释放</h2>
          </div>
          <n-tag type="info" round size="small">CUDA</n-tag>
        </div>
        <p class="settings-card-copy">
          控制本地推理设备、OCR 区域并发、Hy 翻译批大小，以及翻译完成后保持模型在显存中的时间。
        </p>
        <div class="settings-field-grid">
          <label class="settings-field">
            <span>设备</span>
            <n-select v-model:value="device" :options="deviceOptions" aria-label="模型运行设备" />
            <span class="settings-help">CPU 可用于状态检查；Hy 翻译仍需要 CUDA。</span>
          </label>
          <label class="settings-field settings-number-field">
            <span>OCR 并发</span>
            <n-input-number
              v-model:value="regionParallelism"
              :min="1"
              :max="16"
              :step="1"
              aria-label="OCR 区域并发"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>Hy 批大小</span>
            <n-input-number
              v-model:value="translationBatchSize"
              :min="1"
              :max="4"
              :step="1"
              aria-label="Hy 翻译批大小"
            />
          </label>
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
        </div>
      </n-card>
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
          <div>
            <span>OCR 并发</span>
            <strong>{{ backendStatus?.regionParallelism ?? regionParallelism }}</strong>
          </div>
          <div>
            <span>Hy 批大小</span>
            <strong>{{ backendStatus?.translationBatchSize ?? translationBatchSize }}</strong>
          </div>
        </div>
      </n-card>


      <details class="settings-advanced settings-card-wide">
        <summary class="settings-advanced-summary">
          <span>
            <span class="panel-kicker">高级设置</span>
            <strong>Hy 生成与记忆参数</strong>
          </span>
          <span class="settings-advanced-summary-copy">生成：文本、OCR、实时；记忆：仅文本、OCR，实时翻译使用独立配置。</span>
        </summary>
      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">Hy 生成参数</p>
            <h2>采样与惩罚</h2>
          </div>
          <n-switch v-model:value="generationSampling" aria-label="启用 Hy sampling">
            <template #checked>Sampling</template>
            <template #unchecked>Greedy</template>
          </n-switch>
        </div>
        <p class="settings-scope-note">
          <strong>适用：</strong>文本翻译、OCR 翻译、实时翻译。
          <span class="settings-scope-warning">maxNewTokens 会在 OCR 与实时翻译中按文本长度自动调整，预热阶段临时使用 1。</span>
        </p>
        <div class="settings-field-grid">
          <label class="settings-field settings-number-field">
            <span>最大生成 token</span>
            <n-input-number
              v-model:value="generationMaxNewTokens"
              :min="1"
              :max="4096"
              :step="16"
              aria-label="Hy 最大生成 token"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>temperature</span>
            <n-input-number
              v-model:value="generationTemperature"
              :min="0.01"
              :step="0.05"
              aria-label="Hy temperature"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>top-k</span>
            <n-input-number
              v-model:value="generationTopK"
              :min="0"
              :max="1024"
              :step="1"
              aria-label="Hy top-k"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>top-p</span>
            <n-input-number
              v-model:value="generationTopP"
              :min="0.01"
              :max="1"
              :step="0.01"
              aria-label="Hy top-p"
            />
          </label>
          <label class="settings-field">
            <span>seed</span>
            <n-input v-model:value="generationSeed" placeholder="空表示默认；例如 42" aria-label="Hy sampling seed" />
          </label>
          <label class="settings-field settings-number-field">
            <span>repetition penalty</span>
            <n-input-number
              v-model:value="generationRepetitionPenalty"
              :min="0.01"
              :step="0.05"
              aria-label="Hy repetition penalty"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>frequency penalty</span>
            <n-input-number
              v-model:value="generationFrequencyPenalty"
              :min="0"
              :step="0.05"
              aria-label="Hy frequency penalty"
            />
          </label>
        </div>
      </n-card>

      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">停止条件与记忆</p>
            <h2>Stop tokens、stop strings 与对话记忆</h2>
          </div>
          <n-switch v-model:value="memoryEnabled" aria-label="启用 Hy 对话记忆">
            <template #checked>Memory on</template>
            <template #unchecked>Memory off</template>
          </n-switch>
        </div>
        <p class="settings-scope-note">
          <strong>Stop tokens / stop strings：</strong>适用于文本、OCR、实时翻译。
          <strong>记忆开关、token 预算、轮数：</strong>仅适用于文本翻译和 OCR 翻译；实时翻译使用实时翻译页的独立上下文配置。
        </p>
        <div class="settings-field-grid">
          <label class="settings-field settings-field-wide settings-textarea">
            <span>stop tokens</span>
            <n-input
              v-model:value="generationStopTokensText"
              type="textarea"
              placeholder="例如：120020"
              :autosize="{ minRows: 2, maxRows: 4 }"
              aria-label="Hy stop tokens"
            />
          </label>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>stop strings</span>
            <n-input
              v-model:value="generationStopStringsText"
              type="textarea"
              placeholder="每行一个停止字符串"
              :autosize="{ minRows: 2, maxRows: 5 }"
              aria-label="Hy stop strings"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>记忆 token 预算</span>
            <n-input-number
              v-model:value="memoryMaxTokens"
              :min="1"
              :max="262144"
              :step="256"
              aria-label="Hy 记忆 token 预算"
            />
          </label>
          <label class="settings-field settings-number-field">
            <span>记忆轮数</span>
            <n-input-number
              v-model:value="memoryMaxTurns"
              :min="1"
              :max="1024"
              :step="1"
              aria-label="Hy 记忆轮数"
            />
          </label>
        </div>
      </n-card>
      </details>

      <div class="settings-card-actions settings-page-actions settings-card-wide">
        <n-alert v-if="settingsMessage" class="settings-actions-feedback" :type="settingsMessageType" :show-icon="false">
          {{ settingsMessage }}
        </n-alert>
        <n-button secondary :loading="settingsLoading" @click="refreshBackendStatus()">刷新状态</n-button>
        <n-button type="primary" :loading="settingsLoading" @click="saveModelSettings">保存设置</n-button>
      </div>
    </div>
  </section>
</template>
<style scoped src="../styles/settings-page.css"></style>
