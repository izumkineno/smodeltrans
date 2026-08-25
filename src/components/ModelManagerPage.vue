<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { open as openNativeDialog } from "@tauri-apps/plugin-dialog";
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NInputNumber,
  NModal,
  NPopconfirm,
  NProgress,
  NSelect,
  NSwitch,
  NTag,
  NTooltip,
  useMessage,
} from "naive-ui";
import {
  DEFAULT_IDLE_UNLOAD_SECONDS,
  IDLE_UNLOAD_SECONDS_MAX,
  IDLE_UNLOAD_SECONDS_MIN,
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
  targetLanguage,
} from "../services/workspace-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";
import {
  type DownloadSource,
  type DownloadTaskState,
  activateDownloadedModel,
  cancelModelDownload,
  deleteDownloadedModel,
  listDownloadableModels,
  listDownloadedModels,
  listenDownloadProgress,
  startModelDownload,
} from "../services/model-download-provider";

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
const idleUnloadSeconds = ref(DEFAULT_IDLE_UNLOAD_SECONDS);
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
const settingsMessage = ref("");
const settingsMessageType = ref<WorkspaceToastType>("info");
const settingsLoading = ref(false);
const dialogSaving = ref(false);
const catalogLoaded = ref(false);
const deviceOptions: Array<{ label: string; value: DeviceKind }> = [
  { label: "CUDA", value: "cuda" },
  { label: "CPU（仅用于状态检查；Hy 翻译需要 CUDA）", value: "cpu" },
];
const toast = useMessage();

// --- ModelScope 下载状态，前端以 ModelScope 为默认源 ---
const downloadSource = ref<DownloadSource>("modelscope");
const downloadSourceOptions: Array<{ label: string; value: DownloadSource }> = [
  { label: "ModelScope（默认）", value: "modelscope" },
  { label: "Hugging Face", value: "huggingface" },
];
const downloadTasks = ref<Record<string, DownloadTaskState>>({});
const downloadedSet = ref<Set<string>>(new Set());
const downloadedBaseDir = ref<Record<string, string>>({});
const translationModels = computed(() =>
  listDownloadableModels(downloadSource.value).filter((m) => m.kind === "translation"),
);
const ocrModels = computed(() =>
  listDownloadableModels(downloadSource.value).filter((m) => m.kind === "ocr"),
);

function downloadTaskFor(modelId: string): DownloadTaskState | undefined {
  return downloadTasks.value[modelId];
}

function isModelInstalled(modelId: string): boolean {
  if (modelId.startsWith("hy-mt2")) {
    const cur = (modelHyPath.value || backendStatus.value?.hyModel || "").trim();
    if (!cur) return false;
    const normalizedCur = cur.replace(/\\/g, "/").toLowerCase();
    const base = downloadedBaseDir.value[modelId];
    if (base) {
      const normalizedBase = base.replace(/\\/g, "/").toLowerCase();
      if (normalizedCur === normalizedBase || normalizedCur.startsWith(normalizedBase + "/")) return true;
    }
    // 回退：路径中包含 modelId 目录或以 file 名结尾
    const model = translationModels.value.find((m) => m.id === modelId);
    if (model) {
      if (normalizedCur.includes(`/${modelId.toLowerCase()}/`)) return true;
      for (const f of model.files) {
        if (normalizedCur.toLowerCase().endsWith(`/${f.toLowerCase()}`)) return true;
      }
    }
    return false;
  }
  const ocrVariantMap: Record<string, string> = {
    "ppocr-v5-mobile": "v5-mobile",
    "ppocr-v5-server": "v5-server",
    "ppocr-v6-tiny": "v6-tiny",
    "ppocr-v6-small": "v6-small",
    "ppocr-v6-medium": "v6-medium",
  };
  const variant = ocrVariantMap[modelId];
  if (variant) {
    return backendStatus.value?.detectorVariant === variant;
  }
  return false;
}

function isDownloaded(modelId: string): boolean {
  return downloadedSet.value.has(modelId);
}

async function refreshDownloaded(): Promise<void> {
  try {
    const list = await listDownloadedModels();
    const next = new Set<string>();
    const base: Record<string, string> = {};
    for (const item of list) {
      if (item.downloaded) next.add(item.modelId);
      base[item.modelId] = item.baseDir;
    }
    downloadedSet.value = next;
    downloadedBaseDir.value = base;
  } catch {
    // 忽略
  }
}

async function handleSwitchModel(modelId: string): Promise<void> {
  if (!isDownloaded(modelId)) {
    setSettingsFeedback("error", "该模型尚未下载，请先下载");
    return;
  }
  if (isModelInstalled(modelId)) {
    setSettingsFeedback("info", "该模型已是当前启用模型");
    return;
  }
  settingsLoading.value = true;
  try {
    const status = (await activateDownloadedModel(modelId)) as BackendStatus;
    applyBackendStatus(status);
    await loadModelCatalog();
    await refreshDownloaded();
    setSettingsFeedback("success", `已切换到模型 ${modelId}，后端已更新`);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : `切换失败 ${modelId}`);
  } finally {
    settingsLoading.value = false;
  }
}

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
  idleUnloadSeconds.value = status.idleUnloadSeconds;
  regionParallelism.value = status.regionParallelism;
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
}

async function refreshBackendStatus(notify = true) {
  if (!isDesktopRuntime) {
    setSettingsFeedback("warning", "模型状态仅在 Tauri 桌面端可读取。", notify);
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
    const normalized = path.replace(/\\/g, "/").toLowerCase();
    const exists = next.translation.some((e) => e.path.replace(/\\/g, "/").toLowerCase() === normalized);
    if (!exists) {
      next.translation.push({ name: entryName, path });
    } else {
      // 已存在则更新路径（避免额外显示重复条目）
      next.translation = next.translation.map((e) =>
        e.path.replace(/\\/g, "/").toLowerCase() === normalized ? { name: entryName, path } : e,
      );
    }
  } else if (mode === "ocr") {
    entryName = dialogOcrType.value;
    const detectorDir = dialogOcrDetectorPath.value.trim();
    const recognizerDir = dialogOcrRecognizerPath.value.trim();
    if (!detectorDir || !recognizerDir) {
      setSettingsFeedback("error", "请选择 detector 与 recognizer 两个文件夹。");
      return;
    }
    const normDet = detectorDir.replace(/\\/g, "/").toLowerCase();
    const normRec = recognizerDir.replace(/\\/g, "/").toLowerCase();
    const exists = next.ocr.some(
      (e) =>
        e.detectorDir.replace(/\\/g, "/").toLowerCase() === normDet &&
        e.recognizerDir.replace(/\\/g, "/").toLowerCase() === normRec,
    );
    if (!exists) {
      next.ocr.push({ name: entryName, detectorDir, recognizerDir });
    } else {
      next.ocr = next.ocr.map((e) =>
        e.detectorDir.replace(/\\/g, "/").toLowerCase() === normDet &&
        e.recognizerDir.replace(/\\/g, "/").toLowerCase() === normRec
          ? { name: entryName, detectorDir, recognizerDir }
          : e,
      );
    }
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
    const normalized = path.replace(/\\/g, "/").toLowerCase();
    const exists = next.fonts.some((e) => e.path.replace(/\\/g, "/").toLowerCase() === normalized);
    if (!exists) {
      next.fonts.push({ name: entryName, path });
    } else {
      // 同路径仅更新名称，避免字体下拉额外显示重复条目
      next.fonts = next.fonts.map((e) =>
        e.path.replace(/\\/g, "/").toLowerCase() === normalized ? { name: entryName, path } : e,
      );
    }
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
    setSettingsFeedback("warning", "模型设置仅在 Tauri 桌面端可保存。");
    return;
  }

  const detectorModelDir = modelDetectorPath.value.trim();
  const recognizerModelDir = modelRecognizerPath.value.trim();
  const hyModel = modelHyPath.value.trim();
  if (!detectorModelDir || !recognizerModelDir || !hyModel) {
    setSettingsFeedback("error", "请选择完整的 PP-OCR 与 Hy-MT2 模型路径。");
    return;
  }
  const idleSeconds = requireInteger(
    idleUnloadSeconds.value ?? 0,
    IDLE_UNLOAD_SECONDS_MIN,
    IDLE_UNLOAD_SECONDS_MAX,
    "模型空闲释放时间",
  );
  const ocrParallelism = requireInteger(regionParallelism.value ?? 0, 1, 16, "OCR 并发");
  const batchSize = requireInteger(translationBatchSize.value ?? 0, 1, 4, "Hy 批大小");
  const maxNewTokens = requireInteger(generationMaxNewTokens.value ?? 0, 1, 4096, "最大生成 token");
  const topK = requireInteger(generationTopK.value ?? 0, 0, 1024, "top-k");
  const memoryTokens = requireInteger(memoryMaxTokens.value ?? 0, 1, 262144, "记忆 token 预算");
  const memoryTurns = requireInteger(memoryMaxTurns.value ?? 0, 1, 1024, "记忆轮数");
  if (
    idleSeconds === null ||
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

  // 保留当前的语言与提示词，不在此页修改
  const status = backendStatus.value;
  if (!status) {
    setSettingsFeedback("error", "后端状态未就绪，请先刷新。");
    return;
  }

  const settings: BackendSettingsUpdate = {
    detectorModelDir,
    recognizerModelDir,
    hyModel,
    fontPath: modelFontPath.value?.trim() || null,
    targetLanguage: (targetLanguage.value || status.targetLanguage || "Chinese").trim(),
    device: device.value,
    regionParallelism: ocrParallelism,
    translationBatchSize: batchSize,
    idleUnloadSeconds: idleSeconds,
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
      system: status.prompt.system,
      user: status.prompt.user,
    },
  };

  settingsLoading.value = true;
  try {
    const nextStatus = await updateBackendSettings(settings);
    applyBackendStatus(nextStatus);
    setSettingsFeedback(
      "success",
      idleSeconds === 0
        ? "设置已保存，下一次翻译会使用新的参数。自动释放已关闭。"
        : "设置已保存，下一次翻译会使用新的参数。",
    );
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存模型设置。");
  } finally {
    settingsLoading.value = false;
  }
}

// --- ModelScope 下载交互 ---
let downloadProgressUnlisten: (() => void) | undefined;
let mockTimers = new Map<string, ReturnType<typeof setInterval>>();

function handleDownloadProgress(payload: { modelId: string; source: DownloadSource; progress: number; downloadedBytes: number; totalBytes: number; status: string; message?: string }) {
  const statusMap: Record<string, DownloadTaskState["status"]> = {
    downloading: "downloading",
    completed: "completed",
    error: "error",
    cancelled: "cancelled",
    idle: "idle",
  };
  const normalizedStatus = (statusMap[payload.status] ?? "downloading") as DownloadTaskState["status"];
  downloadTasks.value[payload.modelId] = {
    modelId: payload.modelId,
    source: payload.source,
    status: normalizedStatus,
    progress: payload.progress,
    downloadedBytes: payload.downloadedBytes,
    totalBytes: payload.totalBytes,
    message: payload.message,
  };
  if (normalizedStatus === "completed") {
    setSettingsFeedback("success", `模型 ${payload.modelId} 下载完成，已落盘。`);
    void loadModelCatalog();
    void refreshBackendStatus(false);
    void refreshDownloaded();
  } else if (normalizedStatus === "error") {
    setSettingsFeedback("error", payload.message || `模型 ${payload.modelId} 下载失败。`);
  }
}

async function handleDownload(modelId: string) {
  if (downloadTasks.value[modelId]?.status === "downloading") return;
  if (!isDesktopRuntime) {
    let progress = 0;
    downloadTasks.value[modelId] = {
      modelId,
      source: downloadSource.value,
      status: "downloading",
      progress: 0,
      downloadedBytes: 0,
      totalBytes: 100,
    };
    const timer = setInterval(() => {
      progress += 7;
      if (progress >= 100) {
        progress = 100;
        downloadTasks.value[modelId] = {
          modelId,
          source: downloadSource.value,
          status: "completed",
          progress: 100,
          downloadedBytes: 100,
          totalBytes: 100,
          message: "下载完成（浏览器模拟）",
        };
        clearInterval(timer);
        mockTimers.delete(modelId);
        setSettingsFeedback("success", `模型 ${modelId} 下载完成（模拟）。`);
      } else {
        downloadTasks.value[modelId] = {
          modelId,
          source: downloadSource.value,
          status: "downloading",
          progress,
          downloadedBytes: progress,
          totalBytes: 100,
        };
      }
    }, 180);
    mockTimers.set(modelId, timer);
    return;
  }
  try {
    const state = await startModelDownload(modelId, downloadSource.value);
    downloadTasks.value[modelId] = state;
    setSettingsFeedback("info", `已开始从 ${downloadSource.value === "modelscope" ? "ModelScope" : "Hugging Face"} 下载 ${modelId}。`);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法开始下载。");
  }
}

async function handleCancelDownload(modelId: string) {
  const timer = mockTimers.get(modelId);
  if (timer) {
    clearInterval(timer);
    mockTimers.delete(modelId);
    downloadTasks.value[modelId] = {
      modelId,
      source: downloadSource.value,
      status: "cancelled",
      progress: downloadTasks.value[modelId]?.progress ?? 0,
      downloadedBytes: 0,
      totalBytes: 100,
      message: "已取消",
    };
    setSettingsFeedback("info", `已取消下载 ${modelId}。`);
    return;
  }
  try {
    await cancelModelDownload(modelId);
    const existing = downloadTasks.value[modelId];
    if (existing) {
      downloadTasks.value[modelId] = { ...existing, status: "cancelled", message: "已取消" };
    }
    setSettingsFeedback("info", `已取消下载 ${modelId}。`);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "取消失败。");
  }
}
async function handleDeleteModel(modelId: string): Promise<void> {
  if (!isDownloaded(modelId)) {
    setSettingsFeedback("error", "该模型尚未下载，无需删除");
    return;
  }
  if (isModelInstalled(modelId)) {
    setSettingsFeedback("error", "该模型当前已启用，请先切换到其他模型后再删除");
    return;
  }
  const task = downloadTasks.value[modelId];
  if (task?.status === "downloading") {
    setSettingsFeedback("error", "模型正在下载中，请先取消后再删除");
    return;
  }
  settingsLoading.value = true;
  try {
    await deleteDownloadedModel(modelId);
    await refreshDownloaded();
    // 清理对应的下载任务状态
    if (downloadTasks.value[modelId]) {
      delete downloadTasks.value[modelId];
    }
    setSettingsFeedback("success", `已删除模型 ${modelId}`);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : `删除失败 ${modelId}`);
  } finally {
    settingsLoading.value = false;
  }
}
onMounted(async () => {
  if (backendStatus.value) {
    applyBackendStatus(backendStatus.value);
  }
  void loadModelCatalog();
  void refreshBackendStatus(false);
  void refreshDownloaded();
  try {
    downloadProgressUnlisten = await listenDownloadProgress(handleDownloadProgress);
  } catch {
    // 浏览器预览无事件
  }
});


onBeforeUnmount(() => {
  downloadProgressUnlisten?.();
  mockTimers.forEach((timer) => clearInterval(timer));
  mockTimers.clear();
});
</script>

<template>
  <section class="model-manager-page" aria-labelledby="model-manager-title">
    <header class="model-manager-header">
      <div>
        <p class="panel-kicker">Model Management</p>
        <h2 id="model-manager-title">模型管理</h2>
        <p class="model-manager-intro">
          管理本地 Hy-MT2 与 PP-OCR 模型、下载模型、配置运行与生成参数。配置路径后点击“保存设置”生效。
        </p>
      </div>
      <div class="model-manager-header-actions">
        <n-select
          v-model:value="downloadSource"
          :options="downloadSourceOptions"
          size="small"
          style="width: 150px"
          aria-label="下载源"
        />
        <n-tag :type="settingsTagType" round size="small">{{ settingsStatusLabel }}</n-tag>
        <n-button secondary size="small" :loading="settingsLoading" @click="refreshBackendStatus()">刷新状态</n-button>
      </div>
    </header>

    <n-alert v-if="!isDesktopRuntime" type="info" title="桌面端功能" :show-icon="true">
      模型下载与路径配置仅在 Tauri 桌面端可用；浏览器预览仅展示界面。
    </n-alert>

    <div class="model-manager-grid">
      <!-- 翻译模型：平铺，下载与切换合并，两列布局，窄屏自动单列 -->
      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">Translation · Hy-MT2</p>
            <h2>翻译模型</h2>
          </div>
          <n-button secondary size="small" @click="openModelDialog('translation')">导入本地…</n-button>
        </div>
        <p class="settings-card-copy">
          共 {{ translationModels.length }} 个量化版本，自动检测 <code>downloads/&lt;modelId&gt;</code> 是否已下载；已下载可一键切换。
        </p>
        <div class="model-flat-list">
          <div v-for="model in translationModels" :key="model.id" class="model-flat-item">
            <div class="model-flat-main">
              <div class="model-flat-title">
                <strong>{{ model.name }}</strong>
                <n-tag size="tiny" type="info" round>{{ model.sizeText }}</n-tag>
                <n-tag v-if="model.recommended" size="tiny" type="success" round>推荐</n-tag>
                <n-tag v-if="isDownloaded(model.id)" size="tiny" type="info" round>已下载</n-tag>
                <n-tag v-if="isModelInstalled(model.id)" size="tiny" type="success" round>已启用</n-tag>
              </div>
              <div class="model-flat-desc">{{ model.description }} · {{ model.repoId }}</div>
              <n-progress
                v-if="downloadTaskFor(model.id)?.status === 'downloading'"
                :percentage="downloadTaskFor(model.id)?.progress ?? 0"
                :show-indicator="true"
                :height="6"
                style="margin-top: 6px"
              />
              <span v-if="downloadTaskFor(model.id)?.message" class="model-download-message">{{ downloadTaskFor(model.id)?.message }}</span>
              <n-tooltip v-else-if="isDownloaded(model.id) && downloadedBaseDir[model.id]" trigger="hover">
                <template #trigger>
                  <span class="model-flat-path" :title="downloadedBaseDir[model.id]">{{ downloadedBaseDir[model.id] }}</span>
                </template>
                {{ downloadedBaseDir[model.id] }}
              </n-tooltip>
            </div>
            <div class="model-flat-actions">
              <n-button v-if="downloadTaskFor(model.id)?.status === 'downloading'" secondary size="small" @click="handleCancelDownload(model.id)">取消</n-button>
              <template v-else-if="isModelInstalled(model.id)"><n-tag type="success" size="small" round>已启用</n-tag></template>
              <template v-else-if="isDownloaded(model.id)">
                <n-button secondary size="small" :loading="settingsLoading" @click="handleSwitchModel(model.id)">切换</n-button>
                <n-popconfirm @positive-click="handleDeleteModel(model.id)">
                  <template #trigger>
                    <n-button secondary size="small" type="error" :loading="settingsLoading" style="margin-left:6px;">删除</n-button>
                  </template>
                  确定删除 {{ model.name }}？本地文件将被移除。
                </n-popconfirm>
              </template>
              <template v-else><n-button secondary size="small" @click="handleDownload(model.id)">{{ downloadTaskFor(model.id)?.status === 'error' ? '重试' : '下载' }}</n-button><n-tag v-if="downloadTaskFor(model.id)?.status === 'error'" type="error" size="small" round style="margin-left:6px;">失败</n-tag></template>
              <n-tag v-if="downloadTaskFor(model.id)?.status === 'completed' && !isDownloaded(model.id)" type="success" size="small" round style="margin-left:6px;">已完成</n-tag>
            </div>
          </div>
        </div>
      </n-card>

      <!-- OCR 模型：平铺全部 PP-OCR，下载与切换合并 -->
      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">OCR · PaddleOCR</p>
            <h2>OCR 模型</h2>
          </div>
          <n-button secondary size="small" @click="openModelDialog('ocr')">导入本地…</n-button>
        </div>
        <p class="settings-card-copy">
          共 {{ ocrModels.length }} 个规格（V5/V6），自动检测是否已下载；已下载可一键切换。
        </p>
        <div class="model-flat-list">
          <div v-for="model in ocrModels" :key="model.id" class="model-flat-item">
            <div class="model-flat-main">
              <div class="model-flat-title">
                <strong>{{ model.name }}</strong>
                <n-tag size="tiny" type="warning" round>{{ model.sizeText }}</n-tag>
                <n-tag v-if="model.recommended" size="tiny" type="success" round>推荐</n-tag>
                <n-tag v-if="isDownloaded(model.id)" size="tiny" type="info" round>已下载</n-tag>
                <n-tag v-if="isModelInstalled(model.id)" size="tiny" type="success" round>已启用</n-tag>
              </div>
              <div class="model-flat-desc">{{ model.description }} · {{ model.repoId }}</div>
              <n-progress
                v-if="downloadTaskFor(model.id)?.status === 'downloading'"
                :percentage="downloadTaskFor(model.id)?.progress ?? 0"
                :show-indicator="true"
                :height="6"
                style="margin-top: 6px"
              />
              <span v-if="downloadTaskFor(model.id)?.message" class="model-download-message">{{ downloadTaskFor(model.id)?.message }}</span>
              <n-tooltip v-else-if="isDownloaded(model.id) && downloadedBaseDir[model.id]" trigger="hover">
                <template #trigger>
                  <span class="model-flat-path" :title="downloadedBaseDir[model.id]">{{ downloadedBaseDir[model.id] }}</span>
                </template>
                {{ downloadedBaseDir[model.id] }}
              </n-tooltip>
            </div>
            <div class="model-flat-actions">
              <n-button v-if="downloadTaskFor(model.id)?.status === 'downloading'" secondary size="small" @click="handleCancelDownload(model.id)">取消</n-button>
              <template v-else-if="isModelInstalled(model.id)"><n-tag type="success" size="small" round>已启用</n-tag></template>
              <template v-else-if="isDownloaded(model.id)">
                <n-button secondary size="small" :loading="settingsLoading" @click="handleSwitchModel(model.id)">切换</n-button>
                <n-popconfirm @positive-click="handleDeleteModel(model.id)">
                  <template #trigger>
                    <n-button secondary size="small" type="error" :loading="settingsLoading" style="margin-left:6px;">删除</n-button>
                  </template>
                  确定删除 {{ model.name }}？本地文件将被移除。
                </n-popconfirm>
              </template>
              <template v-else><n-button secondary size="small" @click="handleDownload(model.id)">{{ downloadTaskFor(model.id)?.status === 'error' ? '重试' : '下载' }}</n-button><n-tag v-if="downloadTaskFor(model.id)?.status === 'error'" type="error" size="small" round style="margin-left:6px;">失败</n-tag></template>
              <n-tag v-if="downloadTaskFor(model.id)?.status === 'completed' && !isDownloaded(model.id)" type="success" size="small" round style="margin-left:6px;">已完成</n-tag>
            </div>
          </div>
        </div>
        <p class="settings-help">切换后持久化到 <code>model-settings.json</code> 并清空引擎，下次推理自动加载。</p>
      </n-card>

      <!-- 标注字体（保留手动导入） -->
      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">Fonts</p>
            <h2>标注字体</h2>
          </div>
          <n-button secondary size="small" @click="openModelDialog('font')">导入字体…</n-button>
        </div>
        <dl class="settings-path-list" style="margin-top:12px;">
          <div>
            <dt>当前字体</dt>
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
              <n-tooltip v-if="modelFontPath" trigger="hover">
                <template #trigger>
                  <span class="settings-model-help" :title="modelFontPath" style="max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; display:inline-block; vertical-align:bottom;">{{ modelFontPath }}</span>
                </template>
                {{ modelFontPath }}
              </n-tooltip>
              <span v-else class="settings-model-help">系统自动匹配</span>
            </dd>
          </div>
        </dl>
      </n-card>
<n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">运行资源</p>
            <h2>设备、批处理与释放</h2>
          </div>
          <n-tag type="info" round size="small">CUDA</n-tag>
        </div>
        <p class="settings-card-copy">
          控制本地推理设备、OCR 区域并发、Hy 翻译批大小，以及翻译完成后保持模型在显存中的秒数。
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
            <span>空闲释放时间（秒）</span>
            <n-input-number
              v-model:value="idleUnloadSeconds"
              :min="IDLE_UNLOAD_SECONDS_MIN"
              :max="IDLE_UNLOAD_SECONDS_MAX"
              :step="30"
              aria-label="模型空闲释放时间（秒）"
            />
            <span class="settings-help">设置为 0 可关闭自动卸载；超时后模型会从显存中释放。</span>
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
  </section>
</template>

<style scoped src="../styles/model-manager-page.css"></style>
