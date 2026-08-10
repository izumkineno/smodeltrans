<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { open as openNativeDialog } from "@tauri-apps/plugin-dialog";
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NInputNumber,
  NSelect,
  NSwitch,
  NTag,
  useMessage,
} from "naive-ui";
import {
  updateBackendSettings,
} from "../services/translation-provider";
import type {
  BackendSettingsUpdate,
  BackendStatus,
  DeviceKind,
} from "../services/translation-provider";
import {
  applySharedBackendStatus,
  backendStatus,
  fetchSharedBackendStatus,
  loadPersistedTargetLanguage,
  savePersistedTargetLanguage,
  targetLanguage,
} from "../services/workspace-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";

type TagType = "default" | "success" | "warning" | "error" | "info";
type ModelPathField = "detector" | "recognizer" | "hy" | "font";

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const modelDetectorPath = ref("");
const modelRecognizerPath = ref("");
const modelHyPath = ref("");
const modelFontPath = ref<string | null>(null);
const device = ref<DeviceKind>("cuda");
const regionParallelism = ref(16);
const translationBatchSize = ref(4);
const idleUnloadMinutes = ref(30);
const generationMaxNewTokens = ref(128);
const generationSampling = ref(false);
const generationTemperature = ref(1);
const generationTopK = ref(0);
const generationTopP = ref(1);
const generationSeed = ref("");
const generationRepetitionPenalty = ref(1);
const generationFrequencyPenalty = ref(0);
const generationStopTokensText = ref("");
const generationStopStringsText = ref("");
const memoryEnabled = ref(false);
const memoryMaxTokens = ref(4096);
const memoryMaxTurns = ref(16);
const systemPrompt = ref("");
const userPrompt = ref("");
const settingsMessage = ref("");
const settingsLoading = ref(false);
const deviceOptions: Array<{ label: string; value: DeviceKind }> = [
  { label: "CUDA", value: "cuda" },
  { label: "CPU（仅用于状态检查；Hy 翻译需要 CUDA）", value: "cpu" },
];


const toast = useMessage();

function setSettingsFeedback(
  type: WorkspaceToastType,
  message: string,
  notify = true,
) {
  settingsMessage.value = message;
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
  regionParallelism.value = status.regionParallelism;
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
    setSettingsFeedback(status.ready ? "success" : "warning", status.message, notify);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法读取后端模型状态。", notify);
  } finally {
    settingsLoading.value = false;
  }
}

function modelPathFor(field: ModelPathField): string {
  if (field === "detector") {
    return modelDetectorPath.value;
  }
  if (field === "recognizer") {
    return modelRecognizerPath.value;
  }
  if (field === "hy") {
    return modelHyPath.value;
  }
  return modelFontPath.value ?? "";
}

async function chooseModelPath(field: ModelPathField) {
  if (!isDesktopRuntime) {
    setSettingsFeedback("warning", "模型路径选择仅在 Tauri 桌面端可用。");
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
        : field === "font"
          ? await openNativeDialog({
              title: "选择标注字体",
              defaultPath: currentPath || undefined,
              multiple: false,
              filters: [{ name: "字体文件", extensions: ["ttf", "otf"] }],
            })
          : await openNativeDialog({
              title:
                field === "detector"
                  ? "选择 PP-OCRv5 detector 文件夹"
                  : "选择 PP-OCRv5 recognizer 文件夹",
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
    } else if (field === "hy") {
      modelHyPath.value = selected;
    } else {
      modelFontPath.value = selected;
    }
    setSettingsFeedback("success", "路径已选择，点击保存模型设置后生效。");
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法打开模型路径选择器。");
  }
}

function useSystemFont() {
  modelFontPath.value = null;
  setSettingsFeedback("info", "字体已切换为系统自动匹配，点击保存模型设置后生效。");
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
    setSettingsFeedback("error", "请选择完整的 PP-OCRv5 与 Hy-MT2 模型路径。");
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
        ? "模型设置已保存，下一次翻译会使用新的参数。自动释放已关闭。"
        : "模型设置已保存，下一次翻译会使用新的参数。",
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
  void refreshBackendStatus(false);
});

</script>

<template>
  <section class="settings-page" aria-labelledby="settings-title">
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
          <div>
            <span>OCR 并发</span>
            <strong>{{ backendStatus?.regionParallelism ?? regionParallelism }}</strong>
          </div>
          <div>
            <span>Hy 批大小</span>
            <strong>{{ backendStatus?.translationBatchSize ?? translationBatchSize }}</strong>
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
            <h2>翻译默认值</h2>
          </div>
        </div>
        <p class="settings-card-copy">目标语言、system 预设提示词和 user 预设提示词会作为下一次翻译请求的默认模型上下文。</p>
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
            <p class="panel-kicker">模型资源</p>
            <h2>本地模型路径</h2>
          </div>
        </div>
        <p class="settings-card-copy">
          选择后端实际使用的 PP-OCRv5 文件夹、Hy-MT2 GGUF 文件和可选字体；保存后立即应用，下一次翻译会按新路径加载。
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
          <div>
            <dt>标注字体</dt>
            <dd>
              <span class="settings-path-value">{{ modelFontPath || "系统自动匹配" }}</span>
              <n-button secondary size="small" @click="chooseModelPath('font')">选择字体</n-button>
              <n-button tertiary size="small" @click="useSystemFont">使用系统字体</n-button>
            </dd>
          </div>
        </dl>
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
        <p class="settings-card-copy">
          Greedy 模式忽略 top-k；开启 sampling 时 top-k 必须大于 0，且最大为 1024。
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
        <p class="settings-card-copy">
          Stop tokens 用逗号或空白分隔；stop strings 每行一条。记忆关闭时仍保存预算，方便后续重新启用。
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

      <div class="settings-card-actions settings-page-actions settings-card-wide">
        <n-button secondary :loading="settingsLoading" @click="refreshBackendStatus()">刷新状态</n-button>
        <n-button type="primary" :loading="settingsLoading" @click="saveModelSettings">
          保存模型设置
        </n-button>
      </div>
    </div>
  </section>
</template>
