<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, ref } from "vue";
import { NAlert, NButton, NCard, NSelect, NSpin, NTag, useMessage } from "naive-ui";
import {
  controlModel,
  getModelRuntimeStatus,
  type ModelAction,
  type ModelRuntimeStatus,
  type ModelTarget,
} from "../services/translation-provider";
import TargetLanguageField from "./TargetLanguageField.vue";
import {
  HUGGINGFACE_DOWNLOADABLE_MODELS,
  MODELSCOPE_DOWNLOADABLE_MODELS,
  activateDownloadedModel,
  listDownloadedModels,
  translationEngineDisplayName,
  type DownloadedModelInfo,
} from "../services/model-download-provider";
import { applySharedModelRuntimeStatus } from "../services/workspace-settings";
import { showWorkspaceToast } from "../services/workspace-toast";

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const toast = useMessage();
const runtimeStatus = ref<ModelRuntimeStatus | null>(null);
const refreshing = ref(false);
const controlKey = ref("");
const errorMessage = ref("");
let refreshTimer: ReturnType<typeof setInterval> | undefined;

const loadedModelCount = computed(() => {
  const status = runtimeStatus.value;
  return status ? Number(status.ocrLoaded) + Number(status.translatorLoaded) : 0;
});

const translationEngineLabel = computed(() =>
  translationEngineDisplayName(runtimeStatus.value?.backend?.hyModel),
);

const downloadedModels = ref<DownloadedModelInfo[]>([]);
const switchKey = ref("");

const translationManifestIds = computed(() => {
  const ids = new Set<string>();
  for (const m of MODELSCOPE_DOWNLOADABLE_MODELS) {
    if (m.kind === "translation") ids.add(m.id);
  }
  for (const m of HUGGINGFACE_DOWNLOADABLE_MODELS) {
    if (m.kind === "translation") ids.add(m.id);
  }
  return ids;
});

const downloadedTranslationModels = computed(() =>
  downloadedModels.value.filter((m) => m.downloaded && translationManifestIds.value.has(m.modelId)),
);

function displayNameFor(modelId: string): string {
  const found =
    MODELSCOPE_DOWNLOADABLE_MODELS.find((m) => m.id === modelId) ??
    HUGGINGFACE_DOWNLOADABLE_MODELS.find((m) => m.id === modelId);
  return found?.name ?? modelId;
}

function normPath(s: string): string {
  return s.replace(/\\/g, "/").toLowerCase();
}

const ocrManifestIds = computed(() => {
  const ids = new Set<string>();
  for (const m of MODELSCOPE_DOWNLOADABLE_MODELS) {
    if (m.kind === "ocr") ids.add(m.id);
  }
  for (const m of HUGGINGFACE_DOWNLOADABLE_MODELS) {
    if (m.kind === "ocr") ids.add(m.id);
  }
  return ids;
});

const downloadedOcrModels = computed(() =>
  downloadedModels.value.filter((m) => m.downloaded && ocrManifestIds.value.has(m.modelId)),
);

const activeTranslationModelId = computed<string | null>(() => {
  const cur = normPath((runtimeStatus.value?.backend?.hyModel || "").trim());
  if (!cur) return null;
  return (
    downloadedTranslationModels.value.find((m) => {
      const base = normPath(m.baseDir);
      return base !== "" && (cur === base || cur.startsWith(base + "/"));
    })?.modelId ?? null
  );
});

const activeOcrModelId = computed<string | null>(() => {
  const cur = normPath((runtimeStatus.value?.backend?.detectorModelDir || "").trim());
  if (!cur) return null;
  return (
    downloadedOcrModels.value.find((m) => {
      const base = normPath(m.baseDir);
      return base !== "" && (cur === base || cur.startsWith(base + "/"));
    })?.modelId ?? null
  );
});

const translationOptions = computed(() =>
  downloadedTranslationModels.value.map((m) => ({ label: displayNameFor(m.modelId), value: m.modelId })),
);

const ocrOptions = computed(() =>
  downloadedOcrModels.value.map((m) => ({ label: displayNameFor(m.modelId), value: m.modelId })),
);

async function refreshDownloaded(): Promise<void> {
  if (!isDesktopRuntime) {
    downloadedModels.value = [];
    return;
  }
  try {
    downloadedModels.value = await listDownloadedModels();
  } catch {
    downloadedModels.value = [];
  }
}

function handleTargetLanguageSaved(lang: string): void {
  void refreshStatus();
  showWorkspaceToast(toast, "success", `目标语言已切换为 ${lang}。`);
}

function handleTargetLanguageFailed(message: string): void {
  errorMessage.value = message;
  showWorkspaceToast(toast, "error", message);
}

async function quickSwitchModel(modelId: string | null): Promise<void> {
  if (!isDesktopRuntime || modelId == null || switchKey.value || runtimeStatus.value?.busy) {
    return;
  }
  if (modelId === activeTranslationModelId.value || modelId === activeOcrModelId.value) {
    showWorkspaceToast(toast, "info", "该模型已是当前启用模型");
    return;
  }
  switchKey.value = modelId;
  try {
    await activateDownloadedModel(modelId);
    await refreshStatus();
    await refreshDownloaded();
    showWorkspaceToast(toast, "success", `已切换到模型 ${displayNameFor(modelId)}。`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    errorMessage.value = message;
    showWorkspaceToast(toast, "error", message);
  } finally {
    switchKey.value = "";
  }
}

const runtimeLabel = computed(() => {
  const status = runtimeStatus.value;
  if (!status) {
    return isDesktopRuntime ? "读取中" : "桌面端可用";
  }
  if (status.busy) {
    return "处理中";
  }
  return loadedModelCount.value > 0 ? `已加载 ${loadedModelCount.value}/2` : "按需加载";
});

const runtimeTagType = computed<"default" | "success" | "warning" | "info">(() => {
  if (!runtimeStatus.value) {
    return isDesktopRuntime ? "warning" : "default";
  }
  return runtimeStatus.value.busy ? "warning" : "success";
});

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function refreshStatus(showError = true): Promise<void> {
  if (!isDesktopRuntime || refreshing.value || controlKey.value) {
    return;
  }
  refreshing.value = true;
  try {
    const status = await getModelRuntimeStatus();
    runtimeStatus.value = status;
    applySharedModelRuntimeStatus(status);
    errorMessage.value = "";
  } catch (error) {
    if (showError) {
      errorMessage.value = errorText(error);
    }
  } finally {
    refreshing.value = false;
  }
}

async function runModelControl(model: ModelTarget, action: ModelAction): Promise<void> {
  if (!isDesktopRuntime || controlKey.value) {
    return;
  }
  const key = `${model}:${action}`;
  controlKey.value = key;
  try {
    const status = await controlModel(model, action);
    runtimeStatus.value = status;
    applySharedModelRuntimeStatus(status);
    errorMessage.value = "";
    const modelLabel = model === "ocr" ? "PP-OCR" : translationEngineLabel.value;
    showWorkspaceToast(toast, "success", `${modelLabel} 已${action === "load" ? "加载" : "卸载"}。`);
  } catch (error) {
    const message = errorText(error);
    errorMessage.value = message;
    showWorkspaceToast(toast, "error", message);
  } finally {
    controlKey.value = "";
  }
}

function startPolling(): void {
  if (!isDesktopRuntime || refreshTimer) {
    return;
  }
  void refreshStatus();
  void refreshDownloaded();
  refreshTimer = setInterval(() => void refreshStatus(false), 2000);
}

function stopPolling(): void {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = undefined;
  }
}

onActivated(startPolling);
onDeactivated(stopPolling);
onBeforeUnmount(stopPolling);
</script>

<template>
  <section class="model-monitor-page" aria-labelledby="model-monitor-page-title">
    <header class="monitor-toolbar">
      <p>管理本地 OCR 与翻译模型的驻留状态；模型安装与参数配置请在模型管理中完成。</p>
      <div class="monitor-toolbar-actions">
        <n-tag :type="runtimeTagType" round size="small">{{ runtimeLabel }}</n-tag>
        <n-button secondary :loading="refreshing" :disabled="!isDesktopRuntime" @click="refreshStatus()">
          刷新状态
        </n-button>
      </div>
    </header>

    <n-alert v-if="!isDesktopRuntime" type="info" :show-icon="true">
      模型运行状态来自 Tauri 后端，请在桌面应用中查看和控制。
    </n-alert>
    <n-alert v-else-if="errorMessage" type="error" title="无法读取模型状态" :show-icon="true">
      {{ errorMessage }}
    </n-alert>

    <div v-if="!runtimeStatus && isDesktopRuntime" class="monitor-loading" aria-live="polite">
      <n-spin size="small" />
      <span>正在读取模型运行状态…</span>
    </div>

    <template v-if="runtimeStatus">
      <div class="runtime-summary" role="status" aria-live="polite">
        <div class="runtime-state">
          <span class="runtime-state-dot" :class="{ 'runtime-state-dot-busy': runtimeStatus.busy }"
            aria-hidden="true"></span>
          <div>
            <span>Runtime 状态</span>
            <strong>{{ runtimeStatus.busy ? "正在推理" : "空闲" }}</strong>
          </div>
        </div>
        <dl class="runtime-facts">
          <div>
            <dt>运行设备</dt>
            <dd>{{ runtimeStatus.backend.device.toUpperCase() }}</dd>
          </div>
          <div>
            <dt>驻留模型</dt>
            <dd>{{ loadedModelCount }} / 2</dd>
          </div>
          <div>
            <dt>自动刷新</dt>
            <dd>2,000 ms</dd>
          </div>
          <div v-if="runtimeStatus.processMemoryMib != null">
            <dt>进程内存</dt>
            <dd>{{ runtimeStatus.processMemoryMib }} MiB</dd>
          </div>
        </dl>
      </div>

      <section class="monitor-section" aria-labelledby="loaded-models-title">
        <div class="section-heading">
          <div>
            <p class="panel-kicker">Model Residency</p>
            <h3 id="loaded-models-title">模型驻留</h3>
          </div>
          <span>推理过程中暂不可加载或卸载模型</span>
        </div>

        <div class="model-card-grid">
          <n-card class="monitor-card model-card" :class="{ 'model-card-loaded': runtimeStatus.ocrLoaded }"
            :bordered="false">
            <div class="model-card-heading">
              <div class="model-identity">
                <span class="model-kind">OCR</span>
                <div>
                  <h4>PP-OCR</h4>
                  <p>文字检测与识别</p>
                </div>
              </div>
              <n-tag :type="runtimeStatus.ocrLoaded ? 'success' : 'default'" round size="small">
                {{ runtimeStatus.ocrLoaded ? "已加载" : "未加载" }}
              </n-tag>
            </div>

            <dl class="model-paths">
              <div>
                <dt>Detector</dt>
                <dd><code>{{ runtimeStatus.backend.detectorModelDir }}</code></dd>
              </div>
              <div>
                <dt>Recognizer</dt>
                <dd><code>{{ runtimeStatus.backend.recognizerModelDir }}</code></dd>
              </div>
            </dl>

            <div v-if="downloadedOcrModels.length" class="quick-switch-field" style="padding: 0.5rem 0;">
              <n-select :value="activeOcrModelId" :options="ocrOptions" placeholder="选择 OCR 模型切换"
                :disabled="runtimeStatus.busy || Boolean(switchKey) || Boolean(controlKey)"
                :loading="Boolean(switchKey)" @update:value="quickSwitchModel" />
            </div>

            <div class="model-card-footer">
              <span>{{ runtimeStatus.ocrLoaded ? "模型已驻留，可直接执行 OCR" : "首次请求时会自动加载" }}</span>
              <n-button v-if="runtimeStatus.ocrLoaded" secondary type="warning" :loading="controlKey === 'ocr:unload'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)" @click="runModelControl('ocr', 'unload')">
                卸载 OCR
              </n-button>
              <n-button v-else type="primary" :loading="controlKey === 'ocr:load'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)" @click="runModelControl('ocr', 'load')">
                加载 OCR
              </n-button>
            </div>
          </n-card>

          <n-card class="monitor-card model-card" :class="{ 'model-card-loaded': runtimeStatus.translatorLoaded }"
            :bordered="false">
            <div class="model-card-heading">
              <div class="model-identity">
                <span class="model-kind">MT</span>
                <div>
                  <h4>{{ translationEngineLabel }}</h4>
                  <p>本地文本翻译</p>
                </div>
              </div>
              <n-tag :type="runtimeStatus.translatorLoaded ? 'success' : 'default'" round size="small">
                {{ runtimeStatus.translatorLoaded ? "已加载" : "未加载" }}
              </n-tag>
            </div>

            <dl class="model-paths">
              <div>
                <dt>GGUF 模型</dt>
                <dd><code>{{ runtimeStatus.backend.hyModel }}</code></dd>
              </div>
              <div>
                <dt>目标语言</dt>
                <dd>
                  <TargetLanguageField immediate aria-label="监控页翻译目标语言"
                    :disabled="runtimeStatus.busy || Boolean(switchKey) || Boolean(controlKey)"
                    @saved="handleTargetLanguageSaved" @failed="handleTargetLanguageFailed" />
                </dd>
              </div>
            </dl>

            <div v-if="downloadedTranslationModels.length" class="quick-switch-field" style="padding: 0.5rem 0;">
              <n-select :value="activeTranslationModelId" :options="translationOptions" placeholder="选择翻译模型切换"
                :disabled="runtimeStatus.busy || Boolean(switchKey) || Boolean(controlKey)"
                :loading="Boolean(switchKey)" @update:value="quickSwitchModel" />
            </div>

            <div class="model-card-footer">
              <span>{{ runtimeStatus.translatorLoaded ? "模型已驻留，可直接执行翻译" : "首次请求时会自动加载" }}</span>
              <n-button v-if="runtimeStatus.translatorLoaded" secondary type="warning"
                :loading="controlKey === 'translator:unload'" :disabled="runtimeStatus.busy || Boolean(controlKey)"
                @click="runModelControl('translator', 'unload')">
                卸载 {{ translationEngineLabel }}
              </n-button>
              <n-button v-else type="primary" :loading="controlKey === 'translator:load'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)" @click="runModelControl('translator', 'load')">
                加载 {{ translationEngineLabel }}
              </n-button>
            </div>
          </n-card>
        </div>
      </section>

    </template>
  </section>
</template>

<style scoped src="../styles/model-monitor-page.css"></style>
