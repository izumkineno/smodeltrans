<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { open as openNativeDialog } from "@tauri-apps/plugin-dialog";
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NModal,
  NPopconfirm,
  NSelect,
  NTooltip,
  useMessage,
} from "naive-ui";
import { getModelCatalog, saveModelCatalog, updateBackendSettings } from "../services/translation-provider";
import type { BackendSettingsUpdate, BackendStatus, ModelCatalogOptions, ModelCatalogUpdate } from "../services/translation-provider";
import {
  applySharedBackendStatus,
  backendStatus,
  fetchSharedBackendStatus,
  savePersistedTargetLanguage,
  targetLanguage,
} from "../services/workspace-settings";
import { setThemeMode, themeMode } from "../services/theme-settings";
import type { ThemeMode } from "../services/theme-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";
import { isSupportedTargetLanguage } from "../constants/targetLanguageOptions";
import OpenAiCompatCard from "./OpenAiCompatCard.vue";
import OpenAiRequestHistory from "./OpenAiRequestHistory.vue";
import TargetLanguageSelect from "./TargetLanguageSelect.vue";
const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const promptTemplate = ref("");
const settingsMessage = ref("");
const settingsMessageType = ref<WorkspaceToastType>("info");
const settingsLoading = ref(false);
const modelFontPath = ref<string | null>(null);
const modelCatalog = ref<ModelCatalogOptions>({ translation: [], ocr: [], fonts: [] });
const catalogLoaded = ref(false);
const dialogSaving = ref(false);
const dialogName = ref("");
const dialogFontPath = ref("");
const fontDialogOpen = ref(false);
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

function pathBaseName(path: string): string {
  const trimmed = path.replace(/[\\/]+$/, "");
  const separator = Math.max(trimmed.lastIndexOf("\\"), trimmed.lastIndexOf("/"));
  return separator >= 0 ? trimmed.slice(separator + 1) : trimmed;
}

const SYSTEM_FONT_VALUE = "__system__";
const fontModelOptions = computed(() => {
  const opts = [
    { label: "系统自动匹配", value: SYSTEM_FONT_VALUE },
    ...modelCatalog.value.fonts.map((option) => ({
      label: `${option.name}（${option.path ? pathBaseName(option.path) : ""}）`,
      value: option.path ?? SYSTEM_FONT_VALUE,
    })),
  ];
  const current = modelFontPath.value;
  if (current && !opts.some((o) => o.value === current)) {
    opts.push({ label: `${pathBaseName(current)}（当前生效）`, value: current });
  }
  return opts;
});
const selectedFontValue = computed(() => modelFontPath.value ?? SYSTEM_FONT_VALUE);

async function loadModelCatalog(): Promise<void> {
  console.debug("[SettingsPage] loadModelCatalog: loading catalog", { isDesktopRuntime });
  if (!isDesktopRuntime) {
    catalogLoaded.value = true;
    console.info("[SettingsPage] loadModelCatalog: skipped (browser preview)");
    return;
  }
  try {
    modelCatalog.value = await getModelCatalog();
    console.info("[SettingsPage] loadModelCatalog: loaded", {
      translation: modelCatalog.value.translation.length,
      ocr: modelCatalog.value.ocr.length,
      fonts: modelCatalog.value.fonts.length,
    });
  } catch (error) {
    console.warn("[SettingsPage] loadModelCatalog: failed, fallback to empty", {
      error: error instanceof Error ? error.message : String(error),
    });
    modelCatalog.value = { translation: [], ocr: [], fonts: [] };
  } finally {
    catalogLoaded.value = true;
  }
}

function selectFontModel(value: string): void {
  console.info("[SettingsPage] selectFontModel: user selected font", { value });
  console.debug("[SettingsPage] selectFontModel: params", { value, systemValue: SYSTEM_FONT_VALUE });
  modelFontPath.value = value === SYSTEM_FONT_VALUE ? null : value;
  setSettingsFeedback("info", "已选择标注字体，点击“保存设置”生效。");
}

function openFontDialog(): void {
  console.info("[SettingsPage] openFontDialog: user opened font dialog");
  dialogName.value = "";
  dialogFontPath.value = "";
  fontDialogOpen.value = true;
}

function closeFontDialog(): void {
  console.debug("[SettingsPage] closeFontDialog: closing dialog", { wasOpen: fontDialogOpen.value });
  fontDialogOpen.value = false;
}

async function pickDialogFontPath(): Promise<void> {
  console.info("[SettingsPage] pickDialogFontPath: user triggered file picker");
  console.debug("[SettingsPage] pickDialogFontPath: currentFontPath", { current: dialogFontPath.value });
  const selected = await openNativeDialog({
    title: "选择标注字体",
    defaultPath: dialogFontPath.value || undefined,
    multiple: false,
    filters: [{ name: "字体文件", extensions: ["ttf", "otf"] }],
  });
  if (typeof selected === "string" && selected.trim()) {
    console.info("[SettingsPage] pickDialogFontPath: selected", { path: selected });
    dialogFontPath.value = selected;
  } else {
    console.debug("[SettingsPage] pickDialogFontPath: no selection or cancelled");
  }
}

async function saveFontDialog(): Promise<void> {
  console.info("[SettingsPage] saveFontDialog: user triggered save font entry", {
    fontName: dialogName.value.trim(),
    fontPath: dialogFontPath.value.trim(),
  });
  if (dialogSaving.value) {
    console.warn("[SettingsPage] saveFontDialog: already saving, ignored");
    return;
  }
  if (!catalogLoaded.value) {
    console.debug("[SettingsPage] saveFontDialog: catalog not loaded, loading first");
    await loadModelCatalog();
  }
  const entryName = dialogName.value.trim();
  if (!entryName) {
    console.warn("[SettingsPage] saveFontDialog: validation failed - missing font name");
    setSettingsFeedback("error", "请输入字体名称。");
    return;
  }
  const path = dialogFontPath.value.trim();
  if (!path) {
    console.warn("[SettingsPage] saveFontDialog: validation failed - missing font path");
    setSettingsFeedback("error", "请选择字体文件。");
    return;
  }
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
  const normalized = path.replace(/\\/g, "/").toLowerCase();
  const exists = next.fonts.some((e) => e.path.replace(/\\/g, "/").toLowerCase() === normalized);
  if (!exists) {
    next.fonts.push({ name: entryName, path });
  } else {
    next.fonts = next.fonts.map((e) =>
      e.path.replace(/\\/g, "/").toLowerCase() === normalized ? { name: entryName, path } : e,
    );
  }
  console.debug("[SettingsPage] saveFontDialog: saving catalog", { entryName, path, exists: exists ? "update" : "create", fontsCount: next.fonts.length });
  dialogSaving.value = true;
  try {
    await saveModelCatalog(next);
    await loadModelCatalog();
    // 从重载后的 catalog 中精确定位新条目的实际存储路径，避免大小写/斜杠差异导致选中失效
    const normalized = path.replace(/\\/g, "/").toLowerCase();
    const matched = modelCatalog.value.fonts.find(
      (e) => e.path && e.path.replace(/\\/g, "/").toLowerCase() === normalized,
    );
    const toSelect = matched?.path ?? path;
    console.info("[SettingsPage] saveFontDialog: catalog reloaded, selecting", { toSelect, matched: !!matched, fonts: modelCatalog.value.fonts.length });
    selectFontModel(toSelect);
    closeFontDialog();
    console.info("[SettingsPage] saveFontDialog: saved successfully", { entryName, path: toSelect });
    setSettingsFeedback("success", `已配置「${entryName}」并选中，点击“保存设置”生效。`);
  } catch (error) {
    console.error("[SettingsPage] saveFontDialog: failed", { error: error instanceof Error ? error.message : String(error), entryName, path });
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存模型条目。");
  } finally {
    dialogSaving.value = false;
  }
}

async function deleteSelectedFont(): Promise<void> {
  const current = modelFontPath.value;
  console.info("[SettingsPage] deleteSelectedFont: user triggered delete", { current, selectedFontValue: selectedFontValue.value });
  if (!current) {
    console.warn("[SettingsPage] deleteSelectedFont: no custom font selected");
    setSettingsFeedback("warning", "当前为系统字体，无需删除。");
    return;
  }
  if (!catalogLoaded.value) {
    await loadModelCatalog();
  }
  const normalized = current.replace(/\\/g, "/").toLowerCase();
  const exists = modelCatalog.value.fonts.some((e) => e.path && e.path.replace(/\\/g, "/").toLowerCase() === normalized);
  if (!exists) {
    console.warn("[SettingsPage] deleteSelectedFont: font not in catalog", { current });
    setSettingsFeedback("error", "该字体不在已导入列表中。");
    return;
  }
  if (dialogSaving.value) {
    console.warn("[SettingsPage] deleteSelectedFont: busy");
    return;
  }
  const next: ModelCatalogUpdate = {
    translation: modelCatalog.value.translation.map((entry) => ({ name: entry.name, path: entry.path })),
    ocr: modelCatalog.value.ocr.map((entry) => ({ name: entry.name, detectorDir: entry.detectorDir, recognizerDir: entry.recognizerDir })),
    fonts: modelCatalog.value.fonts
      .filter((entry) => entry.path && entry.path.replace(/\\/g, "/").toLowerCase() !== normalized)
      .filter((entry) => entry.path !== null)
      .map((entry) => ({ name: entry.name, path: entry.path as string })),
  };
  console.debug("[SettingsPage] deleteSelectedFont: saving catalog after filter", { before: modelCatalog.value.fonts.length, after: next.fonts.length, deleted: current });
  dialogSaving.value = true;
  try {
    await saveModelCatalog(next);
    await loadModelCatalog();
    // 若删除的是当前选中字体，则回退到系统字体
    const stillExists = modelCatalog.value.fonts.some((e) => e.path && e.path.replace(/\\/g, "/").toLowerCase() === normalized);
    if (!stillExists) {
      console.info("[SettingsPage] deleteSelectedFont: deleted font was selected, resetting to system", { deleted: current });
      modelFontPath.value = null;
      setSettingsFeedback("success", "已删除该字体并切回系统字体，点击“保存设置”生效。");
    } else {
      setSettingsFeedback("success", "已删除该字体。");
    }
    console.info("[SettingsPage] deleteSelectedFont: deleted successfully", { deleted: current, remaining: modelCatalog.value.fonts.length });
  } catch (error) {
    console.error("[SettingsPage] deleteSelectedFont: failed", { error: error instanceof Error ? error.message : String(error), current });
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法删除字体。");
  } finally {
    dialogSaving.value = false;
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

function handleThemeModeChange(nextMode: ThemeMode | null): void {
  console.info("[SettingsPage] handleThemeModeChange: user requested theme change", { nextMode, current: themeMode.value });
  if (!nextMode) {
    console.warn("[SettingsPage] handleThemeModeChange: ignored null mode");
    return;
  }
  const persistError = setThemeMode(nextMode);
  if (persistError) {
    console.error("[SettingsPage] handleThemeModeChange: persist failed", { nextMode, error: persistError });
    setSettingsFeedback("error", persistError);
    return;
  }
  console.info("[SettingsPage] handleThemeModeChange: theme changed successfully", { nextMode, label: themeModeLabels[nextMode] });
  setSettingsFeedback("success", `界面主题已切换为${themeModeLabels[nextMode]}。`, false);
}


function applyBackendStatus(status: BackendStatus) {
  console.debug("[SettingsPage] applyBackendStatus: applying", {
    ready: status.ready,
    targetLanguage: status.targetLanguage,
    device: status.device,
    fontPath: status.fontPath,
    detectorModelDir: status.detectorModelDir,
    hyModel: status.hyModel,
  });
  applySharedBackendStatus(status);
  modelFontPath.value = status.fontPath ?? null;
  // targetLanguage is global ref, keep in sync
  if (status.targetLanguage) {
    targetLanguage.value = status.targetLanguage;
  }
  // 兼容旧字段：优先 `template`，回退 `prompt`/`system`+`user`
  const p: any = status.prompt as any;
  promptTemplate.value = p.template ?? p.prompt ?? [p.system, p.user].filter(Boolean).join("\n\n") ?? "";
}
async function refreshBackendStatus(notify = true) {
  console.info("[SettingsPage] refreshBackendStatus: user triggered refresh", { notify, isDesktopRuntime });
  if (!isDesktopRuntime) {
    console.warn("[SettingsPage] refreshBackendStatus: not in Tauri desktop runtime");
    setSettingsFeedback("warning", "设置状态仅在 Tauri 桌面端可读取。", notify);
    return;
  }
  console.debug("[SettingsPage] refreshBackendStatus: fetching backend status");
  const t0 = Date.now();
  settingsLoading.value = true;
  try {
    const status = await fetchSharedBackendStatus();
    const duration = Date.now() - t0;
    console.info("[SettingsPage] refreshBackendStatus: success", { ready: status.ready, message: status.message, durationMs: duration });
    applyBackendStatus(status);
    void loadModelCatalog();
    setSettingsFeedback(status.ready ? "success" : "warning", status.message, notify);
  } catch (error) {
    const duration = Date.now() - t0;
    console.error("[SettingsPage] refreshBackendStatus: failed", { error: error instanceof Error ? error.message : String(error), durationMs: duration });
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法读取后端状态。", notify);
  } finally {
    settingsLoading.value = false;
  }
}
async function saveSettings() {
  console.info("[SettingsPage] saveSettings: user triggered save", { targetLanguage: targetLanguage.value, fontPath: modelFontPath.value });
  const t0 = Date.now();
  const nextLanguage = targetLanguage.value.trim();
  console.debug("[SettingsPage] saveSettings: params", { nextLanguage, promptLen: promptTemplate.value.length, fontPath: modelFontPath.value });
  if (!isSupportedTargetLanguage(nextLanguage)) {
    console.warn("[SettingsPage] saveSettings: validation failed - unsupported language", { nextLanguage });
    setSettingsFeedback("error", "目标语言不在 Hy-MT2 支持列表内，请从下拉选择（38 种）。");
    return;
  }


  const trimmedPromptTemplate = promptTemplate.value.trim();
  if (Array.from(trimmedPromptTemplate).length > 8192) {
    console.warn("[SettingsPage] saveSettings: validation failed - prompt too long", { promptLen: Array.from(trimmedPromptTemplate).length });
    setSettingsFeedback("error", "提示词模板最多 8192 个字符。");
    return;
  }

  if (!isDesktopRuntime) {
    console.debug("[SettingsPage] saveSettings: browser preview path, saving locally");
    targetLanguage.value = nextLanguage;
    const persistError = savePersistedTargetLanguage();
    if (persistError) {
      console.error("[SettingsPage] saveSettings: persist failed (browser)", { error: persistError });
      setSettingsFeedback("error", persistError);
      return;
    }
    console.info("[SettingsPage] saveSettings: saved locally (browser preview)", { nextLanguage, durationMs: Date.now() - t0 });
    setSettingsFeedback("success", "目标语言已保存（浏览器预览）。");
    return;
  }

  let status = backendStatus.value;
  if (!status) {
    console.info("[SettingsPage] saveSettings: backend status not cached, fetching", { isDesktopRuntime });
    try {
      status = await fetchSharedBackendStatus();
      applyBackendStatus(status);
      console.info("[SettingsPage] saveSettings: fetched backend status for save", { fontPath: status.fontPath, device: status.device });
    } catch (error) {
      console.error("[SettingsPage] saveSettings: fetch status failed", { error: error instanceof Error ? error.message : String(error) });
      setSettingsFeedback("error", error instanceof Error ? error.message : "后端状态未就绪，请先刷新。");
      return;
    }
  }

  const settings: BackendSettingsUpdate = {
    detectorModelDir: status.detectorModelDir,
    recognizerModelDir: status.recognizerModelDir,
    hyModel: status.hyModel,
    fontPath: modelFontPath.value?.trim() || null,
    targetLanguage: nextLanguage,
    device: status.device === "cpu" ? "cpu" : "cuda",
    regionParallelism: status.regionParallelism,
    translationBatchSize: status.translationBatchSize,
    idleUnloadSeconds: status.idleUnloadSeconds,
    generation: status.generation,
    memory: status.memory,
    prompt: {
      template: trimmedPromptTemplate,
    },
  };

  console.debug("[SettingsPage] saveSettings: sending to backend", { settings });
  settingsLoading.value = true;
  try {
    const nextStatus = await updateBackendSettings(settings);
    const duration = Date.now() - t0;
    console.info("[SettingsPage] saveSettings: backend update success", { nextLanguage, fontPath: settings.fontPath, durationMs: duration, ready: nextStatus.ready });
    applyBackendStatus(nextStatus);
    targetLanguage.value = nextLanguage;
    const persistError = savePersistedTargetLanguage();
    if (persistError) {
      console.warn("[SettingsPage] saveSettings: backend saved but local persist failed", { error: persistError });
      setSettingsFeedback("warning", `后端已保存，但本地语言持久化失败：${persistError}`);
      return;
    }
    setSettingsFeedback("success", "翻译偏好已保存，下一次翻译会使用新的提示词与目标语言。");
  } catch (error) {
    const duration = Date.now() - t0;
    console.error("[SettingsPage] saveSettings: backend update failed", { error: error instanceof Error ? error.message : String(error), durationMs: duration });
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存设置。");
  } finally {
    settingsLoading.value = false;
  }
}
onMounted(() => {
  console.info("[SettingsPage] onMounted: initializing settings page", { hasBackendStatus: !!backendStatus.value, isDesktopRuntime });
  if (backendStatus.value) {
    console.debug("[SettingsPage] onMounted: applying existing backend status");
    applyBackendStatus(backendStatus.value);
  }
  void loadModelCatalog();
  void refreshBackendStatus(false);
  console.debug("[SettingsPage] onMounted: triggered loadModelCatalog + refreshBackendStatus");
});
</script>

<template>
  <section class="settings-page" aria-labelledby="settings-title">
    <div class="settings-grid">
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

      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">Fonts</p>
            <h2>OCR标注字体</h2>
          </div>
          <n-button secondary size="small" @click="openFontDialog">导入字体…</n-button>
        </div>
        <p class="settings-card-copy" style="margin-bottom: 12px">
          用于图像翻译结果的文字标注字体；系统自动匹配为默认，导入后可在下拉中选择并保存生效。
        </p>
        <dl class="settings-path-list" style="margin-top: 0">
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
              <n-popconfirm
                :show-icon="false"
                positive-text="删除"
                negative-text="取消"
                @positive-click="deleteSelectedFont"
              >
                <template #trigger>
                  <n-button
                    secondary
                    size="small"
                    :disabled="!modelFontPath || dialogSaving"
                    :loading="dialogSaving"
                    aria-label="删除选中字体"
                  >
                    删除
                  </n-button>
                </template>
                确定删除「{{ modelFontPath ? pathBaseName(modelFontPath) : '' }}」？删除后需保存设置生效。
              </n-popconfirm>
              <n-tooltip v-if="modelFontPath" trigger="hover">
                <template #trigger>
                  <span class="settings-model-help" :title="modelFontPath" style="max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; vertical-align: bottom">{{ modelFontPath }}</span>
                </template>
                {{ modelFontPath }}
              </n-tooltip>
              <span v-else class="settings-model-help">系统自动匹配</span>
            </dd>
          </div>
        </dl>
      </n-card>

      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">翻译偏好</p>
            <h2>目标语言与提示词</h2>
          </div>
        </div>
        <p class="settings-card-copy">
          目标语言决定 Hy-MT2 的 <code>target_lang</code>（需使用<strong>完整语言名</strong>，English prompt 用英文名，中文 prompt 用中文名）。
          本地翻译默认使用官方 Default 模板：<code>Translate the following text into {target_language}. Note that you should only output the translated result without any additional explanation:\n\n{source_text}</code>。
          单模板支持 <code>{source_text}</code> / <code>{target_lang}</code> / <code>{target_language}</code> / <code>{format_type}</code> 占位符。
        </p>
        <div class="settings-field-grid">
          <label class="settings-field">
            <span>目标语言</span>
            <TargetLanguageSelect v-model="targetLanguage" />
          </label>
          <span class="settings-help" style="grid-column: 1 / -1">仅支持 Hy-MT2 官方 38 语言，已自动归一化英文全称；不支持的语言将校验失败。</span>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>翻译提示词模板</span>
            <n-input
              v-model:value="promptTemplate"
              type="textarea"
              maxlength="8192"
              placeholder="可选：单模板，支持 {source_text} / {target_lang} / {target_language} / {format_type} 占位符。留空则使用官方 Default：Translate the following text into {target}. Note that you should only output the translated result without any additional explanation: {text}"
              :autosize="{ minRows: 3, maxRows: 6 }"
              aria-label="Hy prompt template"
            />
          </label>
          <span class="settings-help" style="grid-column: 1 / -1">留空使用官方 Default 模板；若模板包含 {source_text} 则视为完整模板并直接替换占位符后使用（覆盖 Default）；否则视为附加约束拼于 Default 之前。支持批量 {format_type}。</span>
        </div>
      </n-card>
      <OpenAiCompatCard />
      <OpenAiRequestHistory />
      <div class="settings-card-actions settings-page-actions settings-card-wide">
        <n-alert v-if="settingsMessage" class="settings-actions-feedback" :type="settingsMessageType" :show-icon="false">
          {{ settingsMessage }}
        </n-alert>
        <n-button secondary :loading="settingsLoading" @click="refreshBackendStatus()">刷新状态</n-button>
        <n-button type="primary" :loading="settingsLoading" @click="saveSettings">保存设置</n-button>
      </div>
    </div>
    <n-modal
      v-model:show="fontDialogOpen"
      preset="card"
      title="配置标注字体路径"
      :mask-closable="true"
      style="width: 520px; max-width: calc(100vw - 48px)"
      class="model-path-dialog"
    >
      <div class="model-dialog-fields">
        <n-input v-model:value="dialogName" maxlength="64" placeholder="字体名称（用于下拉框显示）" />
        <div class="model-dialog-path-row">
          <span class="model-dialog-path-value" :title="dialogFontPath">{{ dialogFontPath || "未选择" }}</span>
          <n-button secondary size="small" @click="pickDialogFontPath">选择字体文件</n-button>
        </div>
      </div>
      <template #footer>
        <div class="model-dialog-footer">
          <n-button secondary size="small" :disabled="dialogSaving" @click="closeFontDialog">取消</n-button>
          <n-button type="primary" size="small" :loading="dialogSaving" @click="saveFontDialog">注册并选择</n-button>
        </div>
      </template>
    </n-modal>
  </section>
</template>
<style scoped src="../styles/settings-page.css"></style>
