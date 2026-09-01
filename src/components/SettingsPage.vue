<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { open as openNativeDialog } from "@tauri-apps/plugin-dialog";
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NModal,
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

function selectFontModel(value: string): void {
  modelFontPath.value = value === SYSTEM_FONT_VALUE ? null : value;
  setSettingsFeedback("info", "已选择标注字体，点击“保存设置”生效。");
}

function openFontDialog(): void {
  dialogName.value = "";
  dialogFontPath.value = "";
  fontDialogOpen.value = true;
}

function closeFontDialog(): void {
  fontDialogOpen.value = false;
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

async function saveFontDialog(): Promise<void> {
  if (dialogSaving.value) {
    return;
  }
  if (!catalogLoaded.value) {
    await loadModelCatalog();
  }
  const entryName = dialogName.value.trim();
  if (!entryName) {
    setSettingsFeedback("error", "请输入字体名称。");
    return;
  }
  const path = dialogFontPath.value.trim();
  if (!path) {
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
  dialogSaving.value = true;
  try {
    await saveModelCatalog(next);
    await loadModelCatalog();
    selectFontModel(dialogFontPath.value);
    closeFontDialog();
    setSettingsFeedback("success", `已配置「${entryName}」并选中，点击“保存设置”生效。`);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存模型条目。");
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


function applyBackendStatus(status: BackendStatus) {
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
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法读取后端状态。", notify);
  } finally {
    settingsLoading.value = false;
  }
}
async function saveSettings() {
  const nextLanguage = targetLanguage.value.trim();
  if (!isSupportedTargetLanguage(nextLanguage)) {
    setSettingsFeedback("error", "目标语言不在 Hy-MT2 支持列表内，请从下拉选择（38 种）。");
    return;
  }


  const trimmedPromptTemplate = promptTemplate.value.trim();
  if (Array.from(trimmedPromptTemplate).length > 8192) {
    setSettingsFeedback("error", "提示词模板最多 8192 个字符。");
    return;
  }

  if (!isDesktopRuntime) {
    targetLanguage.value = nextLanguage;
    const persistError = savePersistedTargetLanguage();
    if (persistError) {
      setSettingsFeedback("error", persistError);
      return;
    }
    setSettingsFeedback("success", "目标语言已保存（浏览器预览）。");
    return;
  }

  const status = backendStatus.value;
  if (!status) {
    setSettingsFeedback("error", "后端状态未就绪，请先刷新。");
    return;
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

  settingsLoading.value = true;
  try {
    const nextStatus = await updateBackendSettings(settings);
    applyBackendStatus(nextStatus);
    targetLanguage.value = nextLanguage;
    const persistError = savePersistedTargetLanguage();
    if (persistError) {
      setSettingsFeedback("warning", `后端已保存，但本地语言持久化失败：${persistError}`);
      return;
    }
    setSettingsFeedback("success", "翻译偏好已保存，下一次翻译会使用新的提示词与目标语言。");
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存设置。");
  } finally {
    settingsLoading.value = false;
  }
}
onMounted(() => {
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
            <h2>标注字体</h2>
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
              <n-tooltip v-if="modelFontPath" trigger="hover">
                <template #trigger>
                  <span class="settings-model-help" :title="modelFontPath" style="max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: inline-block; vertical-align: bottom">{{ modelFontPath }}</span>
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
