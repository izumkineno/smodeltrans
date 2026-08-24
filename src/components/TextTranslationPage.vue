<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, ref } from "vue";
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
  NSwitch,
  NTag,
  useMessage,
} from "naive-ui";
import { copyTranslationText, saveTranslationText } from "../services/file-adapter";
import {
  createTranslationRequestId,
  isTranslationCancellation,
  textTranslationProvider,
} from "../services/translation-provider";
import type { TranslationProgress } from "../services/translation-provider";
import { targetLanguage } from "../services/workspace-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";
import {
  isQuickTranslationShortcutModifierCode,
  loadPersistedQuickTranslationSettings,
  quickTranslationSettings,
  quickTranslationShortcutFromKeyboardEvent,
  quickTranslationShortcutLabel,
  saveQuickTranslationSettings,
} from "../services/quick-translation-settings";

type TextWorkflowState = "idle" | "processing" | "result" | "cancelled" | "error";
type TagType = "default" | "success" | "warning" | "error" | "info";

loadPersistedQuickTranslationSettings();

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const sourceText = ref("");
const translatedText = ref<string | null>(null);
const providerLabel = ref("");
const durationMs = ref<number | null>(null);
const actionFeedback = ref("");
const errorMessage = ref("");
const statusMessage = ref("输入文本后开始翻译。");
const progress = ref(0);
const workflowState = ref<TextWorkflowState>("idle");
const pageRoot = ref<HTMLElement | null>(null);
const activeController = ref<AbortController | null>(null);
const quickTranslationEnabled = ref(quickTranslationSettings.value.enabled);
const quickTranslationShortcut = ref(quickTranslationSettings.value.shortcut);
const quickTranslationSettingsSaving = ref(false);
const isCapturingQuickTranslationShortcut = ref(false);
const quickTranslationShortcutCaptureHint = ref("");
let progressUnlisten: UnlistenFn | undefined;

const toast = useMessage();

function notify(type: WorkspaceToastType, message: string) {
  showWorkspaceToast(toast, type, message);
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
    case "processing":
      return "翻译中";
    case "result":
      return "结果就绪";
    case "cancelled":
      return "已取消";
    case "error":
      return "需要处理";
    default:
      return "等待输入";
  }
});

const workflowTagType = computed<TagType>(() => {
  switch (workflowState.value) {
    case "processing":
      return "warning";
    case "result":
      return "success";
    case "cancelled":
      return "default";
    case "error":
      return "error";
    default:
      return "info";
  }
});

const durationLabel = computed(() => {
  if (durationMs.value === null || !Number.isFinite(durationMs.value) || durationMs.value < 0) {
    return "";
  }
  return durationMs.value < 1000
    ? `${Math.max(1, Math.round(durationMs.value))} ms`
    : `${(durationMs.value / 1000).toFixed(2)} 秒`;
});

const canStart = computed(
  () =>
    isDesktopRuntime &&
    sourceText.value.trim().length > 0 &&
    targetLanguage.value.trim().length > 0 &&
    workflowState.value !== "processing",
);

function clearProgressListener() {
  progressUnlisten?.();
  progressUnlisten = undefined;
}

function resetResult() {
  translatedText.value = null;
  providerLabel.value = "";
  durationMs.value = null;
  actionFeedback.value = "";
  errorMessage.value = "";
}

function validateRequest(): { text: string; language: string } | null {
  const text = sourceText.value.trim();
  if (!text) {
    errorMessage.value = "请输入需要翻译的文本。";
    return null;
  }
  if (new TextEncoder().encode(text).length > 8 * 1024 * 1024) {
    errorMessage.value = "文本不能超过 8 MiB。";
    return null;
  }

  const language = targetLanguage.value.trim();
  const languageLength = Array.from(language).length;
  if (languageLength < 1 || languageLength > 64) {
    errorMessage.value = "目标语言长度必须为 1 到 64 个字符。";
    return null;
  }

  return { text, language };
}

async function startTranslation() {
  const request = validateRequest();
  if (!request) {
    setErrorState(errorMessage.value || "请检查文本翻译请求。");
    return;
  }
  if (!isDesktopRuntime) {
    setErrorState("文本翻译后端只在 Tauri 桌面端可用。");
    return;
  }
  if (workflowState.value === "processing") {
    return;
  }

  const controller = new AbortController();
  activeController.value = controller;
  resetResult();
  workflowState.value = "processing";
  progress.value = 0;
  statusMessage.value = "正在准备文本翻译。";
  const requestId = createTranslationRequestId();
  clearProgressListener();

  try {
    progressUnlisten = await listen<TranslationProgress>("translation-progress", (event) => {
      if (
        event.payload.requestId !== requestId ||
        activeController.value !== controller ||
        workflowState.value !== "processing"
      ) {
        return;
      }
      progress.value = Math.min(100, Math.max(0, Math.round(event.payload.progress)));
      statusMessage.value = event.payload.stage;
    });

    const result = await textTranslationProvider.translate(
      {
        text: request.text,
        targetLanguage: request.language,
        requestId,
      },
      controller.signal,
    );

    if (controller.signal.aborted || activeController.value !== controller) {
      return;
    }

    translatedText.value = result.text;
    providerLabel.value = result.providerLabel;
    durationMs.value = result.durationMs;
    progress.value = 100;
    workflowState.value = "result";
    statusMessage.value = "文本翻译结果已准备好。";
    notify("success", "文本翻译完成。");
  } catch (error) {
    if (controller.signal.aborted || isTranslationCancellation(error)) {
      if (activeController.value === controller) {
        workflowState.value = "cancelled";
        statusMessage.value = "翻译已取消，可以重新开始。";
      }
      return;
    }
    if (activeController.value !== controller) {
      return;
    }
    setErrorState(error instanceof Error ? error.message : "文本翻译未完成，请检查模型配置后重试。");
  } finally {
    if (activeController.value === controller) {
      activeController.value = null;
      clearProgressListener();
      if (workflowState.value !== "result") {
        progress.value = 0;
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
  progress.value = 0;
  workflowState.value = "cancelled";
  statusMessage.value = "翻译已取消，可以重新开始。";
  notify("warning", "翻译已取消。");
}

function clearText() {
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  sourceText.value = "";
  resetResult();
  progress.value = 0;
  workflowState.value = "idle";
  statusMessage.value = "输入文本后开始翻译。";
  notify("info", "输入和翻译结果已清空。");
}

async function copyResult() {
  if (!translatedText.value) {
    return;
  }
  try {
    await copyTranslationText(translatedText.value);
    setActionFeedback("success", "译文已复制到剪贴板。");
  } catch {
    setActionFeedback("error", "剪贴板不可用，请手动选择文本进行复制。");
  }
}

function saveResult() {
  if (!translatedText.value) {
    return;
  }
  try {
    saveTranslationText(translatedText.value, "text-translation.txt");
    setActionFeedback("success", "译文已保存为文本文件。");
  } catch {
    setActionFeedback("error", "译文无法保存，请重试。");
  }
}
async function saveQuickTranslationConfiguration(): Promise<void> {
  if (isCapturingQuickTranslationShortcut.value) {
    return;
  }
  quickTranslationSettingsSaving.value = true;
  try {
    const saved = await saveQuickTranslationSettings({
      enabled: quickTranslationEnabled.value,
      shortcut: quickTranslationShortcut.value,
    });
    quickTranslationEnabled.value = saved.enabled;
    quickTranslationShortcut.value = saved.shortcut;
    notify("success", "快捷翻译设置已保存并立即生效。");
  } catch (error) {
    quickTranslationEnabled.value = quickTranslationSettings.value.enabled;
    quickTranslationShortcut.value = quickTranslationSettings.value.shortcut;
    notify(
      "error",
      error instanceof Error ? error.message : "快捷翻译设置保存失败，请重试。",
    );
  } finally {
    quickTranslationSettingsSaving.value = false;
  }
}

function stopQuickTranslationShortcutCapture(): void {
  if (typeof window !== "undefined") {
    window.removeEventListener("keydown", handleQuickTranslationShortcutCapture, true);
    window.removeEventListener("blur", handleQuickTranslationShortcutCaptureBlur, true);
  }
  isCapturingQuickTranslationShortcut.value = false;
}

function handleQuickTranslationShortcutCaptureBlur(): void {
  stopQuickTranslationShortcutCapture();
  quickTranslationShortcutCaptureHint.value = "窗口失去焦点，已取消快捷键录入。";
}

function handleQuickTranslationEnabledChange(enabled: boolean): void {
  if (!enabled && isCapturingQuickTranslationShortcut.value) {
    stopQuickTranslationShortcutCapture();
    quickTranslationShortcutCaptureHint.value = "已取消快捷键录入。";
  }
}

function handleQuickTranslationShortcutCapture(event: KeyboardEvent): void {
  event.preventDefault();
  event.stopPropagation();
  if (event.isComposing || event.repeat) {
    return;
  }
  if (event.code === "Escape") {
    stopQuickTranslationShortcutCapture();
    quickTranslationShortcutCaptureHint.value = "已取消快捷键录入。";
    return;
  }
  if (isQuickTranslationShortcutModifierCode(event.code)) {
    return;
  }

  const shortcut = quickTranslationShortcutFromKeyboardEvent(event);
  if (!shortcut) {
    quickTranslationShortcutCaptureHint.value =
      event.ctrlKey || event.altKey || event.shiftKey || event.metaKey
        ? "该按键暂不支持，请改用字母、数字、功能键、方向键或其他标准键。"
        : "请先按住 Ctrl、Alt、Shift 或 Win，再按最后一个按键。";
    return;
  }
  quickTranslationShortcut.value = shortcut;
  quickTranslationShortcutCaptureHint.value = `已录入 ${quickTranslationShortcutLabel(shortcut)}，点击保存后生效。`;
  stopQuickTranslationShortcutCapture();
}

function toggleQuickTranslationShortcutCapture(): void {
  if (isCapturingQuickTranslationShortcut.value) {
    stopQuickTranslationShortcutCapture();
    quickTranslationShortcutCaptureHint.value = "已取消快捷键录入。";
    return;
  }
  if (!isDesktopRuntime || typeof window === "undefined") {
    return;
  }
  quickTranslationShortcutCaptureHint.value =
    "请同时按住 Ctrl、Alt、Shift 或 Win，再按最后一个按键；Esc 取消。";
  isCapturingQuickTranslationShortcut.value = true;
  window.addEventListener("keydown", handleQuickTranslationShortcutCapture, true);
  window.addEventListener("blur", handleQuickTranslationShortcutCaptureBlur, true);
}


function handleSourceInput() {
  if (workflowState.value === "result" || workflowState.value === "error") {
    workflowState.value = sourceText.value.trim() ? "idle" : "idle";
    resetResult();
    statusMessage.value = "输入已更新，可以重新翻译。";
  }
}

function handleTranslationShortcut(event: KeyboardEvent) {
  if (isCapturingQuickTranslationShortcut.value) {
    return;
  }
  const isShortcut =
    event.key === "Enter" &&
    (event.ctrlKey || event.metaKey) &&
    !event.altKey &&
    !event.shiftKey;
  if (
    !isShortcut ||
    event.defaultPrevented ||
    event.repeat ||
    !pageRoot.value ||
    !document.body.contains(pageRoot.value) ||
    event.isComposing ||
    !canStart.value
  ) {
    return;
  }
  event.preventDefault();
  void startTranslation();
}

function bindTranslationShortcut() {
  window.addEventListener("keydown", handleTranslationShortcut, true);
}

function unbindTranslationShortcut() {
  window.removeEventListener("keydown", handleTranslationShortcut, true);
}

onActivated(bindTranslationShortcut);
onDeactivated(() => {
  stopQuickTranslationShortcutCapture();
  unbindTranslationShortcut();
});

onBeforeUnmount(() => {
  stopQuickTranslationShortcutCapture();
  unbindTranslationShortcut();
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
});
</script>

<template>
  <section ref="pageRoot" class="text-translation-page" aria-label="文本翻译工作区">
    <p class="one-shot-context">适合已有文字的直接翻译，不需要先识别图片。</p>
    <div class="workflow-grid text-page-grid">
      <n-card class="panel text-page-card text-input-card" :bordered="false">
        <div class="panel-heading text-card-heading">
          <div class="panel-title text-card-title">
            <span class="section-number text-section-number">01</span>
            <div>
              <p class="panel-kicker text-panel-kicker">输入 / 文本</p>
              <h2>输入待翻译内容</h2>
              <p class="panel-copy text-panel-copy">直接调用 Hy-MT2 模型翻译文本，不经过 OCR。</p>
            </div>
          </div>
          <n-tag round size="small" type="info">Hy-MT2</n-tag>
        </div>

        <n-input
          v-model:value="sourceText"
          class="text-source-input"
          type="textarea"
          placeholder="在这里粘贴或输入文本……"
          :autosize="{ minRows: 14, maxRows: 24 }"
          aria-label="待翻译文本"
          @input="handleSourceInput"
        />

        <div class="text-options">
          <label class="text-option-field">
            <span>目标语言</span>
            <n-input
              v-model:value="targetLanguage"
              maxlength="64"
              placeholder="例如：Chinese"
              aria-label="文本翻译目标语言"
            />
          </label>
          <p class="text-option-help">支持自然语言名称，例如 Chinese、English、Japanese。</p>
        </div>

        <n-alert v-if="!isDesktopRuntime" class="text-runtime-alert" type="info" :show-icon="false">
          浏览器预览仅展示界面；文本翻译需要在 Tauri 桌面端运行。
        </n-alert>
        <n-alert v-if="workflowState === 'error'" class="text-inline-alert" type="error" title="翻译未完成" :show-icon="true">
          {{ errorMessage }}
        </n-alert>

        <div class="text-card-actions">
          <span class="text-shortcut-hint" aria-label="快捷键：Control 或 Command 加 Enter">
            <span>快捷键</span>
            <kbd>Ctrl</kbd>
            <span>/</span>
            <kbd>⌘</kbd>
            <span>+</span>
            <kbd>Enter</kbd>
          </span>
          <n-button secondary @click="clearText">清空</n-button>
          <n-button
            v-if="workflowState !== 'processing'"
            type="primary"
            :disabled="!canStart"
            aria-keyshortcuts="Control+Enter Meta+Enter"
            @click="startTranslation"
          >
            {{ workflowState === 'cancelled' || workflowState === 'error' ? '再次翻译' : '开始翻译' }}
          </n-button>
          <n-button v-else tertiary type="warning" @click="cancelTranslation">取消翻译</n-button>
        </div>
      </n-card>

      <n-card class="panel text-page-card text-output-card" :bordered="false">
        <div class="panel-heading text-card-heading">
          <div class="panel-title text-card-title">
            <span class="section-number text-section-number">02</span>
            <div>
              <p class="panel-kicker text-panel-kicker">输出 / 译文</p>
              <h2>翻译结果</h2>
              <p class="panel-copy text-panel-copy">复制译文，或保存为文本文件。</p>
            </div>
          </div>
          <n-tag :type="workflowTagType" round size="small">{{ workflowStatus }}</n-tag>
        </div>

        <div v-if="workflowState === 'processing'" class="text-processing-state" aria-busy="true">
          <div class="text-processing-icon" aria-hidden="true"><n-spin size="medium" /></div>
          <p class="panel-kicker text-panel-kicker">模型生成中</p>
          <p class="text-status-copy">{{ statusMessage }}</p>
          <n-progress
            type="line"
            :percentage="progress"
            :show-indicator="false"
            processing
            aria-label="文本翻译进度"
            :aria-valuetext="`${statusMessage}，${progress}% 完成`"
          />
          <div class="text-progress-meta"><span>{{ statusMessage }}</span><strong>{{ progress }}%</strong></div>
        </div>

        <div v-else-if="workflowState === 'result' && translatedText" class="text-result-state">
          <div class="text-result-meta">
            <span>{{ providerLabel || "Hy-MT2" }}</span>
            <span v-if="durationLabel">用时 {{ durationLabel }}</span>
          </div>
          <n-input
            class="text-result-input"
            type="textarea"
            :value="translatedText"
            readonly
            :autosize="{ minRows: 14, maxRows: 24 }"
            aria-label="翻译结果文本"
          />
          <div class="text-result-actions">
            <n-space :size="10" wrap>
              <n-button secondary @click="copyResult">复制译文</n-button>
              <n-button type="primary" @click="saveResult">保存 .txt</n-button>
            </n-space>
            <p v-if="actionFeedback" class="text-action-feedback" aria-live="polite">{{ actionFeedback }}</p>
          </div>
        </div>

        <div v-else-if="workflowState === 'cancelled'" class="text-output-message">
          <n-alert class="text-inline-alert" type="warning" title="翻译已取消" :show-icon="true">
            输入内容仍然保留，可以再次开始翻译。
          </n-alert>
          <n-button type="primary" :disabled="!canStart" @click="startTranslation">再次翻译</n-button>
        </div>

        <div v-else-if="workflowState === 'error'" class="text-output-message">
          <n-alert class="text-inline-alert" type="error" title="需要处理" :show-icon="true">
            {{ errorMessage }}
          </n-alert>
          <n-button type="primary" :disabled="!canStart" @click="startTranslation">再次翻译</n-button>
        </div>

        <div v-else class="text-empty-output">
          <n-empty size="small" description="翻译结果将在此处显示。" />
        </div>
      </n-card>
    </div>

    <n-card class="panel text-quick-settings-card" :bordered="false">
      <div class="panel-heading text-card-heading text-quick-settings-heading">
        <div class="panel-title text-card-title">
          <span class="section-number text-section-number">03</span>
          <div>
            <p class="panel-kicker text-panel-kicker">快捷翻译 / Windows UI Automation</p>
            <h2>选中文字快捷翻译</h2>
            <p class="panel-copy text-panel-copy">
              在其他应用中选中文字后按全局快捷键，使用 UI Automation 读取选区，并复用实时字幕框显示译文。
            </p>
          </div>
        </div>
        <n-switch
          v-model:value="quickTranslationEnabled"
          aria-label="启用选中文字快捷翻译"
          @update:value="handleQuickTranslationEnabledChange"
        />
      </div>

      <div class="text-quick-settings-grid">
        <div class="text-option-field">
          <span>全局快捷键</span>
          <div class="text-quick-shortcut-control">
            <div
              class="text-quick-shortcut-value"
              role="status"
              :title="quickTranslationShortcut"
              aria-label="当前快捷翻译全局快捷键"
            >
              {{ quickTranslationShortcutLabel(quickTranslationShortcut) }}
            </div>
            <n-button
              secondary
              :disabled="!quickTranslationEnabled || quickTranslationSettingsSaving"
              @click="toggleQuickTranslationShortcutCapture"
            >
              {{ isCapturingQuickTranslationShortcut ? "取消录入" : "录入快捷键" }}
            </n-button>
          </div>
          <p v-if="quickTranslationShortcutCaptureHint" class="text-quick-shortcut-hint" role="status">
            {{ quickTranslationShortcutCaptureHint }}
          </p>
        </div>
      </div>

      <div class="text-quick-settings-actions">
        <p class="text-option-help">
          快捷翻译窗口沿用实时字幕框样式，并根据原文和译文内容自动调整尺寸。
        </p>
        <n-button
          type="primary"
          :loading="quickTranslationSettingsSaving"
          :disabled="!isDesktopRuntime"
          @click="saveQuickTranslationConfiguration"
        >
          保存快捷翻译设置
        </n-button>
      </div>
    </n-card>

    <footer class="text-page-footer">
      <span>Hy-MT2 直接翻译</span>
      <span class="text-footer-separator" aria-hidden="true"></span>
      <span>本地 Candle 推理</span>
      <span class="text-footer-spacer"></span>
      <span>{{ statusMessage }}</span>
    </footer>
  </section>
</template>

<style scoped src="../styles/text-translation-page.css"></style>
