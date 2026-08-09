<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from "vue";
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

type TextWorkflowState = "idle" | "processing" | "result" | "cancelled" | "error";
type TagType = "default" | "success" | "warning" | "error" | "info";


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
const activeController = ref<AbortController | null>(null);
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

function handleSourceInput() {
  if (workflowState.value === "result" || workflowState.value === "error") {
    workflowState.value = sourceText.value.trim() ? "idle" : "idle";
    resetResult();
    statusMessage.value = "输入已更新，可以重新翻译。";
  }
}

onBeforeUnmount(() => {
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
});
</script>

<template>
  <section class="text-translation-page" aria-label="文本翻译工作区">
    <div class="text-page-grid">
      <n-card class="text-page-card text-input-card" :bordered="false">
        <div class="text-card-heading">
          <div class="text-card-title">
            <span class="text-section-number">01</span>
            <div>
              <p class="text-panel-kicker">输入 / 文本</p>
              <h2>输入待翻译内容</h2>
              <p class="text-panel-copy">直接调用 Hy-MT2 模型翻译文本，不经过 OCR。</p>
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
          <n-button secondary @click="clearText">清空</n-button>
          <n-button
            v-if="workflowState !== 'processing'"
            type="primary"
            :disabled="!canStart"
            @click="startTranslation"
          >
            {{ workflowState === 'cancelled' || workflowState === 'error' ? '再次翻译' : '开始翻译' }}
          </n-button>
          <n-button v-else tertiary type="warning" @click="cancelTranslation">取消翻译</n-button>
        </div>
      </n-card>

      <n-card class="text-page-card text-output-card" :bordered="false">
        <div class="text-card-heading">
          <div class="text-card-title">
            <span class="text-section-number">02</span>
            <div>
              <p class="text-panel-kicker">输出 / 译文</p>
              <h2>翻译结果</h2>
              <p class="text-panel-copy">复制译文，或保存为文本文件。</p>
            </div>
          </div>
          <n-tag :type="workflowTagType" round size="small">{{ workflowStatus }}</n-tag>
        </div>

        <div v-if="workflowState === 'processing'" class="text-processing-state" aria-busy="true">
          <div class="text-processing-icon" aria-hidden="true"><n-spin size="medium" /></div>
          <p class="text-panel-kicker">模型生成中</p>
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

    <footer class="text-page-footer">
      <span>Hy-MT2 直接翻译</span>
      <span class="text-footer-separator" aria-hidden="true"></span>
      <span>本地 Candle 推理</span>
      <span class="text-footer-spacer"></span>
      <span>{{ statusMessage }}</span>
    </footer>
  </section>
</template>

<style scoped>
.text-translation-page {
  width: 100%;
  margin-top: 24px;
}

.text-page-grid {
  display: grid;
  width: 100%;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.text-page-card {
  min-height: max(520px, calc(100dvh - 360px));
}

.text-page-card :deep(.n-card__content) {
  display: flex;
  min-height: inherit;
  flex-direction: column;
  padding: 20px;
}

.text-card-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.text-card-title {
  display: flex;
  min-width: 0;
  gap: 12px;
}

.text-section-number {
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

.text-panel-kicker {
  margin: 0 0 4px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}

.text-card-heading h2 {
  margin: 0;
  color: var(--text);
  font-size: 16px;
  font-weight: 650;
  line-height: 1.35;
}

.text-panel-copy {
  max-width: 340px;
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.text-source-input,
.text-result-input {
  width: 100%;
}

.text-source-input {
  flex: 1;
}

.text-source-input :deep(textarea),
.text-result-input :deep(textarea) {
  min-height: 250px !important;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.6;
}

.text-options {
  display: grid;
  gap: 8px;
  margin-top: 16px;
}

.text-option-field {
  display: grid;
  max-width: 360px;
  gap: 7px;
  color: var(--text-soft);
  font-size: 13px;
  font-weight: 600;
}

.text-option-help {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.text-runtime-alert,
.text-inline-alert {
  margin-top: 14px;
}

.text-card-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: auto;
  padding-top: 18px;
}

.text-processing-state,
.text-output-message,
.text-empty-output {
  display: flex;
  min-height: 360px;
  flex: 1;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
}

.text-processing-state {
  align-items: stretch;
}

.text-processing-icon {
  display: grid;
  place-items: center;
  min-height: 56px;
}

.text-processing-state .text-panel-kicker,
.text-processing-state .text-status-copy {
  text-align: center;
}

.text-status-copy {
  margin: 0;
  color: var(--text-soft);
  font-size: 13px;
}

.text-progress-meta,
.text-result-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  color: var(--text-muted);
  font-size: 12px;
}

.text-progress-meta strong {
  color: var(--green);
  font-variant-numeric: tabular-nums;
}

.text-result-state {
  display: flex;
  min-height: 360px;
  flex: 1;
  flex-direction: column;
}

.text-result-meta {
  margin-bottom: 10px;
  padding: 0 1px 8px;
  border-bottom: 1px solid var(--divider);
}

.text-result-meta span:first-child {
  color: var(--text-soft);
  font-weight: 600;
}

.text-result-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 14px;
}

.text-action-feedback {
  margin: 0;
  color: var(--green);
  font-size: 12px;
  line-height: 1.5;
  text-align: right;
}

.text-output-message {
  align-items: flex-start;
}

.text-output-message .text-inline-alert {
  width: 100%;
}

.text-page-footer {
  display: flex;
  width: 100%;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.text-footer-separator {
  width: 4px;
  height: 4px;
  flex: 0 0 4px;
  border-radius: 50%;
  background: #c0c4cc;
}

.text-footer-spacer {
  flex: 1;
}

@media (max-width: 900px) {
  .text-page-grid {
    grid-template-columns: 1fr;
  }

  .text-page-card {
    min-height: 0;
  }
}

@media (max-width: 560px) {
  .text-card-heading {
    flex-direction: column;
    gap: 10px;
  }

  .text-result-actions,
  .text-page-footer {
    align-items: flex-start;
    flex-direction: column;
  }

  .text-action-feedback {
    text-align: left;
  }

  .text-footer-spacer {
    display: none;
  }
}
</style>
