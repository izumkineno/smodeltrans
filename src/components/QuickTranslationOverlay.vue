<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import { listen } from "@tauri-apps/api/event";
import { NButton } from "naive-ui";
import { getCurrentWindow, LogicalSize } from "@tauri-apps/api/window";
import type { UnlistenFn } from "@tauri-apps/api/event";
import { copyTranslationText } from "../services/file-adapter";
import {
  createTranslationRequestId,
  isTranslationCancellation,
  textTranslationProvider,
} from "../services/translation-provider";
import type { TranslationProgress } from "../services/translation-provider";
import type {
  LiveSessionState,
  LiveSubtitle,
  SubtitleProgress,
} from "../services/live-translation-provider";
import {
  fetchSharedBackendStatus,
  loadPersistedTargetLanguage,
  liveSubtitleStyleSettings,
  loadPersistedLiveSubtitleStyleSettings,
  targetLanguage,
} from "../services/workspace-settings";
import LiveSubtitlePanel from "./LiveSubtitlePanel.vue";
import "../styles/quick-translation.css";

type QuickTranslationEvent = {
  text: string | null;
  error: string | null;
};

type QuickWorkflowState = "idle" | "processing" | "result" | "error";

const QUICK_TRANSLATION_EVENT = "quick-translation-request";
const quickWindow = getCurrentWindow();
const quickOverlay = ref<HTMLElement | null>(null);
const sourceText = ref("");
const translatedText = ref("");
const errorMessage = ref("");
const statusMessage = ref("选择文字后按快捷键翻译。");
const progress = ref(0);
const workflowState = ref<QuickWorkflowState>("idle");
const activeController = ref<AbortController | null>(null);
let eventUnlisten: UnlistenFn | undefined;
let closeUnlisten: UnlistenFn | undefined;
let progressUnlisten: UnlistenFn | undefined;
let resizeObserver: ResizeObserver | undefined;
let resizeFrame: number | undefined;
let resizeInFlight = false;
let resizePending = false;
let lastRequestedSize: { width: number; height: number } | undefined;
let requestVersion = 0;

function clearProgressListener(): void {
  progressUnlisten?.();
  progressUnlisten = undefined;
}

const quickLiveState = computed<LiveSessionState>(() => {
  if (workflowState.value === "processing") {
    return "running";
  }
  if (workflowState.value === "error") {
    return "error";
  }
  return "idle";
});

const quickSubtitleText = computed(() => {
  if (workflowState.value === "processing") {
    return statusMessage.value;
  }
  if (workflowState.value === "error") {
    return errorMessage.value;
  }
  return translatedText.value;
});

const quickSubtitle = computed<LiveSubtitle>(() => ({
  sessionId: "quick-translation",
  revision: requestVersion,
  sourceText: sourceText.value,
  translatedText: quickSubtitleText.value,
  roi: {
    x: 0,
    y: 0,
    width: 1,
    height: 1,
    clientWidth: 1,
    clientHeight: 1,
  },
  regions: [],
  isStreaming: workflowState.value === "processing",
  observedAtEpochMs: Date.now(),
}));
const quickSubtitleProgress = computed<SubtitleProgress>(() => ({
  mode: "translation",
  active: workflowState.value === "processing",
  overall: progress.value,
  ocr: 0,
  translation: progress.value,
  label: statusMessage.value,
}));

function clearResult(): void {
  translatedText.value = "";
  errorMessage.value = "";
  progress.value = 0;
}

function scheduleWindowResize(): void {
  if (resizeFrame !== undefined) {
    return;
  }
  resizeFrame = window.requestAnimationFrame(() => {
    resizeFrame = undefined;
    void syncWindowSize();
  });
}

async function syncWindowSize(): Promise<void> {
  const overlay = quickOverlay.value;
  const panel = overlay?.querySelector<HTMLElement>(".subtitle-panel");
  if (!overlay || !panel) {
    return;
  }
  if (resizeInFlight) {
    resizePending = true;
    return;
  }
  const bounds = panel.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) {
    return;
  }
  const size = {
    width: Math.max(160, Math.ceil(bounds.width)),
    height: Math.max(72, Math.ceil(bounds.height)),
  };
  if (
    lastRequestedSize?.width === size.width &&
    lastRequestedSize.height === size.height
  ) {
    return;
  }
  resizeInFlight = true;
  try {
    await quickWindow.setSize(new LogicalSize(size.width, size.height));
    lastRequestedSize = size;
  } catch {
    // The quick window may close while content is being measured.
  } finally {
    resizeInFlight = false;
    if (resizePending) {
      resizePending = false;
      scheduleWindowResize();
    }
  }
}

async function hideQuickWindow(): Promise<void> {
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  await quickWindow.hide().catch(() => undefined);
}

async function translateSelection(payload: QuickTranslationEvent): Promise<void> {
  const currentVersion = ++requestVersion;
  activeController.value?.abort();
  activeController.value = null;
  clearProgressListener();
  clearResult();

  if (payload.error || !payload.text?.trim()) {
    sourceText.value = "";
    workflowState.value = "error";
    errorMessage.value = payload.error || "未检测到可翻译的文字选区。";
    statusMessage.value = errorMessage.value;
    progress.value = 0;
    await nextTick();
    scheduleWindowResize();
    return;
  }

  sourceText.value = payload.text;
  workflowState.value = "processing";
  statusMessage.value = "正在读取本地翻译模型。";
  await nextTick();
  scheduleWindowResize();

  const controller = new AbortController();
  activeController.value = controller;
  const requestId = createTranslationRequestId();

  try {
    progressUnlisten = await listen<TranslationProgress>("translation-progress", (event) => {
      if (
        event.payload.requestId !== requestId ||
        currentVersion !== requestVersion ||
        activeController.value !== controller
      ) {
        return;
      }
      progress.value = Math.min(100, Math.max(0, Math.round(event.payload.progress)));
      statusMessage.value = event.payload.stage;
      scheduleWindowResize();
    });

    const result = await textTranslationProvider.translate(
      {
        text: payload.text,
        targetLanguage: targetLanguage.value,
        requestId,
      },
      controller.signal,
    );

    if (
      controller.signal.aborted ||
      currentVersion !== requestVersion ||
      activeController.value !== controller
    ) {
      return;
    }

    translatedText.value = result.text;
    progress.value = 100;
    workflowState.value = "result";
    statusMessage.value = "快捷翻译完成。";
    await nextTick();
    scheduleWindowResize();
  } catch (error) {
    if (controller.signal.aborted || isTranslationCancellation(error)) {
      return;
    }
    if (currentVersion !== requestVersion || activeController.value !== controller) {
      return;
    }
    workflowState.value = "error";
    errorMessage.value = error instanceof Error ? error.message : "快捷翻译未完成。";
    statusMessage.value = errorMessage.value;
    await nextTick();
    scheduleWindowResize();
  } finally {
    clearProgressListener();
    if (activeController.value === controller) {
      activeController.value = null;
    }
  }
}

async function copyResult(): Promise<void> {
  if (!translatedText.value) {
    return;
  }
  await copyTranslationText(translatedText.value).catch(() => undefined);
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") {
    event.preventDefault();
    void hideQuickWindow();
  }
}

async function initialize(): Promise<void> {
  loadPersistedTargetLanguage();
  loadPersistedLiveSubtitleStyleSettings();
  await fetchSharedBackendStatus().catch(() => undefined);
  eventUnlisten = await listen<QuickTranslationEvent>(QUICK_TRANSLATION_EVENT, (event) => {
    void translateSelection(event.payload);
  });
  closeUnlisten = await quickWindow.onCloseRequested((event) => {
    event.preventDefault();
    void hideQuickWindow();
  });
}

onMounted(() => {
  resizeObserver = new ResizeObserver(() => scheduleWindowResize());
  if (quickOverlay.value) {
    resizeObserver.observe(quickOverlay.value);
  }
  window.addEventListener("keydown", handleKeydown, true);
  void initialize().catch((error) => {
    workflowState.value = "error";
    errorMessage.value = error instanceof Error ? error.message : "快捷翻译初始化失败。";
    statusMessage.value = errorMessage.value;
    scheduleWindowResize();
  });
});

onBeforeUnmount(() => {
  requestVersion += 1;
  activeController.value?.abort();
  activeController.value = null;
  resizeObserver?.disconnect();
  if (resizeFrame !== undefined) {
    window.cancelAnimationFrame(resizeFrame);
  }
  eventUnlisten?.();
  closeUnlisten?.();
  window.removeEventListener("keydown", handleKeydown, true);
});
</script>
<template>

  <main ref="quickOverlay" class="quick-translation-overlay" aria-label="快捷翻译" aria-live="polite">
    <LiveSubtitlePanel
      :session-id="''"
      :state="quickLiveState"
      :subtitle="quickSubtitle"
      :progress="quickSubtitleProgress"
      :show-source="true"
      :show-close="true"
      :style-settings="liveSubtitleStyleSettings"
      sizing-mode="content"
      @close="hideQuickWindow"
    />

    <div class="quick-translation-controls" aria-label="快捷翻译操作">
      <n-button
        v-if="workflowState === 'result'"
        class="quick-translation-action"
        secondary
        size="small"
        aria-label="复制译文"
        title="复制译文"
        @click.stop="copyResult"
      >
        复制
      </n-button>
    </div>

    <span class="quick-translation-sr-only" role="status">{{ statusMessage }}</span>
  </main>
</template>

