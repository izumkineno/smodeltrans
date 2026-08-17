<script setup lang="ts">
import { getCurrentWindow } from "@tauri-apps/api/window";
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import {
  getLiveSessionStatus,
  getLiveSubtitle,
  hasRenderableLiveSubtitleContent,
  listenLiveRegionBoxesVisible,
  listenLiveStatus,
  listenLiveSubtitle,
  shouldApplyLiveSubtitle,
} from "../services/live-translation-provider";
import type {
  LiveSessionState,
  LiveSubtitle,
  SubtitleProgress,
} from "../services/live-translation-provider";
import {
  liveSubtitleStyleSettings,
  loadPersistedLiveSubtitleStyleSettings,
} from "../services/workspace-settings";
import LiveRegionReplaceOverlay from "./LiveRegionReplaceOverlay.vue";
import LiveSubtitlePanel from "./LiveSubtitlePanel.vue";

const query = new URLSearchParams(window.location.search);
const sessionId = query.get("liveSessionId") ?? undefined;
const isRegionReplace = query.get("liveOverlayMode") === "region_replace";
const overlayWindow = getCurrentWindow();
const showSource = query.get("showSource") !== "0";
const showRegionBoxes = ref(query.get("showRegionBoxes") === "1");
const subtitle = ref<LiveSubtitle>();
const state = ref<LiveSessionState>("warming");
const liveStatusMessage = ref("");
const liveProgress = computed<SubtitleProgress | undefined>(() => {
  if (state.value !== "warming" && state.value !== "running") {
    return undefined;
  }
  if (state.value === "warming") {
    return {
      mode: "live",
      active: true,
      overall: 10,
      ocr: 10,
      translation: 0,
      label: "正在准备 OCR 与翻译模型",
    };
  }
  const current = subtitle.value;
  const message = liveStatusMessage.value;
  const ocrActive =
    !message.includes("字幕已更新") &&
    /正在.*OCR|执行 OCR|检测到触发键|等待字幕稳定/.test(message);
  if (!current || ocrActive) {
    return {
      mode: "live",
      active: true,
      overall: 25,
      ocr: 25,
      translation: 0,
      label: message || "正在 OCR 识别",
    };
  }
  if (current.isStreaming || /翻译|Hy-MT2/.test(message)) {
    return {
      mode: "live",
      active: true,
      overall: 75,
      ocr: 100,
      translation: 50,
      label: message || "正在翻译字幕",
    };
  }
  return {
    mode: "live",
    active: true,
    overall: 100,
    ocr: 100,
    translation: 100,
    label: message || "字幕处理完成",
  };
});

function closeOverlay(): void {
  void overlayWindow.close().catch(() => undefined);
}
let lastRevision = -1;
let listenersActive = true;
const unlisteners: Array<() => void> = [];

function applySubtitle(next: LiveSubtitle): void {
  if (!shouldApplyLiveSubtitle(next, sessionId, lastRevision)) {
    return;
  }
  lastRevision = next.revision;
  subtitle.value = hasRenderableLiveSubtitleContent(next) ? next : undefined;
}

async function initialize(): Promise<void> {
  const registered = await Promise.all([
    listenLiveSubtitle(applySubtitle),
    listenLiveStatus((status) => {
      liveStatusMessage.value = status.message;
      if (!sessionId || status.sessionId === sessionId || status.state === "idle") {
        state.value = status.state;
      }
    }),
    listenLiveRegionBoxesVisible((visible) => {
      showRegionBoxes.value = visible;
    }),
  ]);
  if (!listenersActive) {
    registered.forEach((unlisten) => unlisten());
    return;
  }
  unlisteners.push(...registered);

  let latestSubtitle: LiveSubtitle | null = null;
  try {
    latestSubtitle = await getLiveSubtitle();
  } catch {
    // The snapshot is a recovery path; live event listeners remain authoritative.
  }
  if (latestSubtitle) {
    applySubtitle(latestSubtitle);
  }

  const status = await getLiveSessionStatus();
  liveStatusMessage.value = status.message;
  if (!sessionId || status.sessionId === sessionId || status.state === "idle") {
    state.value = status.state;
  }
}

onMounted(() => {
  loadPersistedLiveSubtitleStyleSettings();
  void initialize().catch((error) => {
    state.value = "error";
    console.error("实时字幕浮层初始化失败", error);
  });
});

onBeforeUnmount(() => {
  listenersActive = false;
  unlisteners.splice(0).forEach((unlisten) => unlisten());
});
</script>

<template>
  <main
    class="subtitle-overlay"
    :class="{ 'subtitle-overlay-region-replace': isRegionReplace }"
    aria-live="polite"
    aria-atomic="true"
  >
    <LiveSubtitlePanel
      v-if="!isRegionReplace"
      :session-id="sessionId ?? ''"
      :state="state"
      :subtitle="subtitle"
      :progress="liveProgress"
      :show-source="showSource"
      :show-close="true"
      :style-settings="liveSubtitleStyleSettings"
      @close="closeOverlay"
    />
    <LiveRegionReplaceOverlay
      v-else
      :state="state"
      :subtitle="subtitle"
      :show-source="showSource"
      :show-region-boxes="showRegionBoxes"
      :style-settings="liveSubtitleStyleSettings"
    />
  </main>
</template>

<style src="../styles/live-subtitle-overlay.css"></style>
