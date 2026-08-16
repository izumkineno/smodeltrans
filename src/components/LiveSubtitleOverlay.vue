<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
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
const showSource = query.get("showSource") !== "0";
const showRegionBoxes = ref(query.get("showRegionBoxes") === "1");
const subtitle = ref<LiveSubtitle>();
const state = ref<LiveSessionState>("warming");
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
      :show-source="showSource"
      :style-settings="liveSubtitleStyleSettings"
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
