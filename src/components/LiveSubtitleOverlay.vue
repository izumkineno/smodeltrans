<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  getLiveSessionStatus,
  groupLiveSubtitleRegions,
  resolveLiveSubtitleRegionVerticalAnchor,
  listenLiveRegionBoxesVisible,
  listenLiveStatus,
  listenLiveSubtitle,
  shouldApplyLiveSubtitle,
} from "../services/live-translation-provider";
import type {
  LiveRoi,
  LiveSessionState,
  LiveSubtitle,
  LiveSubtitleRegionFlowGroup,
  LiveSubtitleRegionFlowItem,
} from "../services/live-translation-provider";

const query = new URLSearchParams(window.location.search);
const sessionId = query.get("liveSessionId") ?? undefined;
const isRegionReplace = query.get("liveOverlayMode") === "region_replace";
const showSource = query.get("showSource") !== "0";
const showRegionBoxes = ref(query.get("showRegionBoxes") === "1");
const subtitle = ref<LiveSubtitle>();
const state = ref<LiveSessionState>("warming");
let lastRevision = -1;
let listenersActive = true;
const unlisteners: UnlistenFn[] = [];

const visible = computed(
  () =>
    state.value === "warming" ||
    (isRegionReplace
      ? (subtitle.value?.regions.length ?? 0) > 0
      : !!subtitle.value?.isStreaming || !!subtitle.value?.translatedText.trim()),
);
const regionGroups = computed(() =>
  groupLiveSubtitleRegions(subtitle.value?.regions ?? []),
);

function percent(value: number, total: number): string {
  return `${(value / total) * 100}%`;
}

function regionGroupStyle(
  group: LiveSubtitleRegionFlowGroup,
): Record<string, string> {
  const roi = subtitle.value?.roi;
  if (!roi || roi.clientWidth <= 0 || roi.clientHeight <= 0) {
    return {};
  }
  const anchor = resolveLiveSubtitleRegionVerticalAnchor(group, roi.clientHeight);
  if (!anchor) {
    return {};
  }
  const style: Record<string, string> = {
    left: percent(group.left, roi.clientWidth),
    width: percent(group.width, roi.clientWidth),
  };
  style[anchor.edge] = percent(anchor.offset, roi.clientHeight);
  return style;
}

function regionItemStyle(
  item: LiveSubtitleRegionFlowItem,
  group: LiveSubtitleRegionFlowGroup,
  roi: LiveRoi | undefined,
): Record<string, string> {
  if (!roi || roi.clientHeight <= 0 || group.width <= 0) {
    return {};
  }
  return {
    width: percent(item.width, group.width),
    minHeight: `${(item.region.bounds.height / roi.clientHeight) * 100}vh`,
    marginTop: `${(item.gapAbove / roi.clientHeight) * 100}vh`,
    marginLeft: percent(item.leftOffset, group.width),
  };
}

function regionDebugLabel(
  item: LiveSubtitleRegionFlowItem,
  groupIndex: number,
): string {
  return `${item.index + 1} · y=${item.region.bounds.top} · 块=${groupIndex + 1}`;
}

function applySubtitle(next: LiveSubtitle): void {
  if (!shouldApplyLiveSubtitle(next, sessionId, lastRevision)) {
    return;
  }
  lastRevision = next.revision;
  subtitle.value = next.translatedText.trim() || next.sourceText.trim() ? next : undefined;
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
  const status = await getLiveSessionStatus();
  if (!sessionId || status.sessionId === sessionId || status.state === "idle") {
    state.value = status.state;
  }
}

onMounted(() => {
  void initialize().catch(() => {
    state.value = "error";
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
    <div v-if="visible && state === 'warming'" class="subtitle-panel subtitle-panel-warming">
      <p class="translated-text">正在连接窗口捕获…</p>
    </div>
    <div v-else-if="visible && isRegionReplace" class="region-replace-layer">
      <section
        v-for="(group, groupIndex) in regionGroups"
        :key="group.id"
        class="region-replace-group"
        :style="regionGroupStyle(group)"
      >
        <article
          v-for="item in group.items"
          :key="item.id"
          :class="{ 'region-replace-item-debug': showRegionBoxes }"
          :data-region-debug="showRegionBoxes ? regionDebugLabel(item, groupIndex) : undefined"
          class="region-replace-item"
          :style="regionItemStyle(item, group, subtitle?.roi)"
        >
          <p v-if="showSource" class="region-source-text">{{ item.region.sourceText }}</p>
          <p class="region-translated-text">{{ item.region.translatedText }}</p>
        </article>
      </section>
    </div>
    <div v-else-if="visible" class="subtitle-panel">
      <p class="translated-text">{{ subtitle?.translatedText || "翻译中…" }}</p>
      <p v-if="showSource && subtitle?.sourceText" class="source-text">{{ subtitle.sourceText }}</p>
    </div>
  </main>
</template>

<style>
:root,
html,
body,
#app {
  width: 100%;
  height: 100%;
  margin: 0;
  overflow: hidden;
  background: transparent !important;
}

* {
  box-sizing: border-box;
}

.subtitle-overlay {
  display: flex;
  width: 100%;
  height: 100%;
  align-items: flex-end;
  justify-content: center;
  padding: 12px clamp(18px, 4vw, 72px) 18px;
  pointer-events: none;
  user-select: none;
}

.subtitle-overlay-region-replace {
  display: block;
  padding: 0;
}

.subtitle-panel {
  width: min(1100px, 100%);
  padding: 11px 20px 10px;
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 8px;
  color: #ffffff;
  background: rgba(6, 12, 24, 0.82);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.34);
  text-align: center;
  backdrop-filter: blur(10px);
}

.subtitle-panel-warming {
  color: rgba(226, 232, 240, 0.9);
  background: rgba(6, 12, 24, 0.7);
}

.translated-text,
.source-text,
.region-source-text,
.region-translated-text {
  margin: 0;
  overflow-wrap: anywhere;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
}

.translated-text,
.source-text,
.region-source-text,
.region-translated-text {
  white-space: pre-wrap;
}

.translated-text {
  font: 600 clamp(18px, 2.2vw, 28px) / 1.35 "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
  letter-spacing: 0.01em;
}

.source-text {
  margin-top: 5px;
  color: rgba(226, 232, 240, 0.82);
  font: 400 clamp(11px, 1.25vw, 15px) / 1.35 "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}

.region-replace-layer {
  position: relative;
  width: 100%;
  height: 100%;
}

.region-replace-group {
  position: absolute;
  display: flex;
  min-width: 1%;
  flex-direction: column;
  align-items: flex-start;
}

.region-replace-item {
  position: relative;
  display: flex;
  min-width: 1%;
  flex: 0 0 auto;
  flex-direction: column;
  justify-content: center;
  overflow: hidden;
  padding: 3px 5px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 3px;
  color: #ffffff;
  background: rgba(6, 12, 24, 0.82);
  text-align: center;
}

.region-replace-item-debug {
  outline: 2px solid #00e5ff;
  outline-offset: -2px;
  background: rgba(0, 76, 112, 0.72);
}

.region-replace-item-debug::before {
  position: absolute;
  top: 0;
  left: 0;
  min-width: 16px;
  padding: 1px 4px;
  color: #001018;
  background: #00e5ff;
  content: attr(data-region-debug);
  font: 700 10px/1.4 "Segoe UI", sans-serif;
  text-shadow: none;
}

.region-source-text {
  margin-bottom: 2px;
  color: rgba(226, 232, 240, 0.82);
  font: 400 clamp(9px, 1vw, 13px) / 1.2 "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}

.region-translated-text {
  font: 600 clamp(12px, 1.6vw, 24px) / 1.2 "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}
</style>
