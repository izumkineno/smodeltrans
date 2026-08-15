<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import type { ComponentPublicInstance, CSSProperties } from "vue";
import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  getLiveSessionStatus,
  groupLiveSubtitleRegions,
  resolveLiveSubtitleRegionVerticalAnchor,
  listenLiveRegionBoxesVisible,
  listenLiveStatus,
  listenLiveSubtitle,
  shouldApplyLiveSubtitle,
  updateLiveOverlayLayout,
} from "../services/live-translation-provider";
import type {
  LiveOverlayContentSize,
  LiveOverlaySizing,
  LiveRoi,
  LiveSessionState,
  LiveSubtitle,
  LiveSubtitleRegionFlowGroup,
  LiveSubtitleRegionFlowItem,
} from "../services/live-translation-provider";
import {
  LIVE_SUBTITLE_MANUAL_HEIGHT_MAX,
  LIVE_SUBTITLE_MANUAL_HEIGHT_MIN,
  LIVE_SUBTITLE_MANUAL_WIDTH_MAX,
  LIVE_SUBTITLE_MANUAL_WIDTH_MIN,
  liveOverlaySettings,
  loadPersistedLiveOverlaySettings,
  savePersistedLiveOverlaySettings,
} from "../services/workspace-settings";

const HORIZONTAL_OVERLAY_PADDING = 36;
const VERTICAL_OVERLAY_PADDING = 30;

interface ManualResizeState {
  pointerId: number;
  startX: number;
  startY: number;
  startWidth: number;
  startHeight: number;
}

const query = new URLSearchParams(window.location.search);
const sessionId = query.get("liveSessionId") ?? undefined;
const isRegionReplace = query.get("liveOverlayMode") === "region_replace";
const showSource = query.get("showSource") !== "0";
const showRegionBoxes = ref(query.get("showRegionBoxes") === "1");
const subtitle = ref<LiveSubtitle>();
const state = ref<LiveSessionState>("warming");
const subtitleOverlay = ref<HTMLElement | null>(null);
const subtitlePanel = ref<HTMLElement | null>(null);
const sizingEditorOpen = ref(false);
const layoutError = ref("");
const manualResizeActive = ref(false);
let lastRevision = -1;
let listenersActive = true;
let layoutSyncInFlight = false;
let layoutSyncPending = false;
let layoutAnimationFrame: number | undefined;
let lastLayoutSignature = "";
let panelResizeObserver: ResizeObserver | undefined;
let manualResize: ManualResizeState | undefined;
const unlisteners: UnlistenFn[] = [];

function queryPhysicalDimension(name: string, fallback: number): number {
  const value = Number.parseInt(query.get(name) ?? "", 10);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

const deviceScaleFactor = Math.max(window.devicePixelRatio || 1, 1);
const automaticPanelWidthLimit = Math.max(
  1,
  Math.floor(queryPhysicalDimension("overlayMaxWidth", 1_280) / deviceScaleFactor) -
    HORIZONTAL_OVERLAY_PADDING,
);
const automaticPanelHeightLimit = Math.max(
  1,
  Math.floor(
    queryPhysicalDimension("overlayMaxHeight", 720) / deviceScaleFactor,
  ) - VERTICAL_OVERLAY_PADDING,
);

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
const sizingEnabled = computed(() => !isRegionReplace && sessionId !== undefined);
const subtitlePanelStyle = computed<CSSProperties>(() => ({
  width: liveOverlaySettings.value.autoWidth ? "max-content" : "100%",
  height: liveOverlaySettings.value.autoHeight ? "fit-content" : "100%",
  maxWidth: liveOverlaySettings.value.autoWidth
    ? `${automaticPanelWidthLimit}px`
    : "100%",
  maxHeight: liveOverlaySettings.value.autoHeight
    ? `${automaticPanelHeightLimit}px`
    : "100%",
  flex: liveOverlaySettings.value.autoWidth ? "0 0 auto" : "0 1 auto",
}));

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

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function currentSizing(): LiveOverlaySizing {
  const { autoWidth, autoHeight, manualWidth, manualHeight } = liveOverlaySettings.value;
  return { autoWidth, autoHeight, manualWidth, manualHeight };
}
function cssPixel(value: string, fallback: number): number {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}


function measuredContentSize(): LiveOverlayContentSize | undefined {
  const panel = subtitlePanel.value;
  const overlay = subtitleOverlay.value;
  if (!panel || !overlay) {
    return undefined;
  }
  const bounds = panel.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) {
    return undefined;
  }
  const overlayStyle = window.getComputedStyle(overlay);
  const horizontalPadding =
    cssPixel(overlayStyle.paddingLeft, HORIZONTAL_OVERLAY_PADDING / 2) +
    cssPixel(overlayStyle.paddingRight, HORIZONTAL_OVERLAY_PADDING / 2);
  const verticalPadding =
    cssPixel(overlayStyle.paddingTop, VERTICAL_OVERLAY_PADDING / 2) +
    cssPixel(overlayStyle.paddingBottom, VERTICAL_OVERLAY_PADDING / 2);
  const width = Math.max(bounds.width, panel.scrollWidth);
  const height = Math.max(bounds.height, panel.scrollHeight);
  const scaleFactor = Math.max(window.devicePixelRatio || 1, 1);
  return {
    width: Math.max(1, Math.ceil((width + horizontalPadding) * scaleFactor)),
    height: Math.max(1, Math.ceil((height + verticalPadding) * scaleFactor)),
  };
}

function scheduleLayoutSync(force = false): void {
  if (!sizingEnabled.value || typeof window === "undefined") {
    return;
  }
  if (force) {
    lastLayoutSignature = "";
  }
  if (layoutAnimationFrame !== undefined) {
    layoutSyncPending = true;
    return;
  }
  layoutAnimationFrame = window.requestAnimationFrame(() => {
    layoutAnimationFrame = undefined;
    void syncOverlayLayout();
  });
}

async function syncOverlayLayout(): Promise<void> {
  const contentSize = measuredContentSize();
  if (!sessionId || !contentSize) {
    return;
  }
  const sizing = currentSizing();
  const signature = [
    sizing.autoWidth,
    sizing.autoHeight,
    sizing.manualWidth,
    sizing.manualHeight,
    contentSize.width,
    contentSize.height,
  ].join(":");
  if (signature === lastLayoutSignature) {
    return;
  }
  if (layoutSyncInFlight) {
    layoutSyncPending = true;
    return;
  }
  layoutSyncInFlight = true;
  try {
    await updateLiveOverlayLayout(sessionId, sizing, contentSize);
    lastLayoutSignature = signature;
    layoutError.value = "";
  } catch (error) {
    layoutError.value = `尺寸同步失败：${errorText(error)}`;
  } finally {
    layoutSyncInFlight = false;
    if (layoutSyncPending) {
      layoutSyncPending = false;
      scheduleLayoutSync();
    }
  }
}
function setSubtitlePanel(element: Element | ComponentPublicInstance | null): void {
  const panel = element instanceof HTMLElement ? element : null;
  if (subtitlePanel.value === panel) {
    return;
  }
  panelResizeObserver?.disconnect();
  subtitlePanel.value = panel;
  if (panel && panelResizeObserver) {
    panelResizeObserver.observe(panel);
  }
  void nextTick().then(() => scheduleLayoutSync());
}

function persistSizingChange(): void {
  const persistenceError = savePersistedLiveOverlaySettings();
  if (persistenceError) {
    layoutError.value = persistenceError;
    return;
  }
  layoutError.value = "";
  void nextTick().then(() => scheduleLayoutSync(true));
}

function toggleAutoWidth(): void {
  liveOverlaySettings.value.autoWidth = !liveOverlaySettings.value.autoWidth;
  persistSizingChange();
}

function toggleAutoHeight(): void {
  liveOverlaySettings.value.autoHeight = !liveOverlaySettings.value.autoHeight;
  persistSizingChange();
}

function toggleSizingEditor(): void {
  sizingEditorOpen.value = !sizingEditorOpen.value;
  void nextTick().then(() => scheduleLayoutSync());
}

function commitManualDimension(axis: "width" | "height", event: Event): void {
  const input = event.target as HTMLInputElement;
  const value = Number(input.value);
  const minimum =
    axis === "width" ? LIVE_SUBTITLE_MANUAL_WIDTH_MIN : LIVE_SUBTITLE_MANUAL_HEIGHT_MIN;
  const maximum =
    axis === "width" ? LIVE_SUBTITLE_MANUAL_WIDTH_MAX : LIVE_SUBTITLE_MANUAL_HEIGHT_MAX;
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    layoutError.value = `${axis === "width" ? "宽度" : "高度"}必须为 ${minimum} 到 ${maximum} 的整数。`;
    input.value = String(
      axis === "width"
        ? liveOverlaySettings.value.manualWidth
        : liveOverlaySettings.value.manualHeight,
    );
    return;
  }
  if (
    (axis === "width" &&
      liveOverlaySettings.value.manualWidth === value &&
      !liveOverlaySettings.value.autoWidth) ||
    (axis === "height" &&
      liveOverlaySettings.value.manualHeight === value &&
      !liveOverlaySettings.value.autoHeight)
  ) {
    return;
  }
  if (axis === "width") {
    liveOverlaySettings.value.manualWidth = value;
    liveOverlaySettings.value.autoWidth = false;
  } else {
    liveOverlaySettings.value.manualHeight = value;
    liveOverlaySettings.value.autoHeight = false;
  }
  persistSizingChange();
}

function clampManualDimension(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, Math.round(value)));
}

function beginManualResize(event: PointerEvent): void {
  if (event.button !== 0 || !subtitleOverlay.value) {
    return;
  }
  const overlayBounds = subtitleOverlay.value.getBoundingClientRect();
  const scaleFactor = Math.max(window.devicePixelRatio || 1, 1);
  const handle = event.currentTarget as HTMLElement;
  manualResize = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    startWidth: clampManualDimension(
      overlayBounds.width * scaleFactor,
      LIVE_SUBTITLE_MANUAL_WIDTH_MIN,
      LIVE_SUBTITLE_MANUAL_WIDTH_MAX,
    ),
    startHeight: clampManualDimension(
      overlayBounds.height * scaleFactor,
      LIVE_SUBTITLE_MANUAL_HEIGHT_MIN,
      LIVE_SUBTITLE_MANUAL_HEIGHT_MAX,
    ),
  };
  liveOverlaySettings.value.manualWidth = manualResize.startWidth;
  liveOverlaySettings.value.manualHeight = manualResize.startHeight;
  liveOverlaySettings.value.autoWidth = false;
  liveOverlaySettings.value.autoHeight = false;
  layoutError.value = "";
  manualResizeActive.value = true;
  handle.setPointerCapture(event.pointerId);
  scheduleLayoutSync(true);
}

function resizeManually(event: PointerEvent): void {
  const activeResize = manualResize;
  if (!activeResize || activeResize.pointerId !== event.pointerId) {
    return;
  }
  const scaleFactor = Math.max(window.devicePixelRatio || 1, 1);
  liveOverlaySettings.value.manualWidth = clampManualDimension(
    activeResize.startWidth + (event.clientX - activeResize.startX) * scaleFactor,
    LIVE_SUBTITLE_MANUAL_WIDTH_MIN,
    LIVE_SUBTITLE_MANUAL_WIDTH_MAX,
  );
  liveOverlaySettings.value.manualHeight = clampManualDimension(
    activeResize.startHeight + (event.clientY - activeResize.startY) * scaleFactor,
    LIVE_SUBTITLE_MANUAL_HEIGHT_MIN,
    LIVE_SUBTITLE_MANUAL_HEIGHT_MAX,
  );
  scheduleLayoutSync(true);
}

function finishManualResize(event: PointerEvent): void {
  if (!manualResize || manualResize.pointerId !== event.pointerId) {
    return;
  }
  manualResize = undefined;
  manualResizeActive.value = false;
  const handle = event.currentTarget as HTMLElement;
  if (handle.hasPointerCapture(event.pointerId)) {
    handle.releasePointerCapture(event.pointerId);
  }
  persistSizingChange();
}

function applySubtitle(next: LiveSubtitle): void {
  if (!shouldApplyLiveSubtitle(next, sessionId, lastRevision)) {
    return;
  }
  lastRevision = next.revision;
  subtitle.value = next.translatedText.trim() || next.sourceText.trim() ? next : undefined;
  void nextTick().then(() => scheduleLayoutSync());
}

async function initialize(): Promise<void> {
  const registered = await Promise.all([
    listenLiveSubtitle(applySubtitle),
    listenLiveStatus((status) => {
      if (!sessionId || status.sessionId === sessionId || status.state === "idle") {
        state.value = status.state;
        void nextTick().then(() => scheduleLayoutSync());
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
    void nextTick().then(() => scheduleLayoutSync());
  }
}

onMounted(() => {
  const persistenceError = loadPersistedLiveOverlaySettings();
  if (persistenceError) {
    layoutError.value = persistenceError;
  }
  if (!isRegionReplace && typeof ResizeObserver !== "undefined") {
    panelResizeObserver = new ResizeObserver(() => scheduleLayoutSync());
    if (subtitlePanel.value) {
      panelResizeObserver.observe(subtitlePanel.value);
    }
  }
  void initialize().catch((error) => {
    state.value = "error";
    layoutError.value = errorText(error);
  });
});

onBeforeUnmount(() => {
  manualResize = undefined;
  manualResizeActive.value = false;
  listenersActive = false;
  if (layoutAnimationFrame !== undefined) {
    window.cancelAnimationFrame(layoutAnimationFrame);
  }
  panelResizeObserver?.disconnect();
  unlisteners.splice(0).forEach((unlisten) => unlisten());
});
</script>

<template>
  <main
    ref="subtitleOverlay"
    class="subtitle-overlay"
    :class="{ 'subtitle-overlay-region-replace': isRegionReplace }"
    aria-live="polite"
    aria-atomic="true"
  >
    <div
      v-if="visible && !isRegionReplace"
      :ref="setSubtitlePanel"
      class="subtitle-panel"
      :class="{
        'subtitle-panel-warming': state === 'warming',
        'is-manual-resizing': manualResizeActive,
      }"
      :style="subtitlePanelStyle"
    >
      <div v-if="sizingEnabled" class="subtitle-panel-toolbar" aria-label="字幕窗口尺寸设置">
        <button
          type="button"
          class="subtitle-tool-button"
          :class="{ 'is-active': liveOverlaySettings.autoWidth }"
          :aria-pressed="liveOverlaySettings.autoWidth"
          :aria-label="liveOverlaySettings.autoWidth ? '关闭自适应宽度' : '开启自适应宽度'"
          :title="liveOverlaySettings.autoWidth ? '关闭自适应宽度' : '开启自适应宽度'"
          @click="toggleAutoWidth"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M2 8h12M4.5 5.5 2 8l2.5 2.5M11.5 5.5 14 8l-2.5 2.5" />
            <path d="M5.5 4v8M10.5 4v8" />
          </svg>
        </button>
        <button
          type="button"
          class="subtitle-tool-button"
          :class="{ 'is-active': liveOverlaySettings.autoHeight }"
          :aria-pressed="liveOverlaySettings.autoHeight"
          :aria-label="liveOverlaySettings.autoHeight ? '关闭自适应高度' : '开启自适应高度'"
          :title="liveOverlaySettings.autoHeight ? '关闭自适应高度' : '开启自适应高度'"
          @click="toggleAutoHeight"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M8 2v12M5.5 4.5 8 2l2.5 2.5M5.5 11.5 8 14l2.5-2.5" />
            <path d="M4 5.5h8M4 10.5h8" />
          </svg>
        </button>
        <button
          type="button"
          class="subtitle-tool-button"
          :class="{ 'is-active': sizingEditorOpen }"
          :aria-expanded="sizingEditorOpen"
          aria-label="编辑手动宽度和高度"
          title="编辑手动宽度和高度"
          @click="toggleSizingEditor"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect x="2.5" y="2.5" width="11" height="11" rx="1.5" />
            <path d="M5 11 11 5M8.5 4.8H11.2v2.7" />
          </svg>
        </button>
        <button
          type="button"
          class="subtitle-tool-button subtitle-resize-handle"
          :class="{ 'is-active': manualResizeActive }"
          aria-label="拖动调整字幕窗口宽度和高度"
          title="拖动调整字幕窗口宽度和高度"
          @pointerdown.prevent="beginManualResize"
          @pointermove.prevent="resizeManually"
          @pointerup.prevent="finishManualResize"
          @pointercancel.prevent="finishManualResize"
          @lostpointercapture="finishManualResize"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="m5 11 6-6M8 11l3-3M5 8l3-3" />
          </svg>
        </button>

        <div v-if="sizingEditorOpen" class="subtitle-sizing-editor" aria-label="手动字幕尺寸">
          <label class="subtitle-size-field" title="手动宽度（物理像素）">
            <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M2 8h12M4.5 5.5 2 8l2.5 2.5M11.5 5.5 14 8l-2.5 2.5" />
            </svg>
            <input
              :value="liveOverlaySettings.manualWidth"
              type="number"
              inputmode="numeric"
              :min="LIVE_SUBTITLE_MANUAL_WIDTH_MIN"
              :max="LIVE_SUBTITLE_MANUAL_WIDTH_MAX"
              step="1"
              aria-label="手动字幕宽度（物理像素）"
              @change="commitManualDimension('width', $event)"
              @keydown.enter.prevent="commitManualDimension('width', $event)"
            />
          </label>
          <label class="subtitle-size-field" title="手动高度（物理像素）">
            <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M8 2v12M5.5 4.5 8 2l2.5 2.5M5.5 11.5 8 14l2.5-2.5" />
            </svg>
            <input
              :value="liveOverlaySettings.manualHeight"
              type="number"
              inputmode="numeric"
              :min="LIVE_SUBTITLE_MANUAL_HEIGHT_MIN"
              :max="LIVE_SUBTITLE_MANUAL_HEIGHT_MAX"
              step="1"
              aria-label="手动字幕高度（物理像素）"
              @change="commitManualDimension('height', $event)"
              @keydown.enter.prevent="commitManualDimension('height', $event)"
            />
          </label>
        </div>
        <p v-if="layoutError" class="subtitle-layout-error" role="status">{{ layoutError }}</p>
      </div>

      <div class="subtitle-panel-content">
        <p v-if="state === 'warming'" class="translated-text">正在连接窗口捕获…</p>
        <template v-else>
          <p class="translated-text">{{ subtitle?.translatedText || "翻译中…" }}</p>
          <p v-if="showSource && subtitle?.sourceText" class="source-text">
            {{ subtitle.sourceText }}
          </p>
        </template>
      </div>
    </div>
    <div v-else-if="visible" class="region-replace-layer">
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
  position: relative;
  display: flex;
  width: 100%;
  height: 100%;
  align-items: flex-end;
  justify-content: center;
  padding: 12px 18px 18px;
  pointer-events: auto;
  user-select: none;
}

.subtitle-overlay-region-replace {
  display: block;
  padding: 0;
  pointer-events: none;
}

.subtitle-panel {
  position: relative;
  display: flex;
  min-width: 0;
  min-height: 0;
  flex: 0 0 auto;
  flex-direction: column;
  width: max-content;
  max-width: 100%;
  overflow: hidden;
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

.subtitle-panel-toolbar {
  position: sticky;
  top: 0;
  z-index: 1;
  display: flex;
  width: 100%;
  min-width: 0;
  flex: 0 0 auto;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  padding: 4px 6px;
  border-bottom: 1px solid transparent;
  background: rgba(6, 12, 24, 0.56);
  opacity: 0;
  pointer-events: none;
  transition: opacity 150ms ease, background 150ms ease;
}

.subtitle-panel:hover .subtitle-panel-toolbar,
.subtitle-panel:focus-within .subtitle-panel-toolbar,
.subtitle-panel.is-manual-resizing .subtitle-panel-toolbar {
  border-bottom-color: rgba(226, 232, 240, 0.2);
  background: rgba(6, 12, 24, 0.94);
  opacity: 1;
  pointer-events: auto;
}

.subtitle-panel-content {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex: 1 1 auto;
  flex-direction: column;
  justify-content: center;
  overflow-x: hidden;
  overflow-y: auto;
  padding: 9px 20px 10px;
}

.subtitle-panel.is-manual-resizing {
  cursor: nwse-resize;
}

.subtitle-panel.is-manual-resizing .subtitle-panel-content {
  pointer-events: none;
}

.subtitle-tool-button {
  display: grid;
  width: 26px;
  height: 26px;
  flex: 0 0 26px;
  place-items: center;
  padding: 0;
  border: 1px solid rgba(226, 232, 240, 0.32);
  border-radius: 5px;
  color: rgba(226, 232, 240, 0.84);
  background: rgba(6, 12, 24, 0.88);
  cursor: pointer;
}

.subtitle-tool-button:hover,
.subtitle-tool-button:focus-visible {
  border-color: rgba(147, 197, 253, 0.9);
  color: #ffffff;
  background: rgba(30, 64, 175, 0.9);
  outline: none;
}

.subtitle-tool-button.is-active {
  border-color: rgba(147, 197, 253, 0.92);
  color: #dbeafe;
  background: rgba(30, 64, 175, 0.88);
}

.subtitle-resize-handle {
  cursor: nwse-resize;
  touch-action: none;
}

.subtitle-tool-button svg,
.subtitle-size-field svg {
  width: 15px;
  height: 15px;
  stroke: currentColor;
  stroke-width: 1.2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.subtitle-sizing-editor {
  display: flex;
  width: 100%;
  flex: 1 0 100%;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  padding-top: 2px;
  user-select: text;
}

.subtitle-size-field {
  display: grid;
  min-width: 0;
  grid-template-columns: 15px 48px;
  align-items: center;
  gap: 4px;
  color: rgba(226, 232, 240, 0.84);
}

.subtitle-size-field input {
  width: 48px;
  min-width: 0;
  padding: 4px 5px;
  border: 1px solid rgba(226, 232, 240, 0.3);
  border-radius: 4px;
  color: #ffffff;
  background: rgba(15, 23, 42, 0.96);
  font: 600 12px/1.2 "Segoe UI", "Microsoft YaHei", sans-serif;
  font-variant-numeric: tabular-nums;
}

.subtitle-size-field input:focus {
  border-color: #93c5fd;
  outline: 2px solid rgba(147, 197, 253, 0.28);
  outline-offset: 1px;
}

.subtitle-layout-error {
  width: 100%;
  flex: 1 0 100%;
  margin: 0;
  padding: 5px 7px;
  border: 1px solid rgba(252, 165, 165, 0.6);
  border-radius: 5px;
  color: #fee2e2;
  background: rgba(127, 29, 29, 0.9);
  font: 500 11px/1.35 "Microsoft YaHei", "Segoe UI", sans-serif;
  text-align: right;
  white-space: normal;
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
