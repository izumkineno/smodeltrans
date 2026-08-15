<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";
import type { ComponentPublicInstance, CSSProperties } from "vue";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  beginLiveOverlayDrag,
  beginLiveOverlayResize,
  beginLiveRoiUpdate,
  finishLiveOverlayResize,
  getLiveSessionStatus,
  groupLiveSubtitleRegions,
  listenLiveRegionBoxesVisible,
  listenLiveStatus,
  listenLiveSubtitle,
  resolveLiveSubtitleRegionVerticalAnchor,
  shouldApplyLiveSubtitle,
  updateLiveOverlayLayout,
  updateLiveOverlayPosition,
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

type ManualResizeAxis = "width" | "height";


const query = new URLSearchParams(window.location.search);
const sessionId = query.get("liveSessionId") ?? undefined;
const isRegionReplace = query.get("liveOverlayMode") === "region_replace";
const showSource = query.get("showSource") !== "0";
const showRegionBoxes = ref(query.get("showRegionBoxes") === "1");
const subtitle = ref<LiveSubtitle>();
const state = ref<LiveSessionState>("warming");
const subtitleOverlay = ref<HTMLElement | null>(null);
const subtitlePanel = ref<HTMLElement | null>(null);
const subtitleToolbar = ref<HTMLElement | null>(null);
const sizingEditorOpen = ref(false);
const layoutError = ref("");
const selectionOpening = ref(false);
const toolbarPinned = ref(false);
const overlayDragging = ref(false);
const manualResizeAxis = ref<ManualResizeAxis | null>(null);
const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const overlayWindow = isDesktopRuntime ? getCurrentWindow() : null;
let lastRevision = -1;
let listenersActive = true;
let layoutSyncInFlight = false;
let layoutSyncPending = false;
let layoutAnimationFrame: number | undefined;
let lastLayoutSignature = "";
let panelResizeObserver: ResizeObserver | undefined;
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

const hasLiveSubtitle = computed(
  () => !!subtitle.value?.isStreaming || !!subtitle.value?.translatedText.trim(),
);
const visible = computed(() =>
  isRegionReplace
    ? (subtitle.value?.regions.length ?? 0) > 0
    : state.value === "warming" ||
      state.value === "running" ||
      state.value === "paused" ||
      hasLiveSubtitle.value,
);
const standbyText = computed(() => {
  if (state.value === "warming") {
    return "正在准备模型与窗口捕获…";
  }
  if (state.value === "paused") {
    return "实时翻译已暂停";
  }
  return "模型准备完成，等待翻译";
});
const regionGroups = computed(() =>
  groupLiveSubtitleRegions(subtitle.value?.regions ?? []),
);
const sizingEnabled = computed(() => !isRegionReplace && sessionId !== undefined);
const canOpenRegionSelector = computed(
  () =>
    !!sessionId &&
    !selectionOpening.value &&
    (state.value === "running" || state.value === "paused"),
);
const manualResizeActive = computed(() => manualResizeAxis.value !== null);
const subtitlePanelStyle = computed<CSSProperties>(() => {
  const manualPanelWidth = Math.max(
    1,
    liveOverlaySettings.value.manualWidth / deviceScaleFactor - HORIZONTAL_OVERLAY_PADDING,
  );
  const manualPanelHeight = Math.max(
    1,
    liveOverlaySettings.value.manualHeight / deviceScaleFactor - VERTICAL_OVERLAY_PADDING,
  );
  return {
    width: liveOverlaySettings.value.autoWidth
      ? "max-content"
      : `${manualPanelWidth}px`,
    height: liveOverlaySettings.value.autoHeight
      ? "fit-content"
      : `${manualPanelHeight}px`,
    maxWidth: liveOverlaySettings.value.autoWidth
      ? `${automaticPanelWidthLimit}px`
      : `${manualPanelWidth}px`,
    maxHeight: liveOverlaySettings.value.autoHeight
      ? `${automaticPanelHeightLimit}px`
      : `${manualPanelHeight}px`,
    flex: "0 0 auto",
  };
});

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
  const toolbar = subtitleToolbar.value;
  const toolbarBounds = toolbar?.getBoundingClientRect();
  const toolbarWidth = toolbar
    ? Math.max(toolbar.scrollWidth, toolbarBounds?.width ?? 0)
    : 0;
  const toolbarHeight = toolbar
    ? Math.max(toolbar.scrollHeight, toolbarBounds?.height ?? 0)
    : 0;
  const width = Math.max(bounds.width, panel.scrollWidth);
  const height = Math.max(bounds.height, panel.scrollHeight);
  const scaleFactor = Math.max(window.devicePixelRatio || 1, 1);
  return {
    width: Math.max(1, Math.ceil((width + horizontalPadding) * scaleFactor)),
    height: Math.max(1, Math.ceil((height + verticalPadding) * scaleFactor)),
    minimumWidth: Math.max(
      LIVE_SUBTITLE_MANUAL_WIDTH_MIN,
      Math.ceil((toolbarWidth + 12 + horizontalPadding) * scaleFactor),
    ),
    minimumHeight: Math.max(
      LIVE_SUBTITLE_MANUAL_HEIGHT_MIN,
      Math.ceil((toolbarHeight + 8 + verticalPadding) * scaleFactor),
    ),
  };
}

function scheduleLayoutSync(force = false): void {
  if (!sizingEnabled.value || typeof window === "undefined") {
    return;
  }
  if (force) {
    lastLayoutSignature = "";
  }
  if (manualResizeActive.value) {
    layoutSyncPending = true;
    return;
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
  if (manualResizeActive.value) {
    layoutSyncPending = true;
    return;
  }
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
    contentSize.minimumWidth,
    contentSize.minimumHeight,
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
function toggleToolbarPinned(): void {
  toolbarPinned.value = !toolbarPinned.value;
}


async function startOverlayDrag(event: MouseEvent): Promise<void> {
  if (
    event.button !== 0 ||
    !sessionId ||
    !overlayWindow ||
    overlayDragging.value
  ) {
    return;
  }
  event.preventDefault();
  overlayDragging.value = true;
  try {
    await beginLiveOverlayDrag(sessionId);
    await overlayWindow.startDragging();
    const position = await overlayWindow.outerPosition();
    await updateLiveOverlayPosition(sessionId, {
      x: position.x,
      y: position.y,
    });
    layoutError.value = "";
  } catch (error) {
    layoutError.value = `拖动字幕失败：${errorText(error)}`;
  } finally {
    overlayDragging.value = false;
  }
}

async function openRegionSelector(): Promise<void> {
  if (!sessionId || !canOpenRegionSelector.value) {
    return;
  }
  selectionOpening.value = true;
  try {
    await beginLiveRoiUpdate(sessionId);
    layoutError.value = "";
  } catch (error) {
    layoutError.value = `区域选择失败：${errorText(error)}`;
  } finally {
    selectionOpening.value = false;
  }
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

async function beginManualResize(
  axis: ManualResizeAxis,
  event: PointerEvent,
): Promise<void> {
  if (
    event.button !== 0 ||
    !sessionId ||
    !overlayWindow ||
    manualResizeActive.value
  ) {
    return;
  }
  event.preventDefault();
  manualResizeAxis.value = axis;
  let resizeLocked = false;
  try {
    await beginLiveOverlayResize(sessionId);
    resizeLocked = true;
    await overlayWindow.startResizeDragging(axis === "width" ? "East" : "South");
    const size = await overlayWindow.outerSize();
    const minimum =
      axis === "width"
        ? LIVE_SUBTITLE_MANUAL_WIDTH_MIN
        : LIVE_SUBTITLE_MANUAL_HEIGHT_MIN;
    const maximum =
      axis === "width"
        ? LIVE_SUBTITLE_MANUAL_WIDTH_MAX
        : LIVE_SUBTITLE_MANUAL_HEIGHT_MAX;
    const value = clampManualDimension(
      axis === "width" ? size.width : size.height,
      minimum,
      maximum,
    );
    if (axis === "width") {
      liveOverlaySettings.value.manualWidth = value;
      liveOverlaySettings.value.autoWidth = false;
    } else {
      liveOverlaySettings.value.manualHeight = value;
      liveOverlaySettings.value.autoHeight = false;
    }
    const persistenceError = savePersistedLiveOverlaySettings();
    await nextTick();
    const contentSize = measuredContentSize();
    if (!contentSize) {
      throw new Error("无法读取字幕窗口尺寸");
    }
    await updateLiveOverlayLayout(sessionId, currentSizing(), contentSize);
    lastLayoutSignature = "";
    layoutError.value = persistenceError ?? "";
  } catch (error) {
    layoutError.value = `调整字幕${axis === "width" ? "宽度" : "高度"}失败：${errorText(error)}`;
  } finally {
    if (resizeLocked) {
      try {
        await finishLiveOverlayResize(sessionId);
      } catch (error) {
        const message = `结束字幕拉伸失败：${errorText(error)}`;
        layoutError.value = layoutError.value
          ? `${layoutError.value} ${message}`
          : message;
      }
    }
    manualResizeAxis.value = null;
    scheduleLayoutSync(true);
  }
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
  manualResizeAxis.value = null;
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
        'is-resizing-width': manualResizeAxis === 'width',
        'is-resizing-height': manualResizeAxis === 'height',
        'is-toolbar-pinned': toolbarPinned,
      }"
      :style="subtitlePanelStyle"
    >
      <div
        v-if="sizingEnabled"
        ref="subtitleToolbar"
        class="subtitle-panel-toolbar"
        aria-label="实时字幕工具栏"
      >
        <button
          type="button"
          class="subtitle-tool-button subtitle-toolbar-drag-handle"
          :class="{ 'is-active': overlayDragging }"
          :aria-busy="overlayDragging"
          aria-label="拖动字幕窗口"
          title="拖动字幕窗口"
          @mousedown.left.stop.prevent="startOverlayDrag"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M5 4.5h6M5 8h6M5 11.5h6" />
          </svg>
        </button>
        <button
          type="button"
          class="subtitle-tool-button"
          :class="{ 'is-active': state === 'selecting' }"
          :disabled="!canOpenRegionSelector"
          :aria-busy="selectionOpening"
          aria-label="手动框选翻译区域"
          title="手动框选翻译区域"
          @click="openRegionSelector"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M2.5 6V3.5h2.5M13.5 6V3.5H11M2.5 10v2.5H5M13.5 10v2.5H11" />
          </svg>
        </button>
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
          class="subtitle-tool-button subtitle-width-resize-handle"
          :class="{ 'is-active': manualResizeAxis === 'width' }"
          aria-label="拖动调整字幕窗口宽度"
          title="拖动调整字幕窗口宽度"
          @pointerdown.prevent="beginManualResize('width', $event)"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M2 8h12M4.5 5.5 2 8l2.5 2.5M11.5 5.5 14 8l-2.5 2.5" />
          </svg>
        </button>
        <button
          type="button"
          class="subtitle-tool-button subtitle-height-resize-handle"
          :class="{ 'is-active': manualResizeAxis === 'height' }"
          aria-label="拖动调整字幕窗口高度"
          title="拖动调整字幕窗口高度"
          @pointerdown.prevent="beginManualResize('height', $event)"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M8 2v12M5.5 4.5 8 2l2.5 2.5M5.5 11.5 8 14l2.5-2.5" />
          </svg>
        </button>

        <button
          type="button"
          class="subtitle-tool-button subtitle-pin-button"
          :class="{ 'is-active': toolbarPinned }"
          :aria-pressed="toolbarPinned"
          :aria-label="toolbarPinned ? '取消固定工具栏' : '固定工具栏'"
          :title="toolbarPinned ? '取消固定工具栏' : '固定工具栏'"
          @click="toggleToolbarPinned"
        >
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path d="M5 2.5h6l-.8 3.1 2.3 2.3v1H8.5v4.2L7.5 14V8.9H4.5v-1l2.3-2.3L6 2.5Z" />
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
      </div>
      <p v-if="layoutError" class="subtitle-layout-error" role="status">{{ layoutError }}</p>

      <div class="subtitle-panel-content">
        <p v-if="!hasLiveSubtitle" class="translated-text">{{ standbyText }}</p>
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
  overflow: visible;
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
  position: absolute;
  top: 4px;
  left: 50%;
  z-index: 3;
  display: flex;
  width: max-content;
  max-width: calc(100vw - 12px);
  min-width: 0;
  flex: 0 0 auto;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: flex-start;
  gap: 4px;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 4px 6px;
  border: 1px solid rgba(148, 163, 184, 0.4);
  border-radius: 7px;
  background: rgba(51, 65, 85, 0.97);
  box-shadow: 0 5px 16px rgba(0, 0, 0, 0.28);
  opacity: 0;
  visibility: hidden;
  pointer-events: none;
  transform: translate(-50%, -4px);
  transition: opacity 150ms ease, transform 150ms ease, visibility 0s linear 150ms;
  scrollbar-color: rgba(148, 163, 184, 0.72) transparent;
  scrollbar-width: thin;
}

.subtitle-panel:hover .subtitle-panel-toolbar,
.subtitle-panel:focus-within .subtitle-panel-toolbar,
.subtitle-panel.is-toolbar-pinned .subtitle-panel-toolbar,
.subtitle-panel.is-manual-resizing .subtitle-panel-toolbar {
  opacity: 1;
  visibility: visible;
  pointer-events: auto;
  transform: translate(-50%, 0);
  transition-delay: 0s;
}

.subtitle-panel.is-toolbar-pinned .subtitle-panel-content {
  padding-top: 43px;
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

.subtitle-panel.is-resizing-width {
  cursor: ew-resize;
}

.subtitle-panel.is-resizing-height {
  cursor: ns-resize;
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
  color: rgba(226, 232, 240, 0.9);
  background: rgba(15, 23, 42, 0.94);
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

.subtitle-tool-button:disabled {
  border-color: rgba(226, 232, 240, 0.18);
  color: rgba(226, 232, 240, 0.4);
  background: rgba(15, 23, 42, 0.6);
  cursor: not-allowed;
}

.subtitle-toolbar-drag-handle {
  cursor: move;
  touch-action: none;
}
.subtitle-width-resize-handle {
  cursor: ew-resize;
  touch-action: none;
}

.subtitle-height-resize-handle {
  cursor: ns-resize;
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
  width: auto;
  flex: 0 0 auto;
  flex-wrap: nowrap;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  padding: 0;
  white-space: nowrap;
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
  width: calc(100% - 16px);
  margin: 6px 8px 0;
  padding: 5px 7px;
  border: 1px solid rgba(252, 165, 165, 0.6);
  border-radius: 5px;
  color: #fee2e2;
  background: rgba(127, 29, 29, 0.9);
  font: 500 11px/1.35 "Microsoft YaHei", "Segoe UI", sans-serif;
  text-align: left;
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
