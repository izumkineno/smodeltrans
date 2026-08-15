<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  cancelLiveSelection,
  confirmLiveSelection,
  getLiveSessionStatus,
  listenLiveStatus,
} from "../services/live-translation-provider";
import type { LiveRoi, LiveSessionStatus } from "../services/live-translation-provider";

type Point = { x: number; y: number };
type CssSelection = { x: number; y: number; width: number; height: number };

const MIN_ROI_PHYSICAL_PX = 24;
const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const surface = ref<HTMLElement | null>(null);
const status = ref<LiveSessionStatus | null>(null);
const selectorSessionId = ref<string>();
const dragStart = ref<Point | null>(null);
const dragEnd = ref<Point | null>(null);
const activePointerId = ref<number | null>(null);
const submitting = ref<"confirm" | "cancel" | null>(null);
const errorMessage = ref("");
let statusUnlisten: UnlistenFn | undefined;
let listenerActive = true;

const selection = computed<CssSelection | null>(() => {
  const start = dragStart.value;
  const end = dragEnd.value;
  if (!start || !end) {
    return null;
  }
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
});

const selectionStyle = computed(() => {
  const value = selection.value;
  return value
    ? {
        left: `${value.x}px`,
        top: `${value.y}px`,
        width: `${value.width}px`,
        height: `${value.height}px`,
      }
    : undefined;
});

const physicalRoi = computed<LiveRoi | null>(() => {
  const value = selection.value;
  const target = status.value?.target;
  const element = surface.value;
  if (!value || !target || !element) {
    return null;
  }
  const bounds = element.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0 || target.width <= 0 || target.height <= 0) {
    return null;
  }
  const scaleX = target.width / bounds.width;
  const scaleY = target.height / bounds.height;
  const x = Math.max(0, Math.min(target.width, Math.round(value.x * scaleX)));
  const y = Math.max(0, Math.min(target.height, Math.round(value.y * scaleY)));
  const right = Math.max(x, Math.min(target.width, Math.round((value.x + value.width) * scaleX)));
  const bottom = Math.max(y, Math.min(target.height, Math.round((value.y + value.height) * scaleY)));
  return {
    x,
    y,
    width: right - x,
    height: bottom - y,
    clientWidth: target.width,
    clientHeight: target.height,
  };
});

const selectionLabel = computed(() => {
  const roi = physicalRoi.value;
  return roi ? `${roi.width} × ${roi.height} px` : "拖动鼠标框选字幕区域";
});

const selectionValid = computed(() => {
  const roi = physicalRoi.value;
  return !!roi && roi.width >= MIN_ROI_PHYSICAL_PX && roi.height >= MIN_ROI_PHYSICAL_PX;
});

const modeReady = computed(
  () =>
    isDesktopRuntime &&
    status.value?.state === "selecting" &&
    status.value.sessionId === selectorSessionId.value,
);

function applyStatus(next: LiveSessionStatus): void {
  if (!selectorSessionId.value && next.state === "selecting" && next.sessionId) {
    selectorSessionId.value = next.sessionId;
  }
  if (selectorSessionId.value && next.sessionId && next.sessionId !== selectorSessionId.value) {
    errorMessage.value = "实时会话已变更，此选区窗口不能再提交。";
    return;
  }
  status.value = next;
  if (next.state !== "selecting") {
    errorMessage.value = next.message || "当前会话已离开区域选择模式。";
  }
}

function localPoint(event: PointerEvent): Point | null {
  const element = surface.value;
  if (!element) {
    return null;
  }
  const bounds = element.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(bounds.width, event.clientX - bounds.left)),
    y: Math.max(0, Math.min(bounds.height, event.clientY - bounds.top)),
  };
}

function beginDrag(event: PointerEvent): void {
  if (!modeReady.value || submitting.value || event.button !== 0 || !event.isPrimary) {
    return;
  }
  if ((event.target as HTMLElement | null)?.closest(".selector-controls")) {
    return;
  }
  const point = localPoint(event);
  if (!point) {
    return;
  }
  errorMessage.value = "";
  dragStart.value = point;
  dragEnd.value = point;
  activePointerId.value = event.pointerId;
  surface.value?.setPointerCapture(event.pointerId);
  event.preventDefault();
}

function updateDrag(event: PointerEvent): void {
  if (activePointerId.value !== event.pointerId) {
    return;
  }
  const point = localPoint(event);
  if (point) {
    dragEnd.value = point;
  }
}

function endDrag(event: PointerEvent): void {
  if (activePointerId.value !== event.pointerId) {
    return;
  }
  updateDrag(event);
  if (surface.value?.hasPointerCapture(event.pointerId)) {
    surface.value.releasePointerCapture(event.pointerId);
  }
  activePointerId.value = null;
  if (!selectionValid.value) {
    errorMessage.value = `选区宽度和高度至少需要 ${MIN_ROI_PHYSICAL_PX} 个物理像素。`;
  }
}

async function confirmSelection(): Promise<void> {
  const sessionId = selectorSessionId.value;
  const roi = physicalRoi.value;
  if (!modeReady.value || !sessionId || !roi || !selectionValid.value || submitting.value) {
    if (!selectionValid.value) {
      errorMessage.value = `请框选至少 ${MIN_ROI_PHYSICAL_PX} × ${MIN_ROI_PHYSICAL_PX} 物理像素的字幕区域。`;
    }
    return;
  }
  submitting.value = "confirm";
  try {
    applyStatus(await confirmLiveSelection(sessionId, roi));
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    submitting.value = null;
  }
}

async function cancelSelection(): Promise<void> {
  const sessionId = selectorSessionId.value;
  if (!modeReady.value || !sessionId || submitting.value) {
    return;
  }
  submitting.value = "cancel";
  try {
    applyStatus(await cancelLiveSelection(sessionId));
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    submitting.value = null;
  }
}

function handleKeydown(event: KeyboardEvent): void {
  if (event.key === "Escape") {
    event.preventDefault();
    void cancelSelection();
  } else if (event.key === "Enter" && selectionValid.value) {
    event.preventDefault();
    void confirmSelection();
  }
}

async function initializeSelector(): Promise<void> {
  if (!isDesktopRuntime) {
    errorMessage.value = "区域选择器仅在 Windows Tauri 桌面端运行。";
    return;
  }
  try {
    const unlisten = await listenLiveStatus(applyStatus);
    if (!listenerActive) {
      unlisten();
      return;
    }
    statusUnlisten = unlisten;
    applyStatus(await getLiveSessionStatus());
    if (!modeReady.value && !errorMessage.value) {
      errorMessage.value = "当前没有等待确认的实时字幕选区。";
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

onMounted(() => {
  window.addEventListener("keydown", handleKeydown);
  void initializeSelector();
});

onBeforeUnmount(() => {
  listenerActive = false;
  statusUnlisten?.();
  statusUnlisten = undefined;
  window.removeEventListener("keydown", handleKeydown);
});
</script>

<template>
  <main
    ref="surface"
    class="selection-surface"
    :class="{ 'selection-surface-disabled': !modeReady }"
    aria-label="实时字幕区域选择器"
    @pointerdown="beginDrag"
    @pointermove="updateDrag"
    @pointerup="endDrag"
    @pointercancel="endDrag"
  >
    <div class="screen-dim" aria-hidden="true"></div>
    <div v-if="selection" class="selection-box" :style="selectionStyle">
      <span class="selection-size">{{ selectionLabel }}</span>
      <i class="corner corner-tl"></i>
      <i class="corner corner-tr"></i>
      <i class="corner corner-bl"></i>
      <i class="corner corner-br"></i>
    </div>

    <section class="selector-controls" aria-live="polite">
      <div class="selector-copy">
        <strong>框选字幕区域</strong>
        <span>{{ status?.target?.title || "正在读取目标窗口…" }}</span>
        <p class="selector-guidance">首次使用：按住鼠标左键拖动框选字幕区域；完成后点击“确认选区”，不需要时点击“取消”。</p>
      </div>
      <div class="selector-state">
        <span>{{ selectionLabel }}</span>
        <span v-if="errorMessage" class="selector-error">{{ errorMessage }}</span>
      </div>
      <div class="selector-actions">
        <button type="button" :disabled="!modeReady || submitting !== null" @click="cancelSelection">
          {{ submitting === "cancel" ? "正在取消…" : "取消 (Esc)" }}
        </button>
        <button
          class="confirm-button"
          type="button"
          :disabled="!modeReady || !selectionValid || submitting !== null"
          @click="confirmSelection"
        >
          {{ submitting === "confirm" ? "正在确认…" : "确认选区 (Enter)" }}
        </button>
      </div>
    </section>
  </main>
</template>

<style>
:root {
  color: #ffffff;
  background: transparent;
  font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif;
}

* { box-sizing: border-box; }
html, body, #app { width: 100%; height: 100%; margin: 0; overflow: hidden; background: transparent; }
button { font: inherit; }

.selection-surface {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: crosshair;
  touch-action: none;
  user-select: none;
}

.selection-surface-disabled { cursor: not-allowed; }
.screen-dim { position: absolute; inset: 0; background: rgba(3, 8, 18, 0.34); }

.selection-box {
  position: absolute;
  z-index: 2;
  border: 2px solid #48a7ff;
  background: rgba(64, 158, 255, 0.08);
  box-shadow: 0 0 0 9999px rgba(3, 8, 18, 0.32), 0 0 18px rgba(64, 158, 255, 0.55);
  pointer-events: none;
}

.selection-size {
  position: absolute;
  bottom: calc(100% + 7px);
  left: -2px;
  padding: 4px 7px;
  border-radius: 3px;
  color: #ffffff;
  background: #1677c8;
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.corner { position: absolute; width: 12px; height: 12px; border-color: #ffffff; border-style: solid; }
.corner-tl { top: -3px; left: -3px; border-width: 3px 0 0 3px; }
.corner-tr { top: -3px; right: -3px; border-width: 3px 3px 0 0; }
.corner-bl { bottom: -3px; left: -3px; border-width: 0 0 3px 3px; }
.corner-br { right: -3px; bottom: -3px; border-width: 0 3px 3px 0; }

.selector-controls {
  position: absolute;
  z-index: 4;
  top: 18px;
  left: 50%;
  display: grid;
  width: min(780px, calc(100% - 36px));
  grid-template-columns: minmax(230px, 1fr) auto auto;
  align-items: center;
  gap: 18px;
  padding: 13px 14px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 6px;
  background: rgba(15, 23, 42, 0.93);
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.38);
  cursor: default;
  transform: translateX(-50%);
  backdrop-filter: blur(10px);
}

.selector-copy { display: flex; min-width: 0; flex-direction: column; gap: 3px; }
.selector-copy strong { font-size: 14px; }
.selector-copy span { overflow: hidden; color: #cbd5e1; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.selector-copy p { margin: 2px 0 0; color: #94a3b8; font-size: 11px; }
.selector-state { display: flex; max-width: 220px; flex-direction: column; gap: 4px; color: #dbeafe; font-size: 11px; text-align: right; }
.selector-error { color: #fca5a5; }
.selector-actions { display: flex; gap: 8px; }

.selector-actions button {
  min-height: 34px;
  padding: 0 12px;
  border: 1px solid #475569;
  border-radius: 4px;
  color: #e2e8f0;
  background: #1e293b;
  cursor: pointer;
}

.selector-actions button:hover:not(:disabled) { background: #334155; }
.selector-actions .confirm-button { border-color: #409eff; color: #ffffff; background: #1677c8; }
.selector-actions .confirm-button:hover:not(:disabled) { background: #409eff; }
.selector-actions button:disabled { cursor: not-allowed; opacity: 0.48; }
.selector-actions button:focus-visible { outline: 2px solid #93c5fd; outline-offset: 2px; }

@media (max-width: 720px) {
  .selector-controls { grid-template-columns: 1fr; gap: 9px; }
  .selector-state { max-width: none; text-align: left; }
  .selector-actions { justify-content: flex-end; }
}
</style>
