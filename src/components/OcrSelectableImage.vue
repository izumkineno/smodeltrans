<script setup lang="ts">
import {
  computed,
  onBeforeUnmount,
  onDeactivated,
  onMounted,
  ref,
  watch,
} from "vue";
import { NButton } from "naive-ui";
import type { OcrRegion } from "../services/translation-provider";
import {
  buildOcrSelectionModel,
  resolveOcrSelectionRange,
} from "../services/ocr-selection";

const props = defineProps<{
  title: string;
  src: string;
  alt: string;
  imageWidth: number;
  imageHeight: number;
  regions: OcrRegion[];
  stateLabel?: string;
  previewMode?: boolean;
}>();

const emit = defineEmits<{
  (event: "selection-change", value: string): void;
  (event: "close"): void;
}>();

const overlay = ref<SVGSVGElement | null>(null);
const selectionSource = ref<HTMLElement | null>(null);
const selectionAnchor = ref<number | null>(null);
const selectionFocus = ref<number | null>(null);
let pointerSelecting = false;
const zoom = ref(1);
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 4;
const ZOOM_STEP = 0.25;

const selectionModel = computed(() => buildOcrSelectionModel(props.regions));
const activeRange = computed(() => {
  if (selectionAnchor.value === null || selectionFocus.value === null) {
    return null;
  }
  return resolveOcrSelectionRange(
    selectionModel.value,
    selectionAnchor.value,
    selectionFocus.value,
  );
});
const selectedCount = computed(() => {
  const range = activeRange.value;
  return range ? range.lastIndex - range.firstIndex + 1 : 0;
});
const zoomPercent = computed(() => `${Math.round(zoom.value * 100)}%`);
const viewBox = computed(() => `0 0 ${props.imageWidth} ${props.imageHeight}`);

function polygonPoints(quad: OcrRegion["quad"]): string {
  return quad.map(([x, y]) => `${x},${y}`).join(" ");
}

function clearNativeSelection(): void {
  const source = selectionSource.value;
  const selection = window.getSelection();
  if (source && selection?.anchorNode && source.contains(selection.anchorNode)) {
    selection.removeAllRanges();
  }
}

function syncNativeSelection(): void {
  const range = activeRange.value;
  const source = selectionSource.value;
  const textNode = source?.firstChild;
  if (!range || !textNode) {
    clearNativeSelection();
    emit("selection-change", "");
    return;
  }
  const nativeRange = document.createRange();
  nativeRange.setStart(textNode, range.start);
  nativeRange.setEnd(textNode, range.end);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(nativeRange);
  emit("selection-change", range.text);
}

function setSelection(anchorIndex: number, focusIndex: number): void {
  selectionAnchor.value = anchorIndex;
  selectionFocus.value = focusIndex;
  syncNativeSelection();
}

function beginPointerSelection(index: number, event: PointerEvent): void {
  if (event.button !== 0) {
    return;
  }
  event.preventDefault();
  pointerSelecting = true;
  overlay.value?.focus({ preventScroll: true });
  setSelection(index, index);
}

function extendPointerSelection(index: number): void {
  if (pointerSelecting && selectionAnchor.value !== null) {
    setSelection(selectionAnchor.value, index);
  }
}

function endPointerSelection(): void {
  pointerSelecting = false;
}

function handleKeydown(event: KeyboardEvent): void {
  const count = selectionModel.value.characters.length;
  if (count === 0) {
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
    event.preventDefault();
    setSelection(0, count - 1);
    return;
  }

  const current = selectionFocus.value ?? 0;
  let next = current;
  if (event.key === "ArrowLeft") {
    next = Math.max(0, current - 1);
  } else if (event.key === "ArrowRight") {
    next = Math.min(count - 1, current + 1);
  } else if (event.key === "Home") {
    next = 0;
  } else if (event.key === "End") {
    next = count - 1;
  } else {
    return;
  }
  event.preventDefault();
  const anchor = event.shiftKey ? (selectionAnchor.value ?? current) : next;
  setSelection(anchor, next);
}

function clearSelection(): void {
  pointerSelecting = false;
  selectionAnchor.value = null;
  selectionFocus.value = null;
  clearNativeSelection();
  emit("selection-change", "");
}

function setZoom(nextZoom: number): void {
  zoom.value = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom));
}

function zoomIn(): void {
  setZoom(zoom.value + ZOOM_STEP);
}

function zoomOut(): void {
  setZoom(zoom.value - ZOOM_STEP);
}

function resetZoom(): void {
  zoom.value = 1;
}

function handlePreviewWheel(event: WheelEvent): void {
  if (!props.previewMode) {
    return;
  }
  setZoom(zoom.value + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP));
}

watch(selectionModel, clearSelection);
watch(() => props.src, resetZoom);
onMounted(() => window.addEventListener("pointerup", endPointerSelection));
onDeactivated(clearSelection);
onBeforeUnmount(() => {
  window.removeEventListener("pointerup", endPointerSelection);
  clearSelection();
});
</script>

<template>
  <div
    class="selectable-image-frame"
    :class="{ 'is-preview': previewMode }"
    :role="previewMode ? 'dialog' : undefined"
    :aria-modal="previewMode ? 'true' : undefined"
    :aria-label="previewMode ? `${title}放大预览` : undefined"
  >
    <div class="selectable-image-bar">
      <span>{{ title }}</span>
      <span class="selectable-image-actions">
        <span class="selectable-image-state">
          {{ selectedCount > 0 ? `已选 ${selectedCount} 字` : (stateLabel ?? "可选择") }}
        </span>
        <slot name="actions" />
        <template v-if="previewMode">
          <n-button quaternary size="small" aria-label="缩小图片" :disabled="zoom <= MIN_ZOOM" @click="zoomOut">
            缩小
          </n-button>
          <n-button quaternary size="small" aria-label="恢复图片缩放" @click="resetZoom">
            {{ zoomPercent }}
          </n-button>
          <n-button quaternary size="small" aria-label="放大图片" :disabled="zoom >= MAX_ZOOM" @click="zoomIn">
            放大
          </n-button>
          <n-button quaternary size="small" aria-label="关闭图片预览" @click="emit('close')">
            关闭
          </n-button>
        </template>
      </span>
    </div>
    <div class="selectable-image-canvas" @wheel.prevent="handlePreviewWheel">
      <div class="selectable-image-stage" :style="{ transform: `scale(${zoom})` }">
        <img class="selectable-image" :src="src" :alt="alt" draggable="false" />
        <svg
          ref="overlay"
          class="selection-overlay"
          :viewBox="viewBox"
          role="group"
          tabindex="0"
          aria-label="OCR 文本选择层。拖动选择文字，或使用左右方向键移动，按住 Shift 扩展选择。"
          @keydown="handleKeydown"
          @dragstart.prevent
        >
          <title>可选择的 OCR 文字</title>
          <polygon
            v-for="(character, index) in selectionModel.characters"
            :key="character.key"
            class="selection-character"
            :class="{ 'is-selected': activeRange && index >= activeRange.firstIndex && index <= activeRange.lastIndex }"
            :points="polygonPoints(character.quad)"
            :aria-label="character.text"
            @pointerdown="beginPointerSelection(index, $event)"
            @pointerenter="extendPointerSelection(index)"
          />
        </svg>
      </div>
    </div>
    <pre ref="selectionSource" class="native-selection-source" aria-hidden="true">{{ selectionModel.text }}</pre>
  </div>
</template>

<style scoped>
.selectable-image-frame {
  display: flex;
  min-height: 150px;
  max-height: 340px;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface);
}

.selectable-image-frame.is-preview {
  width: min(94vw, 1440px);
  height: min(92vh, 960px);
  min-height: 0;
  max-height: none;
  border-color: rgba(255, 255, 255, 0.16);
  border-radius: 8px;
  background: #202124;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.42);
}

.selectable-image-bar {
  display: flex;
  min-height: 36px;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 10px;
  border-bottom: 1px solid var(--divider);
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 600;
}

.is-preview .selectable-image-bar {
  min-height: 48px;
  border-bottom-color: rgba(255, 255, 255, 0.12);
  color: #f5f7fa;
  background: #25272b;
}

.selectable-image-actions {
  display: inline-flex;
  min-width: 0;
  align-items: center;
  gap: 8px;
}

.selectable-image-state {
  color: var(--text-muted);
  font-weight: 500;
  white-space: nowrap;
}

.is-preview .selectable-image-state {
  color: #c0c4cc;
}

.is-preview .selectable-image-actions :deep(.n-button) {
  color: #e5e7eb;
}

.selectable-image-canvas {
  display: flex;
  min-height: 0;
  flex: 1;
  align-items: center;
  justify-content: center;
  overflow: auto;
  padding: 12px;
  background: #f3f5f8;
}

.is-preview .selectable-image-canvas {
  padding: 32px;
  background: #17191c;
}

.selectable-image-stage {
  position: relative;
  display: inline-block;
  max-width: 100%;
  max-height: 280px;
  line-height: 0;
}

.is-preview .selectable-image-stage {
  max-width: calc(94vw - 80px);
  max-height: calc(92vh - 112px);
  transform-origin: center;
  transition: transform 140ms ease;
}

.selectable-image {
  display: block;
  width: auto;
  max-width: 100%;
  height: auto;
  max-height: 280px;
  user-select: none;
}

.is-preview .selectable-image {
  max-width: calc(94vw - 80px);
  max-height: calc(92vh - 112px);
}

.selection-overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
  touch-action: none;
  user-select: none;
}

.selection-overlay:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.selection-character {
  fill: rgba(64, 158, 255, 0.001);
  stroke: transparent;
  stroke-width: 1.5;
  vector-effect: non-scaling-stroke;
  cursor: text;
  pointer-events: all;
  transition: fill 120ms ease, stroke 120ms ease;
}

.selection-character:hover {
  fill: rgba(64, 158, 255, 0.13);
  stroke: rgba(64, 158, 255, 0.65);
}

.selection-character.is-selected {
  fill: rgba(64, 158, 255, 0.3);
  stroke: var(--primary);
}

.native-selection-source {
  position: fixed;
  width: 1px;
  height: 1px;
  margin: 0;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: pre;
}
@media (max-width: 720px) {
  .selectable-image-frame.is-preview {
    width: 100vw;
    height: 100vh;
    border: 0;
    border-radius: 0;
  }

  .is-preview .selectable-image-bar {
    align-items: flex-start;
    flex-direction: column;
  }

  .selectable-image-actions {
    width: 100%;
    flex-wrap: wrap;
  }

  .is-preview .selectable-image-canvas {
    padding: 16px;
  }

  .is-preview .selectable-image-stage,
  .is-preview .selectable-image {
    max-width: calc(100vw - 32px);
    max-height: calc(100vh - 144px);
  }
}
</style>
