<script setup lang="ts">
import {
  computed,
  onBeforeUnmount,
  onDeactivated,
  onMounted,
  ref,
  watch,
} from "vue";
import { NButton, NScrollbar } from "naive-ui";
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
  event.preventDefault();
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
    <n-scrollbar
      class="selectable-image-canvas"
      x-scrollable
      content-class="selectable-image-scroll-content"
      @wheel="handlePreviewWheel"
    >
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
    </n-scrollbar>
    <pre ref="selectionSource" class="native-selection-source" aria-hidden="true">{{ selectionModel.text }}</pre>
  </div>
</template>

<style scoped src="../styles/ocr-selectable-image.css"></style>
