<script setup lang="ts">
import { computed, ref, type VNodeChild } from "vue";
import { NImage, NModal, type ImageRenderToolbarProps } from "naive-ui";
import type { OcrRegion } from "../services/translation-provider";
import OcrSelectableImage from "./OcrSelectableImage.vue";

type PreviewVariant = "input" | "result";

const props = withDefaults(
  defineProps<{
    title: string;
    src?: string | null;
    previewSrc?: string | null;
    alt: string;
    variant?: PreviewVariant;
    stateLabel?: string;
    renderToolbar: (props: ImageRenderToolbarProps) => VNodeChild;
    imageWidth?: number;
    imageHeight?: number;
    regions?: OcrRegion[];
  }>(),
  {
    variant: "input",
    imageWidth: 0,
    imageHeight: 0,
    regions: () => [],
  },
);

const emit = defineEmits<{
  (event: "error"): void;
  (event: "selection-change", value: string): void;
}>();

const selectablePreviewOpen = ref(false);
const frameClass = computed(() => (props.variant === "result" ? "result-preview-frame" : "image-preview-frame"));
const canvasClass = computed(() => (props.variant === "result" ? "result-preview-canvas" : "preview-canvas"));
const imageClass = computed(() => (props.variant === "result" ? "result-image" : "input-image"));
const previewSource = computed(() => props.previewSrc ?? props.src ?? undefined);
const hasSelectablePreview = computed(
  () =>
    props.imageWidth > 0 &&
    props.imageHeight > 0 &&
    props.regions.length > 0 &&
    previewSource.value !== undefined,
);

function openSelectablePreview(): void {
  if (hasSelectablePreview.value) {
    selectablePreviewOpen.value = true;
  }
}

function handlePreviewKeydown(event: KeyboardEvent): void {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openSelectablePreview();
  }
}

function closeSelectablePreview(): void {
  selectablePreviewOpen.value = false;
}
</script>

<template>
  <div :class="frameClass">
    <div class="preview-frame-bar">
      <span>{{ title }}</span>
      <div v-if="stateLabel || $slots.actions" class="preview-frame-actions">
        <span v-if="stateLabel" class="preview-frame-state">
          <span class="provider-indicator" aria-hidden="true"></span>
          {{ stateLabel }}
        </span>
        <slot name="actions" />
      </div>
    </div>
    <div
      :class="[canvasClass, { 'selectable-preview-trigger': hasSelectablePreview }]"
      :role="hasSelectablePreview ? 'button' : undefined"
      :tabindex="hasSelectablePreview ? 0 : undefined"
      :aria-label="hasSelectablePreview ? `放大${title}并选择 OCR 文字` : undefined"
      @click="openSelectablePreview"
      @keydown="handlePreviewKeydown"
    >
      <n-image
        :class="imageClass"
        :src="src ?? undefined"
        :preview-src="previewSource"
        :preview-disabled="hasSelectablePreview"
        :alt="alt"
        object-fit="contain"
        show-toolbar
        show-toolbar-tooltip
        :render-toolbar="renderToolbar"
        @error="emit('error')"
      />
    </div>
  </div>

  <n-modal
    v-model:show="selectablePreviewOpen"
    display-directive="if"
    :auto-focus="false"
    :mask-closable="true"
  >
    <OcrSelectableImage
      v-if="previewSource"
      preview-mode
      :title="title"
      :src="previewSource"
      :alt="alt"
      :image-width="imageWidth"
      :image-height="imageHeight"
      :regions="regions"
      :state-label="stateLabel"
      @close="closeSelectablePreview"
      @selection-change="emit('selection-change', $event)"
    >
      <template #actions>
        <slot name="actions" />
      </template>
    </OcrSelectableImage>
  </n-modal>
</template>

<style scoped>
.selectable-preview-trigger {
  cursor: zoom-in;
}

.selectable-preview-trigger:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: -2px;
}
</style>
