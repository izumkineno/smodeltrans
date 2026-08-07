<script setup lang="ts">
import { computed, type VNodeChild } from "vue";
import { NImage, type ImageRenderToolbarProps } from "naive-ui";

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
  }>(),
  {
    variant: "input",
  },
);

const emit = defineEmits<{
  (event: "error"): void;
}>();

const frameClass = computed(() => (props.variant === "result" ? "result-preview-frame" : "image-preview-frame"));
const canvasClass = computed(() => (props.variant === "result" ? "result-preview-canvas" : "preview-canvas"));
const imageClass = computed(() => (props.variant === "result" ? "result-image" : "input-image"));
const previewSource = computed(() => props.previewSrc ?? props.src ?? undefined);
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
    <div :class="canvasClass">
      <n-image
        :class="imageClass"
        :src="src ?? undefined"
        :preview-src="previewSource"
        :alt="alt"
        object-fit="contain"
        show-toolbar
        show-toolbar-tooltip
        :render-toolbar="renderToolbar"
        @error="emit('error')"
      />
    </div>
  </div>
</template>
