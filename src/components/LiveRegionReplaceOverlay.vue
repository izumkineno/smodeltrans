<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { ComponentPublicInstance } from "vue";
import {
  mapLiveSubtitleRegionToOverlay,
} from "../services/live-translation-provider";
import type {
  LiveSessionState,
  LiveSubtitle,
  LiveSubtitleRegion,
} from "../services/live-translation-provider";
import type { LiveSubtitleStyleSettings } from "../services/workspace-settings";
import { liveSubtitleBackgroundRgba } from "../services/workspace-settings";

const props = defineProps<{
  state: LiveSessionState;
  subtitle?: LiveSubtitle;
  showSource: boolean;
  showRegionBoxes: boolean;
  styleSettings: LiveSubtitleStyleSettings;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
let resizeObserver: ResizeObserver | undefined;

const regions = computed(() => props.subtitle?.regions ?? []);
const showPlaceholder = computed(
  () =>
    (props.state === "running" || props.state === "paused") &&
    regions.value.length === 0,
);
const visible = computed(() => regions.value.length > 0 || showPlaceholder.value);
const standbyText = computed(() =>
  props.state === "paused" ? "实时翻译已暂停" : "模型准备完成，等待翻译",
);

function setCanvas(element: Element | ComponentPublicInstance | null): void {
  const nextCanvas = element instanceof HTMLCanvasElement ? element : null;
  if (canvas.value === nextCanvas) {
    return;
  }
  resizeObserver?.disconnect();
  canvas.value = nextCanvas;
  if (nextCanvas && resizeObserver) {
    resizeObserver.observe(nextCanvas);
  }
  void nextTick().then(draw);
}

function wrapText(
  context: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
): string[] {
  const lines: string[] = [];
  let line = "";
  for (const character of Array.from(text.trim())) {
    if (character === "\n") {
      if (line) {
        lines.push(line);
        line = "";
      }
      continue;
    }
    const candidate = line + character;
    if (line && context.measureText(candidate).width > maxWidth) {
      lines.push(line);
      line = character;
    } else {
      line = candidate;
    }
  }
  if (line) {
    lines.push(line);
  }
  return lines;
}

function drawSubtitleBackground(
  context: CanvasRenderingContext2D,
  rect: { x: number; y: number; width: number; height: number },
  styleSettings: LiveSubtitleStyleSettings,
): void {
  context.fillStyle = liveSubtitleBackgroundRgba(
    styleSettings.backgroundColor,
    styleSettings.backgroundOpacity,
  );
  context.fillRect(rect.x, rect.y, rect.width, rect.height);
  context.strokeStyle = "rgba(148, 163, 184, 0.52)";
  context.lineWidth = 1;
  context.strokeRect(rect.x, rect.y, rect.width, rect.height);
}

function drawRegionText(
  context: CanvasRenderingContext2D,
  region: LiveSubtitleRegion,
  rect: { x: number; y: number; width: number; height: number },
  styleSettings: LiveSubtitleStyleSettings,
): void {
  const translated = region.translatedText.trim();
  const source = props.showSource ? region.sourceText.trim() : "";
  if (!translated && !source) {
    return;
  }
  drawSubtitleBackground(context, rect, styleSettings);

  const padding = Math.max(3, Math.min(rect.width, rect.height) * 0.12);
  const maxWidth = Math.max(1, rect.width - padding * 2);
  const lineCount = Math.max(1, translated.length + (source ? source.length : 0));
  const fontSize = Math.max(
    8,
    Math.min(
      styleSettings.fontSize,
      rect.height * 0.72,
      rect.width / Math.max(lineCount * 0.55, 1),
    ),
  );
  const lineHeight = fontSize * 1.18;
  const lines: Array<{ text: string; size: number; color: string }> = [];

  if (source) {
    context.font = `400 ${Math.max(8, fontSize * 0.62)}px "Microsoft YaHei", "Segoe UI", sans-serif`;
    lines.push(
      ...wrapText(context, source, maxWidth).map((text) => ({
        text,
        size: Math.max(8, fontSize * 0.62),
        color: "#475569",
      })),
    );
  }
  if (translated) {
    context.font = `600 ${fontSize}px "Microsoft YaHei", "Segoe UI", sans-serif`;
    lines.push(
      ...wrapText(context, translated, maxWidth).map((text) => ({
        text,
        size: fontSize,
        color: styleSettings.fontColor,
      })),
    );
  }

  const totalHeight = lines.length * lineHeight;
  let y = rect.y + Math.max(fontSize / 2, (rect.height - totalHeight) / 2 + lineHeight / 2);
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.shadowColor = "rgba(255, 255, 255, 0.72)";
  context.shadowBlur = 2;
  context.shadowOffsetY = 1;
  for (const line of lines) {
    context.font = `600 ${line.size}px "Microsoft YaHei", "Segoe UI", sans-serif`;
    context.fillStyle = line.color;
    context.fillText(line.text, rect.x + rect.width / 2, y, maxWidth);
    y += lineHeight;
  }
  context.shadowColor = "transparent";
  context.shadowBlur = 0;
  context.shadowOffsetY = 0;
}

function drawPlaceholder(
  context: CanvasRenderingContext2D,
  width: number,
  height: number,
  styleSettings: LiveSubtitleStyleSettings,
): void {
  const boxWidth = Math.min(width * 0.8, 640);
  const boxHeight = Math.max(48, Math.min(96, height * 0.08));
  const x = (width - boxWidth) / 2;
  const y = height - boxHeight - height * 0.1;
  context.fillStyle = liveSubtitleBackgroundRgba(
    styleSettings.backgroundColor,
    styleSettings.backgroundOpacity,
  );
  context.fillRect(x, y, boxWidth, boxHeight);
  context.strokeStyle = "rgba(148, 163, 184, 0.52)";
  context.lineWidth = 1;
  context.strokeRect(x, y, boxWidth, boxHeight);
  context.font = `600 ${Math.max(12, Math.min(styleSettings.fontSize, boxHeight * 0.3))}px "Microsoft YaHei", "Segoe UI", sans-serif`;
  context.fillStyle = styleSettings.fontColor;
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(standbyText.value, width / 2, y + boxHeight / 2, boxWidth - 24);
}

function draw(): void {
  const element = canvas.value;
  if (!element || !visible.value) {
    return;
  }
  const width = element.clientWidth;
  const height = element.clientHeight;
  if (width <= 0 || height <= 0) {
    return;
  }
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  element.width = Math.max(1, Math.round(width * dpr));
  element.height = Math.max(1, Math.round(height * dpr));
  const context = element.getContext("2d");
  if (!context) {
    return;
  }
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);

  if (showPlaceholder.value) {
    drawPlaceholder(context, width, height, props.styleSettings);
    return;
  }
  const roi = props.subtitle?.roi;
  if (!roi) {
    return;
  }
  for (const [index, region] of regions.value.entries()) {
    const rect = mapLiveSubtitleRegionToOverlay(region, roi, width, height);
    if (!rect) {
      continue;
    }
    if (props.showRegionBoxes) {
      context.strokeStyle = "#00e5ff";
      context.lineWidth = 2;
      context.strokeRect(rect.x, rect.y, rect.width, rect.height);
      context.font = "700 10px Segoe UI, sans-serif";
      context.fillStyle = "#00e5ff";
      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillText(`${index + 1} · ${region.bounds.left},${region.bounds.top}`, rect.x + 2, rect.y + 2);
    }
    drawRegionText(context, region, rect, props.styleSettings);
  }
}

onMounted(() => {
  resizeObserver = new ResizeObserver(() => draw());
  if (canvas.value) {
    resizeObserver.observe(canvas.value);
  }
  draw();
});

watch(
  () => [
    props.subtitle,
    props.state,
    props.showSource,
    props.showRegionBoxes,
    props.styleSettings,
    visible.value,
  ],
  () => void nextTick().then(draw),
  { deep: true },
);

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
});
</script>

<template>
  <div v-if="visible" class="region-replace-layer">
    <canvas
      :ref="setCanvas"
      class="region-replace-canvas"
      role="img"
      :aria-label="showPlaceholder ? standbyText : '逐区域翻译字幕'"
    />
  </div>
</template>

<style src="../styles/live-region-replace.css"></style>
