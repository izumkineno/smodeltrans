import { ref } from "vue";
import { getBackendStatus, getModelRuntimeStatus } from "./translation-provider";
import type { BackendStatus, ModelRuntimeStatus } from "./translation-provider";
import type {
  LiveOverlaySettings,
  LiveRecognitionSettings,
  LiveTranslationSettings,
} from "./live-translation-provider";

import {
  DEFAULT_LIVE_MEMORY_ENABLED,
  DEFAULT_LIVE_MEMORY_TOKENS,
  DEFAULT_LIVE_MEMORY_TURNS,
  LIVE_MEMORY_TOKENS_MAX,
  LIVE_MEMORY_TOKENS_MIN,
  LIVE_MEMORY_TURNS_MAX,
  LIVE_MEMORY_TURNS_MIN,
  LIVE_SUPPLEMENTAL_PROMPT_MAX_CHARS,
} from "./live-translation-provider";

const TARGET_LANGUAGE_STORAGE_KEY = "smodeltrans.targetLanguage";
const LIVE_OVERLAY_SETTINGS_STORAGE_KEY = "smodeltrans.liveOverlaySettings";
const LIVE_SUBTITLE_STYLE_SETTINGS_STORAGE_KEY = "smodeltrans.liveSubtitleStyleSettings";
const LIVE_RECOGNITION_SETTINGS_STORAGE_KEY = "smodeltrans.liveRecognitionSettings";
const LIVE_TRANSLATION_SETTINGS_STORAGE_KEY = "smodeltrans.liveTranslationSettings";
export const DEFAULT_LIVE_STABILITY_WAIT_MS = 300;
export const LIVE_STABILITY_WAIT_MIN_MS = 0;
export const LIVE_STABILITY_WAIT_MAX_MS = 5_000;
export const DEFAULT_KEY_TRIGGER_TIMEOUT_MS = 1_000;
export const KEY_TRIGGER_TIMEOUT_MIN_MS = 100;
export const KEY_TRIGGER_TIMEOUT_MAX_MS = 5_000;
export const DEFAULT_LIVE_SUBTITLE_MANUAL_WIDTH = 960;
export const LIVE_SUBTITLE_MANUAL_WIDTH_MIN = 160;
export const LIVE_SUBTITLE_MANUAL_WIDTH_MAX = 8_192;
export const DEFAULT_LIVE_SUBTITLE_MANUAL_HEIGHT = 168;
export const LIVE_SUBTITLE_MANUAL_HEIGHT_MIN = 72;
export const LIVE_SUBTITLE_MANUAL_HEIGHT_MAX = 4_096;
export const DEFAULT_LIVE_SUBTITLE_FONT_COLOR = "#0f172a";
export const LIVE_SUBTITLE_FONT_SIZE_MIN = 12;
export const LIVE_SUBTITLE_FONT_SIZE_MAX = 64;
export const DEFAULT_LIVE_SUBTITLE_FONT_SIZE = 24;
export const DEFAULT_LIVE_SUBTITLE_BACKGROUND_COLOR = "#e2e8f0";
export const LIVE_SUBTITLE_BACKGROUND_OPACITY_MIN = 0;
export const LIVE_SUBTITLE_BACKGROUND_OPACITY_MAX = 100;
export const DEFAULT_LIVE_SUBTITLE_BACKGROUND_OPACITY = 86;

export interface LiveSubtitleStyleSettings {
  fontColor: string;
  fontSize: number;
  backgroundColor: string;
  backgroundOpacity: number;
}

function createDefaultLiveSubtitleStyleSettings(): LiveSubtitleStyleSettings {
  return {
    fontColor: DEFAULT_LIVE_SUBTITLE_FONT_COLOR,
    fontSize: DEFAULT_LIVE_SUBTITLE_FONT_SIZE,
    backgroundColor: DEFAULT_LIVE_SUBTITLE_BACKGROUND_COLOR,
    backgroundOpacity: DEFAULT_LIVE_SUBTITLE_BACKGROUND_OPACITY,
  };
}

export function liveSubtitleBackgroundRgba(
  color: string,
  opacity: number,
): string {
  const match = /^#([0-9a-f]{6})$/i.exec(color);
  if (!match) {
    return `rgba(226, 232, 240, ${Math.min(100, Math.max(0, opacity)) / 100})`;
  }
  const red = Number.parseInt(match[1].slice(0, 2), 16);
  const green = Number.parseInt(match[1].slice(2, 4), 16);
  const blue = Number.parseInt(match[1].slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${Math.min(100, Math.max(0, opacity)) / 100})`;
}

function normalizePersistedLiveSubtitleColor(value: unknown): string | null {
  if (typeof value !== "string" || !/^#[0-9a-f]{6}$/i.test(value.trim())) {
    return null;
  }
  return value.trim().toLowerCase();
}

function normalizePersistedLiveSubtitleNumber(
  value: unknown,
  minimum: number,
  maximum: number,
): number | null {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    return null;
  }
  return value;
}


function createDefaultLiveOverlaySettings(): LiveOverlaySettings {
  return {
    mode: "subtitle",
    attachment: "bottom",
    offset: 0,
    showSource: true,
    showRegionBoxes: false,
    autoWidth: true,
    autoHeight: true,
    manualWidth: DEFAULT_LIVE_SUBTITLE_MANUAL_WIDTH,
    manualHeight: DEFAULT_LIVE_SUBTITLE_MANUAL_HEIGHT,
  };
}

export const targetLanguage = ref("Chinese");
export const backendStatus = ref<BackendStatus | null>(null);
export const modelRuntimeStatus = ref<ModelRuntimeStatus | null>(null);

export const liveOverlaySettings = ref(createDefaultLiveOverlaySettings());
export const liveSubtitleStyleSettings = ref(createDefaultLiveSubtitleStyleSettings());
export const liveRecognitionSettings = ref<LiveRecognitionSettings>({
  mode: "automatic",
  triggerKey: "F8",
  triggerEvent: "press",
  stabilityWaitMs: DEFAULT_LIVE_STABILITY_WAIT_MS,
  keyTriggerTimeoutMs: DEFAULT_KEY_TRIGGER_TIMEOUT_MS,
  textGroupingEnabled: true,
});

export const liveTranslationSettings = ref<LiveTranslationSettings>({
  supplementalPrompt: "",
  memoryEnabled: DEFAULT_LIVE_MEMORY_ENABLED,
  memoryMaxTokens: DEFAULT_LIVE_MEMORY_TOKENS,
  memoryMaxTurns: DEFAULT_LIVE_MEMORY_TURNS,
});

let persistedTargetLanguageLoaded = false;

export function loadPersistedTargetLanguage(): string | null {
  if (persistedTargetLanguageLoaded || typeof window === "undefined") {
    return null;
  }
  persistedTargetLanguageLoaded = true;

  try {
    const persistedLanguage = window.localStorage.getItem(TARGET_LANGUAGE_STORAGE_KEY);
    if (persistedLanguage?.trim()) {
      targetLanguage.value = persistedLanguage.trim();
    }
    return null;
  } catch {
    return "无法读取本地设置，将使用默认目标语言。";
  }
}

export function savePersistedTargetLanguage(): string | null {
  try {
    window.localStorage.setItem(TARGET_LANGUAGE_STORAGE_KEY, targetLanguage.value);
    return null;
  } catch {
    return "无法写入本地设置，请检查应用存储权限。";
  }
}

export function applySharedBackendStatus(status: BackendStatus) {
  backendStatus.value = status;
  targetLanguage.value = status.targetLanguage;
}

export function applySharedModelRuntimeStatus(status: ModelRuntimeStatus) {
  modelRuntimeStatus.value = status;
  applySharedBackendStatus(status.backend);
}

export async function fetchSharedBackendStatus(): Promise<BackendStatus> {
  const status = await getBackendStatus();
  applySharedBackendStatus(status);
  return status;
}

export async function fetchSharedModelRuntimeStatus(): Promise<ModelRuntimeStatus> {
  const status = await getModelRuntimeStatus();
  applySharedModelRuntimeStatus(status);
  return status;
}

let persistedLiveOverlaySettingsLoaded = false;
let liveOverlaySettingsStorageSyncBound = false;
let persistedLiveSubtitleStyleSettingsLoaded = false;
let liveSubtitleStyleSettingsStorageSyncBound = false;

function normalizePersistedLiveOverlayDimension(
  value: unknown,
  minimum: number,
  maximum: number,
): number | null {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    return null;
  }
  return value;
}

function applyPersistedLiveOverlaySettings(
  rawSettings: string | null,
  resetToDefaults: boolean,
): string | null {
  if (!rawSettings) {
    if (resetToDefaults) {
      liveOverlaySettings.value = createDefaultLiveOverlaySettings();
    }
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(rawSettings);
    if (!parsed || typeof parsed !== "object") {
      return "实时浮层设置无效，将使用默认值。";
    }
    const value = parsed as Partial<LiveOverlaySettings>;
    const { mode, attachment, offset, showSource, showRegionBoxes, autoWidth, autoHeight } =
      value;
    const manualWidth =
      value.manualWidth === undefined
        ? DEFAULT_LIVE_SUBTITLE_MANUAL_WIDTH
        : normalizePersistedLiveOverlayDimension(
            value.manualWidth,
            LIVE_SUBTITLE_MANUAL_WIDTH_MIN,
            LIVE_SUBTITLE_MANUAL_WIDTH_MAX,
          );
    const manualHeight =
      value.manualHeight === undefined
        ? DEFAULT_LIVE_SUBTITLE_MANUAL_HEIGHT
        : normalizePersistedLiveOverlayDimension(
            value.manualHeight,
            LIVE_SUBTITLE_MANUAL_HEIGHT_MIN,
            LIVE_SUBTITLE_MANUAL_HEIGHT_MAX,
          );
    if (
      (mode !== "subtitle" && mode !== "region_replace") ||
      (attachment !== "top" &&
        attachment !== "bottom" &&
        attachment !== "left" &&
        attachment !== "right") ||
      !Number.isInteger(offset) ||
      offset === undefined ||
      offset < 0 ||
      offset > 2048 ||
      typeof showSource !== "boolean" ||
      (showRegionBoxes !== undefined && typeof showRegionBoxes !== "boolean") ||
      (autoWidth !== undefined && typeof autoWidth !== "boolean") ||
      (autoHeight !== undefined && typeof autoHeight !== "boolean") ||
      manualWidth === null ||
      manualHeight === null
    ) {
      return "实时浮层设置无效，将使用默认值。";
    }
    liveOverlaySettings.value = {
      mode,
      attachment,
      offset,
      showSource,
      showRegionBoxes: showRegionBoxes ?? false,
      autoWidth: autoWidth ?? true,
      autoHeight: autoHeight ?? true,
      manualWidth,
      manualHeight,
    };
    return null;
  } catch {
    return "无法读取实时浮层设置，将使用默认值。";
  }
}

function bindLiveOverlaySettingsStorageSync(): void {
  if (liveOverlaySettingsStorageSyncBound || typeof window === "undefined") {
    return;
  }
  liveOverlaySettingsStorageSyncBound = true;
  window.addEventListener("storage", (event) => {
    if (event.key !== LIVE_OVERLAY_SETTINGS_STORAGE_KEY) {
      return;
    }
    const error = applyPersistedLiveOverlaySettings(event.newValue, true);
    if (error) {
      liveOverlaySettings.value = createDefaultLiveOverlaySettings();
    }
  });
}

export function loadPersistedLiveOverlaySettings(): string | null {
  bindLiveOverlaySettingsStorageSync();
  if (persistedLiveOverlaySettingsLoaded || typeof window === "undefined") {
    return null;
  }
  persistedLiveOverlaySettingsLoaded = true;
  try {
    return applyPersistedLiveOverlaySettings(
      window.localStorage.getItem(LIVE_OVERLAY_SETTINGS_STORAGE_KEY),
      false,
    );
  } catch {
    return "无法读取实时浮层设置，将使用默认值。";
  }
}

export function savePersistedLiveOverlaySettings(): string | null {
  const settings = liveOverlaySettings.value;
  const offset = settings.offset;
  const manualWidth = settings.manualWidth;
  const manualHeight = settings.manualHeight;
  if (!Number.isInteger(offset) || offset < 0 || offset > 2_048) {
    return "实时翻译框外侧偏移必须为 0 到 2048 的整数。";
  }
  if (
    !Number.isInteger(manualWidth) ||
    manualWidth < LIVE_SUBTITLE_MANUAL_WIDTH_MIN ||
    manualWidth > LIVE_SUBTITLE_MANUAL_WIDTH_MAX
  ) {
    return `字幕窗口手动宽度必须为 ${LIVE_SUBTITLE_MANUAL_WIDTH_MIN} 到 ${LIVE_SUBTITLE_MANUAL_WIDTH_MAX} 的整数。`;
  }
  if (
    !Number.isInteger(manualHeight) ||
    manualHeight < LIVE_SUBTITLE_MANUAL_HEIGHT_MIN ||
    manualHeight > LIVE_SUBTITLE_MANUAL_HEIGHT_MAX
  ) {
    return `字幕窗口手动高度必须为 ${LIVE_SUBTITLE_MANUAL_HEIGHT_MIN} 到 ${LIVE_SUBTITLE_MANUAL_HEIGHT_MAX} 的整数。`;
  }
  const normalizedSettings = {
    ...settings,
    offset,
    manualWidth,
    manualHeight,
  };
  liveOverlaySettings.value = normalizedSettings;
  try {
    window.localStorage.setItem(
      LIVE_OVERLAY_SETTINGS_STORAGE_KEY,
      JSON.stringify(normalizedSettings),
    );
    return null;
  } catch {
    return "无法写入实时浮层设置，请检查应用存储权限。";
  }
}

function applyPersistedLiveSubtitleStyleSettings(
  rawSettings: string | null,
  resetToDefaults: boolean,
): string | null {
  if (!rawSettings) {
    if (resetToDefaults) {
      liveSubtitleStyleSettings.value = createDefaultLiveSubtitleStyleSettings();
    }
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(rawSettings);
    if (!parsed || typeof parsed !== "object") {
      return "字幕样式设置无效，将使用默认值。";
    }
    const value = parsed as Partial<LiveSubtitleStyleSettings>;
    const defaults = createDefaultLiveSubtitleStyleSettings();
    const fontColor =
      value.fontColor === undefined
        ? defaults.fontColor
        : normalizePersistedLiveSubtitleColor(value.fontColor);
    const fontSize =
      value.fontSize === undefined
        ? defaults.fontSize
        : normalizePersistedLiveSubtitleNumber(
            value.fontSize,
            LIVE_SUBTITLE_FONT_SIZE_MIN,
            LIVE_SUBTITLE_FONT_SIZE_MAX,
          );
    const backgroundColor =
      value.backgroundColor === undefined
        ? defaults.backgroundColor
        : normalizePersistedLiveSubtitleColor(value.backgroundColor);
    const backgroundOpacity =
      value.backgroundOpacity === undefined
        ? defaults.backgroundOpacity
        : normalizePersistedLiveSubtitleNumber(
            value.backgroundOpacity,
            LIVE_SUBTITLE_BACKGROUND_OPACITY_MIN,
            LIVE_SUBTITLE_BACKGROUND_OPACITY_MAX,
          );
    if (
      fontColor === null ||
      fontSize === null ||
      backgroundColor === null ||
      backgroundOpacity === null
    ) {
      return "字幕样式设置无效，将使用默认值。";
    }
    liveSubtitleStyleSettings.value = {
      fontColor,
      fontSize,
      backgroundColor,
      backgroundOpacity,
    };
    return null;
  } catch {
    return "无法读取字幕样式设置，将使用默认值。";
  }
}

function bindLiveSubtitleStyleSettingsStorageSync(): void {
  if (
    liveSubtitleStyleSettingsStorageSyncBound ||
    typeof window === "undefined" ||
    typeof window.addEventListener !== "function"
  ) {
    return;
  }
  liveSubtitleStyleSettingsStorageSyncBound = true;
  window.addEventListener("storage", (event) => {
    if (event.key !== LIVE_SUBTITLE_STYLE_SETTINGS_STORAGE_KEY) {
      return;
    }
    const error = applyPersistedLiveSubtitleStyleSettings(event.newValue, true);
    if (error) {
      liveSubtitleStyleSettings.value = createDefaultLiveSubtitleStyleSettings();
    }
  });
}

export function loadPersistedLiveSubtitleStyleSettings(): string | null {
  bindLiveSubtitleStyleSettingsStorageSync();
  if (persistedLiveSubtitleStyleSettingsLoaded || typeof window === "undefined") {
    return null;
  }
  persistedLiveSubtitleStyleSettingsLoaded = true;
  try {
    return applyPersistedLiveSubtitleStyleSettings(
      window.localStorage.getItem(LIVE_SUBTITLE_STYLE_SETTINGS_STORAGE_KEY),
      false,
    );
  } catch {
    return "无法读取字幕样式设置，将使用默认值。";
  }
}

export function savePersistedLiveSubtitleStyleSettings(): string | null {
  const settings = liveSubtitleStyleSettings.value;
  const fontColor = normalizePersistedLiveSubtitleColor(settings.fontColor);
  const fontSize = normalizePersistedLiveSubtitleNumber(
    settings.fontSize,
    LIVE_SUBTITLE_FONT_SIZE_MIN,
    LIVE_SUBTITLE_FONT_SIZE_MAX,
  );
  const backgroundColor = normalizePersistedLiveSubtitleColor(settings.backgroundColor);
  const backgroundOpacity = normalizePersistedLiveSubtitleNumber(
    settings.backgroundOpacity,
    LIVE_SUBTITLE_BACKGROUND_OPACITY_MIN,
    LIVE_SUBTITLE_BACKGROUND_OPACITY_MAX,
  );
  if (fontColor === null) {
    return "字幕字体颜色格式无效。";
  }
  if (fontSize === null) {
    return `字幕字体大小必须为 ${LIVE_SUBTITLE_FONT_SIZE_MIN} 到 ${LIVE_SUBTITLE_FONT_SIZE_MAX} 的整数。`;
  }
  if (backgroundColor === null) {
    return "字幕背景颜色格式无效。";
  }
  if (backgroundOpacity === null) {
    return `字幕背景透明度必须为 ${LIVE_SUBTITLE_BACKGROUND_OPACITY_MIN} 到 ${LIVE_SUBTITLE_BACKGROUND_OPACITY_MAX} 的整数。`;
  }
  const normalizedSettings = {
    fontColor,
    fontSize,
    backgroundColor,
    backgroundOpacity,
  };
  liveSubtitleStyleSettings.value = normalizedSettings;
  try {
    window.localStorage.setItem(
      LIVE_SUBTITLE_STYLE_SETTINGS_STORAGE_KEY,
      JSON.stringify(normalizedSettings),
    );
    return null;
  } catch {
    return "无法写入字幕样式设置，请检查应用存储权限。";
  }
}

let persistedLiveRecognitionSettingsLoaded = false;

function normalizePersistedLiveTriggerKey(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const key = value.trim();
  return key || null;
}

function normalizePersistedLiveStabilityWaitMs(value: unknown): number | null {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < LIVE_STABILITY_WAIT_MIN_MS ||
    value > LIVE_STABILITY_WAIT_MAX_MS
  ) {
    return null;
  }
  return value;
}

function normalizePersistedKeyTriggerTimeoutMs(value: unknown): number | null {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < KEY_TRIGGER_TIMEOUT_MIN_MS ||
    value > KEY_TRIGGER_TIMEOUT_MAX_MS
  ) {
    return null;
  }
  return value;
}

export function loadPersistedLiveRecognitionSettings(): string | null {
  if (persistedLiveRecognitionSettingsLoaded || typeof window === "undefined") {
    return null;
  }
  persistedLiveRecognitionSettingsLoaded = true;
  try {
    const rawSettings = window.localStorage.getItem(LIVE_RECOGNITION_SETTINGS_STORAGE_KEY);
    if (!rawSettings) {
      return null;
    }
    const parsed: unknown = JSON.parse(rawSettings);
    if (!parsed || typeof parsed !== "object") {
      return "实时识别触发设置无效，将使用默认值。";
    }
    const value = parsed as {
      mode?: string;
      triggerKey?: unknown;
      triggerEvent?: string;
      stabilityWaitMs?: unknown;
      keyTriggerTimeoutMs?: unknown;
      textGroupingEnabled?: unknown;
    };
    const mode = value.mode === "hold_key" ? "key_trigger" : value.mode;
    const triggerEvent = value.triggerEvent ?? "press";
    const triggerKey = normalizePersistedLiveTriggerKey(value.triggerKey);
    const stabilityWaitMs =
      value.stabilityWaitMs === undefined
        ? DEFAULT_LIVE_STABILITY_WAIT_MS
        : normalizePersistedLiveStabilityWaitMs(value.stabilityWaitMs);
    const keyTriggerTimeoutMs =
      value.keyTriggerTimeoutMs === undefined
        ? DEFAULT_KEY_TRIGGER_TIMEOUT_MS
        : normalizePersistedKeyTriggerTimeoutMs(value.keyTriggerTimeoutMs);
    const textGroupingEnabled =
      value.textGroupingEnabled === undefined ? true : value.textGroupingEnabled;
    if (
      (mode !== "automatic" && mode !== "key_trigger") ||
      (triggerEvent !== "press" && triggerEvent !== "release") ||
      !triggerKey ||
      stabilityWaitMs === null ||
      keyTriggerTimeoutMs === null ||
      typeof textGroupingEnabled !== "boolean"
    ) {
      return "实时识别触发设置无效，将使用默认值。";
    }
    liveRecognitionSettings.value = {
      mode,
      triggerKey,
      triggerEvent,
      stabilityWaitMs,
      keyTriggerTimeoutMs,
      textGroupingEnabled,
    };
    return null;
  } catch {
    return "无法读取实时识别触发设置，将使用默认值。";
  }
}

export function savePersistedLiveRecognitionSettings(): string | null {
  const settings = liveRecognitionSettings.value;
  if (normalizePersistedLiveStabilityWaitMs(settings.stabilityWaitMs) === null) {
    return `OCR 字幕稳定等待必须为 ${LIVE_STABILITY_WAIT_MIN_MS} 到 ${LIVE_STABILITY_WAIT_MAX_MS} 毫秒的整数。`;
  }
  if (normalizePersistedKeyTriggerTimeoutMs(settings.keyTriggerTimeoutMs) === null) {
    return `按键触发 OCR 超时必须为 ${KEY_TRIGGER_TIMEOUT_MIN_MS} 到 ${KEY_TRIGGER_TIMEOUT_MAX_MS} 毫秒的整数。`;
  }
  const triggerKey = settings.triggerKey.trim();
  if (!triggerKey) {
    return "实时翻译触发按键不能为空，请先录入按键。";
  }
  liveRecognitionSettings.value = {
    ...settings,
    triggerKey,
  };
  try {
    window.localStorage.setItem(
      LIVE_RECOGNITION_SETTINGS_STORAGE_KEY,
      JSON.stringify(liveRecognitionSettings.value),
    );
    return null;
  } catch {
    return "无法写入实时识别触发设置，请检查应用存储权限。";
  }
}

let persistedLiveTranslationSettingsLoaded = false;

export function loadPersistedLiveTranslationSettings(): string | null {
  if (persistedLiveTranslationSettingsLoaded || typeof window === "undefined") {
    return null;
  }
  persistedLiveTranslationSettingsLoaded = true;
  try {
    const rawSettings = window.localStorage.getItem(LIVE_TRANSLATION_SETTINGS_STORAGE_KEY);
    if (!rawSettings) {
      return null;
    }
    const parsed: unknown = JSON.parse(rawSettings);
    if (!parsed || typeof parsed !== "object") {
      return "实时翻译设置无效，将使用默认值。";
    }
    const value = parsed as Partial<LiveTranslationSettings>;
    const supplementalPrompt = value.supplementalPrompt;
    const memoryEnabled =
      value.memoryEnabled === undefined ? DEFAULT_LIVE_MEMORY_ENABLED : value.memoryEnabled;
    const memoryMaxTokens =
      value.memoryMaxTokens === undefined ? DEFAULT_LIVE_MEMORY_TOKENS : value.memoryMaxTokens;
    const memoryMaxTurns =
      value.memoryMaxTurns === undefined ? DEFAULT_LIVE_MEMORY_TURNS : value.memoryMaxTurns;
    if (
      typeof supplementalPrompt !== "string" ||
      Array.from(supplementalPrompt).length > LIVE_SUPPLEMENTAL_PROMPT_MAX_CHARS ||
      typeof memoryEnabled !== "boolean" ||
      !Number.isInteger(memoryMaxTokens) ||
      memoryMaxTokens < LIVE_MEMORY_TOKENS_MIN ||
      memoryMaxTokens > LIVE_MEMORY_TOKENS_MAX ||
      !Number.isInteger(memoryMaxTurns) ||
      memoryMaxTurns < LIVE_MEMORY_TURNS_MIN ||
      memoryMaxTurns > LIVE_MEMORY_TURNS_MAX
    ) {
      return "实时翻译记忆设置无效，将使用默认值。";
    }
    liveTranslationSettings.value = {
      supplementalPrompt,
      memoryEnabled,
      memoryMaxTokens,
      memoryMaxTurns,
    };
    return null;
  } catch {
    return "无法读取实时翻译设置，将使用默认值。";
  }
}

export function savePersistedLiveTranslationSettings(): string | null {
  const settings = liveTranslationSettings.value;
  const supplementalPrompt = settings.supplementalPrompt.trim();
  if (Array.from(supplementalPrompt).length > LIVE_SUPPLEMENTAL_PROMPT_MAX_CHARS) {
    return "实时翻译补充提示不能超过 4096 个字符。";
  }
  if (
    !Number.isInteger(settings.memoryMaxTokens) ||
    settings.memoryMaxTokens < LIVE_MEMORY_TOKENS_MIN ||
    settings.memoryMaxTokens > LIVE_MEMORY_TOKENS_MAX
  ) {
    return `实时翻译记忆 token 预算必须为 ${LIVE_MEMORY_TOKENS_MIN} 到 ${LIVE_MEMORY_TOKENS_MAX} 的整数。`;
  }
  if (
    !Number.isInteger(settings.memoryMaxTurns) ||
    settings.memoryMaxTurns < LIVE_MEMORY_TURNS_MIN ||
    settings.memoryMaxTurns > LIVE_MEMORY_TURNS_MAX
  ) {
    return `实时翻译记忆轮数必须为 ${LIVE_MEMORY_TURNS_MIN} 到 ${LIVE_MEMORY_TURNS_MAX} 的整数。`;
  }
  liveTranslationSettings.value = {
    ...settings,
    supplementalPrompt,
  };
  try {
    window.localStorage.setItem(
      LIVE_TRANSLATION_SETTINGS_STORAGE_KEY,
      JSON.stringify(liveTranslationSettings.value),
    );
    return null;
  } catch {
    return "无法写入实时翻译设置，请检查应用存储权限。";
  }
}
