import { ref } from "vue";
import { getBackendStatus, getModelRuntimeStatus } from "./translation-provider";
import type { BackendStatus, ModelRuntimeStatus } from "./translation-provider";
import type {
  LiveOverlaySettings,
  LiveRecognitionSettings,
} from "./live-translation-provider";

const TARGET_LANGUAGE_STORAGE_KEY = "smodeltrans.targetLanguage";
const LIVE_OVERLAY_SETTINGS_STORAGE_KEY = "smodeltrans.liveOverlaySettings";
const LIVE_RECOGNITION_SETTINGS_STORAGE_KEY = "smodeltrans.liveRecognitionSettings";
export const DEFAULT_LIVE_STABILITY_WAIT_MS = 300;
export const LIVE_STABILITY_WAIT_MIN_MS = 0;
export const LIVE_STABILITY_WAIT_MAX_MS = 5_000;

export const targetLanguage = ref("Chinese");
export const backendStatus = ref<BackendStatus | null>(null);
export const modelRuntimeStatus = ref<ModelRuntimeStatus | null>(null);

export const liveOverlaySettings = ref<LiveOverlaySettings>({
  mode: "subtitle",
  attachment: "bottom",
  offset: 0,
  showSource: true,
});

export const liveRecognitionSettings = ref<LiveRecognitionSettings>({
  mode: "automatic",
  triggerKey: "F8",
  triggerEvent: "press",
  stabilityWaitMs: DEFAULT_LIVE_STABILITY_WAIT_MS,

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

export function loadPersistedLiveOverlaySettings(): string | null {
  if (persistedLiveOverlaySettingsLoaded || typeof window === "undefined") {
    return null;
  }
  persistedLiveOverlaySettingsLoaded = true;
  try {
    const rawSettings = window.localStorage.getItem(LIVE_OVERLAY_SETTINGS_STORAGE_KEY);
    if (!rawSettings) {
      return null;
    }
    const parsed: unknown = JSON.parse(rawSettings);
    if (!parsed || typeof parsed !== "object") {
      return "实时浮层设置无效，将使用默认值。";
    }
    const value = parsed as Partial<LiveOverlaySettings>;
    const { mode, attachment, offset, showSource } = value;
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
      typeof showSource !== "boolean"
    ) {
      return "实时浮层设置无效，将使用默认值。";
    }
    liveOverlaySettings.value = { mode, attachment, offset, showSource };
    return null;
  } catch {
    return "无法读取实时浮层设置，将使用默认值。";
  }
}

export function savePersistedLiveOverlaySettings(): string | null {
  const settings = liveOverlaySettings.value;
  if (
    !Number.isInteger(settings.offset) ||
    settings.offset < 0 ||
    settings.offset > 2048
  ) {
    return "实时翻译框外侧偏移必须为 0 到 2048 的整数。";
  }
  try {
    window.localStorage.setItem(LIVE_OVERLAY_SETTINGS_STORAGE_KEY, JSON.stringify(settings));
    return null;
  } catch {
    return "无法写入实时浮层设置，请检查应用存储权限。";
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
    };
    const mode = value.mode === "hold_key" ? "key_trigger" : value.mode;
    const triggerEvent = value.triggerEvent ?? "press";
    const triggerKey = normalizePersistedLiveTriggerKey(value.triggerKey);
    const stabilityWaitMs =
      value.stabilityWaitMs === undefined
        ? DEFAULT_LIVE_STABILITY_WAIT_MS
        : normalizePersistedLiveStabilityWaitMs(value.stabilityWaitMs);
    if (
      (mode !== "automatic" && mode !== "key_trigger") ||
      (triggerEvent !== "press" && triggerEvent !== "release") ||
      !triggerKey ||
      stabilityWaitMs === null
    ) {
      return "实时识别触发设置无效，将使用默认值。";
    }
    liveRecognitionSettings.value = {
      mode,
      triggerKey,
      triggerEvent,
      stabilityWaitMs,
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
