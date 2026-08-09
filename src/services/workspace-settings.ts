import { ref } from "vue";
import { getBackendStatus, getModelRuntimeStatus } from "./translation-provider";
import type { BackendStatus, ModelRuntimeStatus } from "./translation-provider";

const TARGET_LANGUAGE_STORAGE_KEY = "smodeltrans.targetLanguage";

export const targetLanguage = ref("Chinese");
export const backendStatus = ref<BackendStatus | null>(null);
export const modelRuntimeStatus = ref<ModelRuntimeStatus | null>(null);

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
