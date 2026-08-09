import { ref } from "vue";
import { getBackendStatus } from "./translation-provider";
import type { BackendStatus } from "./translation-provider";

const TARGET_LANGUAGE_STORAGE_KEY = "smodeltrans.targetLanguage";

export const targetLanguage = ref("Chinese");
export const backendStatus = ref<BackendStatus | null>(null);

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

export async function fetchSharedBackendStatus(): Promise<BackendStatus> {
  const status = await getBackendStatus();
  applySharedBackendStatus(status);
  return status;
}
