import { computed, ref } from "vue";
import { backendStatus, fetchSharedBackendStatus, targetLanguage } from "./workspace-settings";
import { updateBackendSettings, type DeviceKind } from "./translation-provider";
import {
  isSupportedTargetLanguage,
  targetLanguageOptionsFor,
} from "../constants/targetLanguageOptions";

const TL_LOG_PREFIX = " [target-language-store]";

export const targetLanguageSaving = ref(false);

export const targetLanguageOptions = computed(() =>
  targetLanguageOptionsFor(backendStatus.value?.hyModel),
);

export function validateTargetLanguageValue(value: string): string | null {
  if (!isSupportedTargetLanguage(value)) {
    return "目标语言不在支持列表内，请从下拉选择。";
  }
  return null;
}

export async function saveTargetLanguage(lang: string): Promise<string | null> {
  const next = lang.trim();
  const invalid = validateTargetLanguageValue(next);
  if (invalid) {
    return invalid;
  }
  const status = backendStatus.value;
  if (!status) {
    return "后端状态未就绪，请先刷新。";
  }
  if (targetLanguageSaving.value) {
    return "正在保存目标语言，请稍候。";
  }
  targetLanguageSaving.value = true;
  try {
    await updateBackendSettings({
      detectorModelDir: status.detectorModelDir,
      recognizerModelDir: status.recognizerModelDir,
      hyModel: status.hyModel,
      fontPath: status.fontPath,
      targetLanguage: next,
      device: (status.device === "cpu" ? "cpu" : "cuda") as DeviceKind,
      regionParallelism: status.regionParallelism,
      translationBatchSize: status.translationBatchSize,
      idleUnloadSeconds: status.idleUnloadSeconds,
      generation: { ...status.generation },
      memory: { ...status.memory },
      prompt: { template: status.prompt.template },
    });
    const refreshed = await fetchSharedBackendStatus();
    targetLanguage.value = refreshed.targetLanguage;
    console.info(`${TL_LOG_PREFIX} saveTargetLanguage success`, { targetLanguage: next });
    return null;
  } catch (error) {
    targetLanguage.value = status.targetLanguage;
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`${TL_LOG_PREFIX} saveTargetLanguage failed`, { targetLanguage: next, error: message });
    return message;
  } finally {
    targetLanguageSaving.value = false;
  }
}
