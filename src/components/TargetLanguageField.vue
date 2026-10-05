<script setup lang="ts">
import TargetLanguageSelect from "./TargetLanguageSelect.vue";
import { saveTargetLanguage, targetLanguageSaving } from "../services/target-language-store";
import { backendStatus, targetLanguage } from "../services/workspace-settings";

defineProps<{
  disabled?: boolean;
  ariaLabel?: string;
  immediate?: boolean;
}>();

const emit = defineEmits<{
  (e: "saved", value: string): void;
  (e: "failed", message: string): void;
}>();

async function handleUpdate(value: string, immediate?: boolean): Promise<void> {
  if (!immediate) {
    return;
  }
  const error = await saveTargetLanguage(value);
  if (error === null) {
    emit("saved", value.trim());
  } else {
    emit("failed", error);
  }
}
</script>

<template>
  <TargetLanguageSelect
    v-model="targetLanguage"
    :model-path="backendStatus?.hyModel ?? null"
    :disabled="disabled || targetLanguageSaving"
    :aria-label="ariaLabel"
    @update:model-value="(value) => handleUpdate(value, immediate)"
  />
</template>
