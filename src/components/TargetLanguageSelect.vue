<script setup lang="ts">
import { NSelect } from "naive-ui";
import { TARGET_LANGUAGE_OPTIONS } from "../constants/targetLanguageOptions";

defineProps<{
  modelValue: string;
  disabled?: boolean;
  placeholder?: string;
  ariaLabel?: string;
  filterable?: boolean;
  size?: "small" | "medium" | "large";
}>();

const emit = defineEmits<{
  (e: "update:modelValue", value: string): void;
}>();

function handleUpdate(value: string | null) {
  emit("update:modelValue", value ?? "");
}

</script>

<template>
  <n-select
    :value="modelValue"
    :options="TARGET_LANGUAGE_OPTIONS"
    :disabled="disabled"
    :filterable="filterable ?? true"
    :placeholder="placeholder ?? '选择目标语言（Hy-MT2 支持 38 种）'"
    :aria-label="ariaLabel ?? '目标语言'"
    :size="size"
    @update:value="handleUpdate"
  />
</template>
