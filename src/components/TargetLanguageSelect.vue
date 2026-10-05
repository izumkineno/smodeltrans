<script setup lang="ts">
import { computed } from "vue";
import { NSelect } from "naive-ui";
import { targetLanguageOptionsFor } from "../constants/targetLanguageOptions";

const props = defineProps<{
  modelValue: string;
  disabled?: boolean;
  placeholder?: string;
  ariaLabel?: string;
  filterable?: boolean;
  size?: "small" | "medium" | "large";
  modelPath?: string | null;
}>();

const options = computed(() => targetLanguageOptionsFor(props.modelPath));
const effectivePlaceholder = computed(
  () => props.placeholder ?? `选择目标语言（支持 ${options.value.length} 种）`,
);

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
    :options="options"
    :disabled="disabled"
    :filterable="filterable ?? true"
    :placeholder="effectivePlaceholder"
    :aria-label="ariaLabel ?? '目标语言'"
    :size="size"
    @update:value="handleUpdate"
  />
</template>
