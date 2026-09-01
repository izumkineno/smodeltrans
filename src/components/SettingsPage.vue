<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import {
  NAlert,
  NButton,
  NCard,
  NInput,
  NSelect,
  NTag,
  useMessage,
} from "naive-ui";
import { updateBackendSettings } from "../services/translation-provider";
import type { BackendSettingsUpdate, BackendStatus } from "../services/translation-provider";
import {
  applySharedBackendStatus,
  backendStatus,
  fetchSharedBackendStatus,
  savePersistedTargetLanguage,
  targetLanguage,
} from "../services/workspace-settings";
import { setThemeMode, themeMode } from "../services/theme-settings";
import type { ThemeMode } from "../services/theme-settings";
import { showWorkspaceToast, type WorkspaceToastType } from "../services/workspace-toast";
import OpenAiCompatCard from "./OpenAiCompatCard.vue";
type TagType = "default" | "success" | "warning" | "error" | "info";

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const systemPrompt = ref("");
const userPrompt = ref("");
const settingsMessage = ref("");
const settingsMessageType = ref<WorkspaceToastType>("info");
const settingsLoading = ref(false);
const themeModeOptions: Array<{ label: string; value: ThemeMode }> = [
  { label: "自动（跟随系统）", value: "system" },
  { label: "浅色（手动）", value: "light" },
  { label: "深色（手动）", value: "dark" },
];
const themeModeLabels: Record<ThemeMode, string> = {
  system: "自动（跟随系统）",
  light: "浅色（手动）",
  dark: "深色（手动）",
};
const toast = useMessage();

function setSettingsFeedback(
  type: WorkspaceToastType,
  message: string,
  notify = true,
) {
  settingsMessage.value = message;
  settingsMessageType.value = type;
  if (notify) {
    showWorkspaceToast(toast, type, message);
  }
}

function handleThemeModeChange(nextMode: ThemeMode | null): void {
  if (!nextMode) {
    return;
  }
  const persistError = setThemeMode(nextMode);
  if (persistError) {
    setSettingsFeedback("error", persistError);
    return;
  }
  setSettingsFeedback("success", `界面主题已切换为${themeModeLabels[nextMode]}。`, false);
}

const settingsTagType = computed<TagType>(() => {
  if (!backendStatus.value) {
    return "default";
  }
  return backendStatus.value.ready ? "success" : "warning";
});

const settingsStatusLabel = computed(() => {
  if (!backendStatus.value) {
    return isDesktopRuntime ? "读取中" : "浏览器预览";
  }
  return backendStatus.value.ready ? "模型已就绪" : "需要检查";
});

function applyBackendStatus(status: BackendStatus) {
  applySharedBackendStatus(status);
  // targetLanguage is global ref, keep in sync
  if (status.targetLanguage) {
    targetLanguage.value = status.targetLanguage;
  }
  systemPrompt.value = status.prompt.system;
  userPrompt.value = status.prompt.user;
}

async function refreshBackendStatus(notify = true) {
  if (!isDesktopRuntime) {
    setSettingsFeedback("warning", "设置状态仅在 Tauri 桌面端可读取。", notify);
    return;
  }
  settingsLoading.value = true;
  try {
    const status = await fetchSharedBackendStatus();
    applyBackendStatus(status);
    setSettingsFeedback(status.ready ? "success" : "warning", status.message, notify);
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法读取后端状态。", notify);
  } finally {
    settingsLoading.value = false;
  }
}

async function saveSettings() {
  const nextLanguage = targetLanguage.value.trim();
  if (Array.from(nextLanguage).length < 1 || Array.from(nextLanguage).length > 64) {
    setSettingsFeedback("error", "目标语言长度必须为 1 到 64 个字符。");
    return;
  }

  const trimmedSystemPrompt = systemPrompt.value.trim();
  if (Array.from(trimmedSystemPrompt).length > 4096) {
    setSettingsFeedback("error", "system 预设提示词最多 4096 个字符。");
    return;
  }
  const trimmedUserPrompt = userPrompt.value.trim();
  if (Array.from(trimmedUserPrompt).length > 4096) {
    setSettingsFeedback("error", "user 预设提示词最多 4096 个字符。");
    return;
  }

  if (!isDesktopRuntime) {
    targetLanguage.value = nextLanguage;
    const persistError = savePersistedTargetLanguage();
    if (persistError) {
      setSettingsFeedback("error", persistError);
      return;
    }
    setSettingsFeedback("success", "目标语言已保存（浏览器预览）。");
    return;
  }

  const status = backendStatus.value;
  if (!status) {
    setSettingsFeedback("error", "后端状态未就绪，请先刷新。");
    return;
  }

  const settings: BackendSettingsUpdate = {
    detectorModelDir: status.detectorModelDir,
    recognizerModelDir: status.recognizerModelDir,
    hyModel: status.hyModel,
    fontPath: status.fontPath,
    targetLanguage: nextLanguage,
    device: status.device === "cpu" ? "cpu" : "cuda",
    regionParallelism: status.regionParallelism,
    translationBatchSize: status.translationBatchSize,
    idleUnloadSeconds: status.idleUnloadSeconds,
    generation: status.generation,
    memory: status.memory,
    prompt: {
      system: trimmedSystemPrompt,
      user: trimmedUserPrompt,
    },
  };

  settingsLoading.value = true;
  try {
    const nextStatus = await updateBackendSettings(settings);
    applyBackendStatus(nextStatus);
    targetLanguage.value = nextLanguage;
    const persistError = savePersistedTargetLanguage();
    if (persistError) {
      setSettingsFeedback("warning", `后端已保存，但本地语言持久化失败：${persistError}`);
      return;
    }
    setSettingsFeedback("success", "翻译偏好已保存，下一次翻译会使用新的提示词与目标语言。");
  } catch (error) {
    setSettingsFeedback("error", error instanceof Error ? error.message : "无法保存设置。");
  } finally {
    settingsLoading.value = false;
  }
}

onMounted(() => {
  if (backendStatus.value) {
    applyBackendStatus(backendStatus.value);
  }
  void refreshBackendStatus(false);
});
</script>

<template>
  <section class="settings-page" aria-labelledby="settings-title">
    <div class="settings-grid">
      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">界面外观</p>
            <h2>颜色主题</h2>
          </div>
        </div>
        <p class="settings-card-copy">
          选择手动主题，或使用自动模式跟随 Windows 系统的浅色与深色设置；切换会立即生效并自动保存。
        </p>
        <label class="settings-field settings-field-wide">
          <span>主题模式</span>
          <n-select
            :value="themeMode"
            :options="themeModeOptions"
            aria-label="主题模式"
            @update:value="handleThemeModeChange"
          />
          <span class="settings-help">自动模式会在系统主题变化时实时切换。</span>
        </label>
      </n-card>

      <n-card class="settings-card" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">运行状态</p>
            <h2>模型服务</h2>
          </div>
          <n-tag :type="settingsTagType" round size="small">{{ settingsStatusLabel }}</n-tag>
        </div>
        <p class="settings-card-copy">
          这里显示后端实际读取到的设备与模型状态，不会伪造就绪结果。详细模型与运行参数请前往“模型管理”配置。
        </p>
        <div class="settings-metrics">
          <div>
            <span>设备</span>
            <strong>{{ backendStatus?.device ?? "未读取" }}</strong>
          </div>
          <div>
            <span>翻译器</span>
            <strong>{{ backendStatus?.translatorLoaded ? "已加载" : "按需加载" }}</strong>
          </div>
          <div>
            <span>OCR 并发</span>
            <strong>{{ backendStatus?.regionParallelism ?? "—" }}</strong>
          </div>
          <div>
            <span>Hy 批大小</span>
            <strong>{{ backendStatus?.translationBatchSize ?? "—" }}</strong>
          </div>
        </div>
        <p class="settings-card-copy">
          模型路径、设备、批处理、空闲释放及生成参数已迁移至
          <strong>模型管理</strong> 页面。
        </p>
      </n-card>

      <n-card class="settings-card settings-card-wide" :bordered="false">
        <div class="settings-card-heading">
          <div>
            <p class="panel-kicker">翻译偏好</p>
            <h2>目标语言与提示词</h2>
          </div>
        </div>
        <p class="settings-card-copy">先设置目标语言；system 和 user 预设提示词会作为下一次翻译请求的默认模型上下文。</p>
        <div class="settings-field-grid">
          <label class="settings-field">
            <span>目标语言</span>
            <n-input
              v-model:value="targetLanguage"
              maxlength="64"
              placeholder="例如：Chinese"
              aria-label="目标语言"
            />
          </label>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>System 预设提示词</span>
            <n-input
              v-model:value="systemPrompt"
              type="textarea"
              maxlength="4096"
              placeholder="可选：例如 Return concise JSON."
              :autosize="{ minRows: 3, maxRows: 6 }"
              aria-label="Hy system prompt"
            />
          </label>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>User 预设提示词</span>
            <n-input
              v-model:value="userPrompt"
              type="textarea"
              maxlength="4096"
              placeholder="可选：例如 Preserve product names and translate only visible text."
              :autosize="{ minRows: 3, maxRows: 6 }"
              aria-label="Hy user preset prompt"
            />
          </label>
        </div>
      </n-card>

      <OpenAiCompatCard />

      <div class="settings-card-actions settings-page-actions settings-card-wide">
        <n-alert v-if="settingsMessage" class="settings-actions-feedback" :type="settingsMessageType" :show-icon="false">
          {{ settingsMessage }}
        </n-alert>
        <n-button secondary :loading="settingsLoading" @click="refreshBackendStatus()">刷新状态</n-button>
        <n-button type="primary" :loading="settingsLoading" @click="saveSettings">保存设置</n-button>
      </div>
    </div>
  </section>
</template>
<style scoped src="../styles/settings-page.css"></style>
