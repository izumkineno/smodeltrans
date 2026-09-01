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
const promptTemplate = ref("");
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
const targetLanguageOptions: Array<{ label: string; value: string }> = [
  { label: "中文 (Chinese · zh)", value: "Chinese" },
  { label: "英语 (English · en)", value: "English" },
  { label: "法语 (French · fr)", value: "French" },
  { label: "葡萄牙语 (Portuguese · pt)", value: "Portuguese" },
  { label: "西班牙语 (Spanish · es)", value: "Spanish" },
  { label: "日语 (Japanese · ja)", value: "Japanese" },
  { label: "土耳其语 (Turkish · tr)", value: "Turkish" },
  { label: "俄语 (Russian · ru)", value: "Russian" },
  { label: "阿拉伯语 (Arabic · ar)", value: "Arabic" },
  { label: "韩语 (Korean · ko)", value: "Korean" },
  { label: "泰语 (Thai · th)", value: "Thai" },
  { label: "意大利语 (Italian · it)", value: "Italian" },
  { label: "德语 (German · de)", value: "German" },
  { label: "越南语 (Vietnamese · vi)", value: "Vietnamese" },
  { label: "马来语 (Malay · ms)", value: "Malay" },
  { label: "印尼语 (Indonesian · id)", value: "Indonesian" },
  { label: "菲律宾语 (Filipino · tl)", value: "Filipino" },
  { label: "印地语 (Hindi · hi)", value: "Hindi" },
  { label: "繁体中文 (Traditional Chinese · zh-Hant)", value: "Traditional Chinese" },
  { label: "波兰语 (Polish · pl)", value: "Polish" },
  { label: "捷克语 (Czech · cs)", value: "Czech" },
  { label: "荷兰语 (Dutch · nl)", value: "Dutch" },
  { label: "高棉语 (Khmer · km)", value: "Khmer" },
  { label: "缅甸语 (Burmese · my)", value: "Burmese" },
  { label: "波斯语 (Persian · fa)", value: "Persian" },
  { label: "古吉拉特语 (Gujarati · gu)", value: "Gujarati" },
  { label: "乌尔都语 (Urdu · ur)", value: "Urdu" },
  { label: "泰卢固语 (Telugu · te)", value: "Telugu" },
  { label: "马拉地语 (Marathi · mr)", value: "Marathi" },
  { label: "希伯来语 (Hebrew · he)", value: "Hebrew" },
  { label: "孟加拉语 (Bengali · bn)", value: "Bengali" },
  { label: "泰米尔语 (Tamil · ta)", value: "Tamil" },
  { label: "乌克兰语 (Ukrainian · uk)", value: "Ukrainian" },
  { label: "藏语 (Tibetan · bo)", value: "Tibetan" },
  { label: "哈萨克语 (Kazakh · kk)", value: "Kazakh" },
  { label: "蒙古语 (Mongolian · mn)", value: "Mongolian" },
  { label: "维吾尔语 (Uyghur · ug)", value: "Uyghur" },
  { label: "粤语 (Cantonese · yue)", value: "Cantonese" },
];
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
  // 兼容旧字段：优先 `template`，回退 `prompt`/`system`+`user`
  const p: any = status.prompt as any;
  promptTemplate.value = p.template ?? p.prompt ?? [p.system, p.user].filter(Boolean).join("\n\n") ?? "";
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
  const isSupported = targetLanguageOptions.some((opt) => opt.value === nextLanguage);
  if (!isSupported) {
    setSettingsFeedback("error", "目标语言不在 Hy-MT2 支持列表内，请从下拉选择（38 种）。");
    return;
  }

  const trimmedPromptTemplate = promptTemplate.value.trim();
  if (Array.from(trimmedPromptTemplate).length > 8192) {
    setSettingsFeedback("error", "提示词模板最多 8192 个字符。");
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
      template: trimmedPromptTemplate,
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
        <p class="settings-card-copy">
          目标语言决定 Hy-MT2 的 <code>target_lang</code>（需使用<strong>完整语言名</strong>，English prompt 用英文名，中文 prompt 用中文名）。
          本地翻译默认使用官方 Default 模板：<code>Translate the following text into {target_language}. Note that you should only output the translated result without any additional explanation:\n\n{source_text}</code>，
          参考 <a href="https://github.com/Tencent-Hunyuan/Hy-MT2#hy-mt2-translation-task-instruction-examples-chinese-english-comparison" target="_blank">Hy-MT2 官方指令示例</a>。
          单模板支持 <code>{source_text}</code> / <code>{target_lang}</code> / <code>{target_language}</code> / <code>{format_type}</code> 占位符。
        </p>
        <div class="settings-field-grid">
          <label class="settings-field">
            <span>目标语言</span>
            <n-select
              v-model:value="targetLanguage"
              :options="targetLanguageOptions"
              placeholder="选择目标语言（Hy-MT2 支持 38 种）"
              aria-label="目标语言"
            />
          </label>
          <span class="settings-help" style="grid-column: 1 / -1">仅支持 Hy-MT2 官方 38 语言，已自动归一化英文全称；不支持的语言将校验失败。</span>
          <label class="settings-field settings-field-wide settings-textarea">
            <span>翻译提示词模板</span>
            <n-input
              v-model:value="promptTemplate"
              type="textarea"
              maxlength="8192"
              placeholder="可选：单模板，支持 {source_text} / {target_lang} / {target_language} / {format_type} 占位符。留空则使用官方 Default：Translate the following text into {target}. Note that you should only output the translated result without any additional explanation: {text}"
              :autosize="{ minRows: 3, maxRows: 6 }"
              aria-label="Hy prompt template"
            />
          </label>
          <span class="settings-help" style="grid-column: 1 / -1">留空使用官方 Default 模板；若模板包含 {source_text} 则视为完整模板并直接替换占位符后使用（覆盖 Default）；否则视为附加约束拼于 Default 之前。支持批量 {format_type}。</span>
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
