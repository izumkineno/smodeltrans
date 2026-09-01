<script setup lang="ts">
import { onMounted, ref, computed } from "vue";
import { NCard, NSwitch, NInput, NInputNumber, NButton, NAlert, NTag, useMessage } from "naive-ui";
import { getOpenAiStatus, updateOpenAiConfig, buildBaseUrl } from "../services/openai-compat";
import type { OpenAiStatus } from "../services/openai-compat";

const message = useMessage();
const loading = ref(false);
const saving = ref(false);
const status = ref<OpenAiStatus | null>(null);
const enabled = ref(false);
const host = ref("127.0.0.1");
const port = ref(11438);
const apiKey = ref("");
const feedback = ref("");
const feedbackType = ref<"success" | "error" | "info">("info");

const baseUrl = computed(() => {
  if (!status.value) return `http://${host.value}:${port.value}`;
  return buildBaseUrl({ host: status.value.host, port: status.value.port, boundPort: status.value.boundPort });
});

const statusTagType = computed(() => {
  if (!status.value) return "default" as const;
  if (!status.value.enabled) return "default" as const;
  return status.value.running ? "success" as const : "error" as const;
});

async function refresh() {
  loading.value = true;
  try {
    const s = await getOpenAiStatus();
    status.value = s;
    enabled.value = s.enabled;
    host.value = s.host;
    port.value = s.port;
    // apiKey 不回显，保留输入框空白
  } catch (e) {
    feedback.value = e instanceof Error ? e.message : String(e);
    feedbackType.value = "error";
  } finally {
    loading.value = false;
  }
}

async function save() {
  saving.value = true;
  feedback.value = "";
  try {
    if (port.value < 1 || port.value > 65535) throw new Error("端口需在 1..65535");
    if (!host.value.trim()) throw new Error("host 不能为空");
    const s = await updateOpenAiConfig({
      enabled: enabled.value,
      host: host.value.trim(),
      port: port.value,
      apiKey: apiKey.value.trim() ? apiKey.value.trim() : null,
    });
    status.value = s;
    feedback.value = s.message || "已保存";
    feedbackType.value = "success";
    message.success("OpenAI 兼容服务已更新");
  } catch (e) {
    feedback.value = e instanceof Error ? e.message : String(e);
    feedbackType.value = "error";
    message.error(feedback.value);
  } finally {
    saving.value = false;
  }
}

async function copyBaseUrl() {
  try {
    await navigator.clipboard.writeText(baseUrl.value + "/v1");
    message.success(`已复制 ${baseUrl.value}/v1`);
  } catch {
    message.warning(baseUrl.value + "/v1");
  }
}

onMounted(refresh);
</script>

<template>
  <n-card class="settings-card settings-card-wide" :bordered="false">
    <div class="settings-card-heading">
      <div>
        <h3 class="settings-card-title">OpenAI 兼容服务</h3>
        <p class="settings-card-subtitle">将本地 Hy-MT2 以 OpenAI API 暴露给其他应用（独立文件夹、不耦合）</p>
      </div>
      <n-tag :type="statusTagType" size="small">{{ status ? status.message : "加载中" }}</n-tag>
    </div>
    <p class="settings-card-copy" style="max-width: none;">
      启用后监听 <code>{{ baseUrl }}</code>，提供 <code>GET /v1/models</code>、<code>POST /v1/chat/completions</code>（支持 stream）与 <code>GET /health</code>。
      鉴权可选：若设置 API Key，外部需 <code>Authorization: Bearer &lt;key&gt;</code>。
    </p>

    <div class="settings-field-grid">
      <label class="settings-field">
        <span>启用服务</span>
        <n-switch v-model:value="enabled" />
        <span class="settings-help">关闭则不监听端口</span>
      </label>

      <label class="settings-field">
        <span>Host</span>
        <n-input v-model:value="host" placeholder="127.0.0.1" />
        <span class="settings-help">仅建议 127.0.0.1（本地回环）</span>
      </label>

      <label class="settings-field settings-number-field">
        <span>端口</span>
        <n-input-number v-model:value="port" :min="1" :max="65535" style="width: 100%" />
        <span class="settings-help">被占用时自动尝试 port+1..+3</span>
      </label>

      <label class="settings-field settings-field-wide">
        <span>API Key（可选）</span>
        <n-input v-model:value="apiKey" type="password" placeholder="留空则不鉴权" show-password-on="click" />
        <span class="settings-help">设置后外部需 Bearer 鉴权；已保存的 Key 不回显</span>
      </label>
    </div>

    <div style="display:flex; gap:12px; flex-wrap:wrap; margin-top:16px;">
      <n-button secondary :loading="loading" @click="refresh">刷新状态</n-button>
      <n-button type="primary" :loading="saving" @click="save">保存并重启服务</n-button>
      <n-button @click="copyBaseUrl">复制 Base URL</n-button>
      <span style="align-self:center; color: var(--n-text-color); font-size: 12px;">{{ baseUrl }}/v1</span>
    </div>

    <n-alert v-if="feedback" :type="feedbackType" style="margin-top:12px;" :show-icon="false">{{ feedback }}</n-alert>
  </n-card>
</template>
