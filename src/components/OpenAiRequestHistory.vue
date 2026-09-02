<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { NButton, NCard, NEmpty, NScrollbar, NSpin, NTag, useMessage } from "naive-ui";
import {
  clearOpenAiHistory,
  getOpenAiHistory,
  type OpenAiHistoryEntry,
} from "../services/openai-compat";

const message = useMessage();
const history = ref<OpenAiHistoryEntry[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);

const hasHistory = computed(() => history.value.length > 0);

function formatTime(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function truncate(text: string, max = 300): string {
  if (text.length <= max) return text;
  return text.slice(0, max) + " …";
}

async function refresh() {
  loading.value = true;
  error.value = null;
  try {
    history.value = await getOpenAiHistory();
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

async function handleClear() {
  loading.value = true;
  try {
    await clearOpenAiHistory();
    history.value = [];
    message.success("已清空远程翻译历史");
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e);
  } finally {
    loading.value = false;
  }
}

function handleCopy(text: string) {
  navigator.clipboard
    .writeText(text)
    .then(() => message.success("已复制"))
    .catch(() => message.error("复制失败"));
}

onMounted(() => {
  refresh();
});

defineExpose({ refresh });
</script>

<template>
  <n-card class="settings-card settings-card-wide" :bordered="false">
    <div class="settings-card-heading">
      <h3 class="settings-card-title">远程翻译历史</h3>
      <span class="settings-card-subtitle">复用 Hy-MT2 官方模板的远程请求记录（最近 100 条）</span>
    </div>
    <p class="settings-card-copy" style="margin-bottom: 12px">
      所有远程请求经 <code>POST /v1/chat/completions</code> 复用本地
      <code>build_translation_prompt</code> 官方模板， <code>system</code> 仅作
      <code>Additional requirements</code> 附加约束；此处仅记录已验证的翻译结果。
    </p>

    <div class="openai-history-actions">
      <n-button size="small" :loading="loading" @click="refresh">刷新</n-button>
      <n-button size="small" :disabled="!hasHistory || loading" @click="handleClear">清空</n-button>
      <span v-if="error" class="openai-history-error">{{ error }}</span>
      <span v-if="!error" class="openai-history-meta">{{ hasHistory ? `${history.length} 条记录` : "" }}</span>
    </div>

    <n-spin :show="loading">
      <n-empty v-if="!loading && !hasHistory" description="暂无远程翻译历史" style="padding: 24px 0" />
      <n-scrollbar v-else style="max-height: 440px" :x-scrollable="false">
        <div class="openai-history-list">
          <div v-for="entry in history" :key="entry.id" class="openai-history-item">
            <div class="openai-history-item-head">
              <div class="openai-history-tags">
                <n-tag size="small" :bordered="false" type="info">{{ entry.model }}</n-tag>
                <n-tag size="small" :bordered="false">{{ entry.targetLanguage }}</n-tag>
                <n-tag v-if="entry.streaming" size="small" :bordered="false" type="success">stream</n-tag>
                <span class="openai-history-time">{{ formatTime(entry.timestampMs) }}</span>
              </div>
              <span class="openai-history-stats">{{ entry.durationMs }}ms · {{ entry.promptTokens }}/{{ entry.completionTokens }} tokens</span>
            </div>
            <div class="openai-history-grid">
              <div class="openai-history-field">
                <div class="openai-history-field-label">原文</div>
                <div class="openai-history-field-box">{{ truncate(entry.sourceText, 300) }}</div>
                <n-button size="tiny" secondary class="openai-history-copy" @click="handleCopy(entry.sourceText)">复制原文</n-button>
              </div>
              <div class="openai-history-field">
                <div class="openai-history-field-label">译文</div>
                <div class="openai-history-field-box openai-history-field-box--translated">{{ truncate(entry.translatedText, 300) }}</div>
                <n-button size="tiny" secondary class="openai-history-copy" @click="handleCopy(entry.translatedText)">复制译文</n-button>
              </div>
            </div>
          </div>
        </div>
      </n-scrollbar>
    </n-spin>
  </n-card>
</template>

<style scoped>
.openai-history-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.openai-history-error {
  color: var(--error);
  font-size: 12px;
  line-height: 28px;
}
.openai-history-meta {
  color: var(--text-muted);
  font-size: 12px;
  margin-left: 4px;
}
.openai-history-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-right: 8px;
}
.openai-history-item {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  background: var(--surface);
  transition: border-color 0.15s, box-shadow 0.15s;
}
.openai-history-item:hover {
  border-color: var(--border-strong);
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
}
[data-theme="dark"] .openai-history-item:hover {
  box-shadow: 0 1px 8px rgba(0, 0, 0, 0.35);
}
.openai-history-item-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.openai-history-tags {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-wrap: wrap;
}
.openai-history-time {
  font-size: 11px;
  color: var(--text-muted);
}
.openai-history-stats {
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
}
.openai-history-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}
@media (max-width: 720px) {
  .openai-history-grid {
    grid-template-columns: 1fr;
  }
}
.openai-history-field-label {
  font-size: 12px;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.openai-history-field-box {
  white-space: pre-wrap;
  word-break: break-all;
  background: var(--surface-soft);
  border: 1px solid var(--divider);
  padding: 8px 10px;
  border-radius: 6px;
  min-height: 42px;
  font-size: 13px;
  line-height: 1.6;
  color: var(--text);
}
.openai-history-field-box--translated {
  background: var(--surface-info);
  border-color: var(--border-info);
}
.openai-history-copy {
  margin-top: 8px;
}
</style>
