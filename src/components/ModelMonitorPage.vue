<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, ref } from "vue";
import {
  NAlert,
  NButton,
  NCard,
  NEmpty,
  NProgress,
  NSpin,
  NTag,
  useMessage,
} from "naive-ui";
import {
  controlModel,
  getModelRuntimeStatus,
  type ModelAction,
  type ModelRuntimeStatus,
  type ModelTarget,
} from "../services/translation-provider";
import { applySharedModelRuntimeStatus } from "../services/workspace-settings";
import { showWorkspaceToast } from "../services/workspace-toast";

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const toast = useMessage();
const runtimeStatus = ref<ModelRuntimeStatus | null>(null);
const refreshing = ref(false);
const controlKey = ref("");
const errorMessage = ref("");
const statusObservedAtMs = ref(Date.now());
const clockNowMs = ref(Date.now());
let refreshTimer: ReturnType<typeof setInterval> | undefined;
let clockTimer: ReturnType<typeof setInterval> | undefined;

const successRate = computed(() => {
  const status = runtimeStatus.value;
  if (!status || status.requestCount === 0) {
    return 0;
  }
  return Math.round((status.succeededRequests / status.requestCount) * 100);
});

const loadedModelCount = computed(() => {
  const status = runtimeStatus.value;
  return status ? Number(status.ocrLoaded) + Number(status.translatorLoaded) : 0;
});

const idleSeconds = computed(() => {
  const status = runtimeStatus.value;
  if (!status || status.busy) {
    return 0;
  }
  const elapsedSinceObservation = Math.max(0, clockNowMs.value - statusObservedAtMs.value);
  return Math.floor((status.idleForMs + elapsedSinceObservation) / 1000);
});

const integerFormatter = new Intl.NumberFormat("zh-CN", {
  maximumFractionDigits: 0,
});

const runtimeLabel = computed(() => {
  const status = runtimeStatus.value;
  if (!status) {
    return isDesktopRuntime ? "读取中" : "桌面端可用";
  }
  if (status.busy) {
    return "处理中";
  }
  const loadedCount = Number(status.ocrLoaded) + Number(status.translatorLoaded);
  return loadedCount > 0 ? `已加载 ${loadedCount}/2` : "按需加载";
});

const runtimeTagType = computed<"default" | "success" | "warning" | "info">(() => {
  if (!runtimeStatus.value) {
    return isDesktopRuntime ? "warning" : "default";
  }
  return runtimeStatus.value.busy ? "warning" : "success";
});

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatInteger(value: number): string {
  return integerFormatter.format(Math.round(value));
}

function formatMilliseconds(value: number | null): string {
  return value === null ? "暂无" : formatInteger(value);
}

function formatEventTime(timestampMs: number): string {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(timestampMs));
}

async function refreshStatus(showError = true): Promise<void> {
  if (!isDesktopRuntime || refreshing.value || controlKey.value) {
    return;
  }
  refreshing.value = true;
  try {
    const status = await getModelRuntimeStatus();
    const observedAt = Date.now();
    runtimeStatus.value = status;
    statusObservedAtMs.value = observedAt;
    clockNowMs.value = observedAt;
    applySharedModelRuntimeStatus(status);
    errorMessage.value = "";
  } catch (error) {
    if (showError) {
      errorMessage.value = errorText(error);
    }
  } finally {
    refreshing.value = false;
  }
}

async function runModelControl(model: ModelTarget, action: ModelAction): Promise<void> {
  if (!isDesktopRuntime || controlKey.value) {
    return;
  }
  const key = `${model}:${action}`;
  controlKey.value = key;
  try {
    const status = await controlModel(model, action);
    const observedAt = Date.now();
    runtimeStatus.value = status;
    statusObservedAtMs.value = observedAt;
    clockNowMs.value = observedAt;
    applySharedModelRuntimeStatus(status);
    errorMessage.value = "";
    const modelLabel = model === "ocr" ? "PP-OCR" : "Hy-MT2";
    showWorkspaceToast(toast, "success", `${modelLabel} 已${action === "load" ? "加载" : "卸载"}。`);
  } catch (error) {
    const message = errorText(error);
    errorMessage.value = message;
    showWorkspaceToast(toast, "error", message);
  } finally {
    controlKey.value = "";
  }
}

function startPolling(): void {
  if (!isDesktopRuntime || refreshTimer || clockTimer) {
    return;
  }
  clockNowMs.value = Date.now();
  void refreshStatus();
  refreshTimer = setInterval(() => void refreshStatus(false), 2000);
  clockTimer = setInterval(() => {
    clockNowMs.value = Date.now();
  }, 1000);
}

function stopPolling(): void {
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = undefined;
  }
  if (clockTimer) {
    clearInterval(clockTimer);
    clockTimer = undefined;
  }
}

onActivated(startPolling);
onDeactivated(stopPolling);
onBeforeUnmount(stopPolling);
</script>

<template>
  <section class="model-monitor-page" aria-labelledby="model-monitor-page-title">
    <header class="monitor-toolbar">
      <div class="monitor-title-group">
        <p class="panel-kicker">Operations / Model Runtime</p>
        <h2 id="model-monitor-page-title">模型运行监控</h2>
        <p>管理本地 OCR 与 Hy-MT2 的驻留状态；运行指标和记录用于诊断。常用模型与目标语言准备请先在设置中完成。</p>
      </div>
      <div class="monitor-toolbar-actions">
        <n-tag :type="runtimeTagType" round size="small">{{ runtimeLabel }}</n-tag>
        <n-button secondary :loading="refreshing" :disabled="!isDesktopRuntime" @click="refreshStatus()">
          刷新状态
        </n-button>
      </div>
    </header>

    <n-alert v-if="!isDesktopRuntime" type="info" :show-icon="true">
      模型运行状态来自 Tauri 后端，请在桌面应用中查看和控制。
    </n-alert>
    <n-alert v-else-if="errorMessage" type="error" title="无法读取模型状态" :show-icon="true">
      {{ errorMessage }}
    </n-alert>

    <div v-if="!runtimeStatus && isDesktopRuntime" class="monitor-loading" aria-live="polite">
      <n-spin size="small" />
      <span>正在读取模型运行状态…</span>
    </div>

    <template v-if="runtimeStatus">
      <div class="runtime-summary" role="status" aria-live="polite">
        <div class="runtime-state">
          <span
            class="runtime-state-dot"
            :class="{ 'runtime-state-dot-busy': runtimeStatus.busy }"
            aria-hidden="true"
          ></span>
          <div>
            <span>Runtime 状态</span>
            <strong>{{ runtimeStatus.busy ? "正在推理" : `空闲 ${formatInteger(idleSeconds)} s` }}</strong>
          </div>
        </div>
        <dl class="runtime-facts">
          <div>
            <dt>运行设备</dt>
            <dd>{{ runtimeStatus.backend.device.toUpperCase() }}</dd>
          </div>
          <div>
            <dt>驻留模型</dt>
            <dd>{{ loadedModelCount }} / 2</dd>
          </div>
          <div>
            <dt>自动刷新</dt>
            <dd>2,000 ms</dd>
          </div>
        </dl>
      </div>

      <section class="monitor-section" aria-labelledby="loaded-models-title">
        <div class="section-heading">
          <div>
            <p class="panel-kicker">Model Residency</p>
            <h3 id="loaded-models-title">模型驻留</h3>
          </div>
          <span>推理过程中暂不可加载或卸载模型</span>
        </div>

        <div class="model-card-grid">
          <n-card
            class="monitor-card model-card"
            :class="{ 'model-card-loaded': runtimeStatus.ocrLoaded }"
            :bordered="false"
          >
            <div class="model-card-heading">
              <div class="model-identity">
                <span class="model-kind">OCR</span>
                <div>
                  <h4>PP-OCR</h4>
                  <p>文字检测与识别</p>
                </div>
              </div>
              <n-tag :type="runtimeStatus.ocrLoaded ? 'success' : 'default'" round size="small">
                {{ runtimeStatus.ocrLoaded ? "已加载" : "未加载" }}
              </n-tag>
            </div>

            <dl class="model-paths">
              <div>
                <dt>Detector</dt>
                <dd><code>{{ runtimeStatus.backend.detectorModelDir }}</code></dd>
              </div>
              <div>
                <dt>Recognizer</dt>
                <dd><code>{{ runtimeStatus.backend.recognizerModelDir }}</code></dd>
              </div>
            </dl>

            <div class="model-card-footer">
              <span>{{ runtimeStatus.ocrLoaded ? "模型已驻留，可直接执行 OCR" : "首次请求时会自动加载" }}</span>
              <n-button
                v-if="runtimeStatus.ocrLoaded"
                secondary
                type="warning"
                :loading="controlKey === 'ocr:unload'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)"
                @click="runModelControl('ocr', 'unload')"
              >
                卸载 OCR
              </n-button>
              <n-button
                v-else
                type="primary"
                :loading="controlKey === 'ocr:load'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)"
                @click="runModelControl('ocr', 'load')"
              >
                加载 OCR
              </n-button>
            </div>
          </n-card>

          <n-card
            class="monitor-card model-card"
            :class="{ 'model-card-loaded': runtimeStatus.translatorLoaded }"
            :bordered="false"
          >
            <div class="model-card-heading">
              <div class="model-identity">
                <span class="model-kind">MT</span>
                <div>
                  <h4>Hy-MT2</h4>
                  <p>本地文本翻译</p>
                </div>
              </div>
              <n-tag :type="runtimeStatus.translatorLoaded ? 'success' : 'default'" round size="small">
                {{ runtimeStatus.translatorLoaded ? "已加载" : "未加载" }}
              </n-tag>
            </div>

            <dl class="model-paths">
              <div>
                <dt>GGUF 模型</dt>
                <dd><code>{{ runtimeStatus.backend.hyModel }}</code></dd>
              </div>
              <div>
                <dt>目标语言</dt>
                <dd><code>{{ runtimeStatus.backend.targetLanguage }}</code></dd>
              </div>
            </dl>

            <div class="model-card-footer">
              <span>{{ runtimeStatus.translatorLoaded ? "模型已驻留，可直接执行翻译" : "首次请求时会自动加载" }}</span>
              <n-button
                v-if="runtimeStatus.translatorLoaded"
                secondary
                type="warning"
                :loading="controlKey === 'translator:unload'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)"
                @click="runModelControl('translator', 'unload')"
              >
                卸载 Hy-MT2
              </n-button>
              <n-button
                v-else
                type="primary"
                :loading="controlKey === 'translator:load'"
                :disabled="runtimeStatus.busy || Boolean(controlKey)"
                @click="runModelControl('translator', 'load')"
              >
                加载 Hy-MT2
              </n-button>
            </div>
          </n-card>
        </div>
      </section>

      <n-card class="monitor-card metrics-card" :bordered="false">
        <div class="section-heading">
          <div>
            <p class="panel-kicker">Runtime Metrics</p>
            <h3>运行指标</h3>
          </div>
          <span>请求耗时按毫秒统计，空闲时间按秒计时</span>
        </div>

        <div class="runtime-metrics">
          <div class="metric-tile">
            <span>请求总数</span>
            <strong>{{ formatInteger(runtimeStatus.requestCount) }}<small>次</small></strong>
          </div>
          <div class="metric-tile metric-tile-success">
            <span>成功请求</span>
            <strong>{{ formatInteger(runtimeStatus.succeededRequests) }}<small>次</small></strong>
          </div>
          <div class="metric-tile metric-tile-error">
            <span>失败请求</span>
            <strong>{{ formatInteger(runtimeStatus.failedRequests) }}<small>次</small></strong>
          </div>
          <div class="metric-tile metric-rate">
            <div>
              <span>请求成功率</span>
              <strong>{{ successRate }}<small>%</small></strong>
            </div>
            <n-progress type="line" :percentage="successRate" :show-indicator="false" status="success" />
          </div>
          <div class="metric-tile metric-tile-latency">
            <span>平均耗时</span>
            <strong>{{ formatMilliseconds(runtimeStatus.averageDurationMs) }}<small>ms</small></strong>
          </div>
          <div class="metric-tile metric-tile-latency">
            <span>最近耗时</span>
            <strong>
              {{ formatMilliseconds(runtimeStatus.lastDurationMs) }}
              <small v-if="runtimeStatus.lastDurationMs !== null">ms</small>
            </strong>
          </div>
          <div class="metric-tile metric-tile-idle">
            <span>空闲计时</span>
            <strong>{{ formatInteger(idleSeconds) }}<small>s</small></strong>
          </div>
        </div>
      </n-card>

      <n-card class="monitor-card events-card" :bordered="false">
        <div class="section-heading">
          <div>
            <p class="panel-kicker">Recent Activity</p>
            <h3>最近运行记录</h3>
          </div>
          <span>最近 {{ runtimeStatus.recentEvents.length }} 条</span>
        </div>
        <div v-if="runtimeStatus.recentEvents.length" class="event-table-wrap">
          <table class="event-table">
            <thead>
              <tr>
                <th scope="col">时间</th>
                <th scope="col">操作</th>
                <th scope="col">状态</th>
                <th scope="col">耗时（ms）</th>
                <th scope="col">说明</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(event, index) in runtimeStatus.recentEvents"
                :key="`${event.timestampMs}-${event.operation}-${index}`"
              >
                <td>{{ formatEventTime(event.timestampMs) }}</td>
                <td>{{ event.operation }}</td>
                <td>
                  <n-tag :type="event.success ? 'success' : 'error'" size="small">
                    {{ event.success ? "成功" : "失败" }}
                  </n-tag>
                </td>
                <td class="event-duration">{{ formatMilliseconds(event.durationMs) }}</td>
                <td>{{ event.message }}</td>
              </tr>
            </tbody>
          </table>
        </div>
        <n-empty v-else size="small" description="完成一次 OCR、翻译或模型控制后，这里会显示运行记录。" />
      </n-card>
    </template>
  </section>
</template>

<style scoped>
.model-monitor-page {
  display: grid;
  width: 100%;
  gap: 20px;
  margin: 24px auto 0;
  padding-bottom: 24px;
}

.monitor-toolbar,
.monitor-toolbar-actions,
.section-heading,
.model-card-heading,
.model-card-footer,
.model-identity {
  display: flex;
  align-items: center;
}

.monitor-toolbar,
.section-heading,
.model-card-heading,
.model-card-footer {
  justify-content: space-between;
}

.monitor-toolbar {
  gap: 24px;
}

.monitor-title-group {
  min-width: 0;
}

.monitor-title-group h2,
.section-heading h3 {
  margin: 4px 0 0;
  font-size: 16px;
  line-height: 1.4;
}

.monitor-title-group > p:last-child {
  max-width: 560px;
  margin: 6px 0 0;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.monitor-toolbar-actions {
  flex: 0 0 auto;
  justify-content: flex-end;
  gap: 10px;
}

.monitor-loading {
  display: flex;
  min-height: 180px;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--text-muted);
  font-size: 13px;
}

.runtime-summary {
  display: grid;
  min-height: 82px;
  grid-template-columns: minmax(220px, 0.8fr) minmax(420px, 1.6fr);
  overflow: hidden;
  border: 1px solid rgba(64, 158, 255, 0.24);
  border-radius: 6px;
  background: #f5f9ff;
}

.runtime-state {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 20px;
}

.runtime-state-dot {
  width: 10px;
  height: 10px;
  flex: 0 0 10px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 0 4px rgba(103, 194, 58, 0.14);
}

.runtime-state-dot-busy {
  background: var(--warning);
  box-shadow: 0 0 0 4px rgba(230, 162, 60, 0.16);
}

.runtime-state span,
.runtime-facts dt {
  color: var(--text-muted);
  font-size: 12px;
}

.runtime-state strong {
  display: block;
  margin-top: 3px;
  color: var(--text);
  font-size: 15px;
}

.runtime-facts {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 0;
}

.runtime-facts > div {
  display: flex;
  min-width: 0;
  flex-direction: column;
  justify-content: center;
  padding: 12px 18px;
  border-left: 1px solid rgba(64, 158, 255, 0.16);
}

.runtime-facts dd {
  margin: 5px 0 0;
  color: var(--text);
  font-size: 14px;
  font-weight: 650;
}

.monitor-section {
  display: grid;
  gap: 12px;
}

.section-heading {
  gap: 16px;
}

.section-heading > span {
  color: var(--text-muted);
  font-size: 12px;
  text-align: right;
}

.model-card-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.monitor-card {
  min-width: 0;
  border: 1px solid var(--divider);
  border-radius: 6px;
  background: var(--surface);
  box-shadow: 0 1px 2px rgba(48, 49, 51, 0.02);
}

.model-card-loaded {
  border-color: rgba(103, 194, 58, 0.34);
}

.model-card :deep(.n-card__content),
.metrics-card :deep(.n-card__content),
.events-card :deep(.n-card__content) {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.model-card-heading {
  align-items: flex-start;
  gap: 16px;
}

.model-identity {
  min-width: 0;
  gap: 12px;
}

.model-kind {
  display: grid;
  width: 36px;
  height: 36px;
  flex: 0 0 36px;
  place-items: center;
  border: 1px solid rgba(64, 158, 255, 0.24);
  border-radius: 4px;
  color: var(--primary);
  background: #ecf5ff;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.04em;
}

.model-identity h4 {
  margin: 0;
  color: var(--text);
  font-size: 15px;
  line-height: 1.4;
}

.model-identity p {
  margin: 3px 0 0;
  color: var(--text-muted);
  font-size: 12px;
}

.model-paths {
  display: grid;
  gap: 10px;
  margin: 0;
}

.model-paths > div {
  display: grid;
  grid-template-columns: 90px minmax(0, 1fr);
  align-items: center;
  gap: 10px;
}

.model-paths dt {
  color: var(--text-muted);
  font-size: 12px;
}

.model-paths dd {
  min-width: 0;
  margin: 0;
}

.model-paths code {
  display: block;
  padding: 8px 10px;
  overflow: hidden;
  border: 1px solid var(--divider);
  border-radius: 4px;
  color: var(--text-soft);
  background: var(--surface-soft);
  font-size: 12px;
  line-height: 1.5;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-card-footer {
  min-height: 44px;
  gap: 12px;
  margin: auto -16px -16px;
  padding: 10px 16px;
  border-top: 1px solid var(--divider);
  background: var(--surface-soft);
}

.model-card-footer > span {
  color: var(--text-muted);
  font-size: 12px;
}

.model-card-footer .n-button {
  flex: 0 0 auto;
}

.runtime-metrics {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
}

.metric-tile {
  display: flex;
  min-height: 86px;
  flex-direction: column;
  justify-content: space-between;
  padding: 13px 14px;
  border: 1px solid var(--divider);
  border-radius: 4px;
  background: var(--surface-soft);
}

.metric-tile > span,
.metric-rate span {
  color: var(--text-muted);
  font-size: 12px;
}

.metric-tile strong {
  display: block;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  font-size: 22px;
  font-weight: 650;
  line-height: 1.2;
}

.metric-tile strong small {
  margin-left: 5px;
  color: var(--text-muted);
  font-size: 11px;
  font-weight: 500;
}

.metric-tile-success {
  background: #f6fbf3;
}

.metric-tile-error {
  background: #fef7f7;
}

.metric-tile-latency {
  border-color: rgba(64, 158, 255, 0.18);
  background: #f5f9ff;
}

.metric-rate {
  gap: 10px;
}

.metric-rate > div {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
}

.event-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--divider);
  border-radius: 4px;
}

.event-table {
  width: 100%;
  border-collapse: collapse;
  color: var(--text-soft);
  font-size: 12px;
  text-align: left;
}

.event-table th,
.event-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--divider);
  white-space: nowrap;
}

.event-table th {
  color: var(--text-muted);
  background: var(--surface-soft);
  font-weight: 600;
}

.event-table td:last-child {
  width: 100%;
  white-space: normal;
}

.event-table tbody tr:last-child td {
  border-bottom: 0;
}

.event-duration {
  font-variant-numeric: tabular-nums;
  text-align: right;
}

@media (max-width: 1100px) {
  .runtime-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 900px) {
  .runtime-summary {
    grid-template-columns: 1fr;
  }

  .runtime-facts > div:first-child {
    border-left: 0;
  }

  .runtime-facts {
    border-top: 1px solid rgba(64, 158, 255, 0.16);
  }

  .model-card-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .model-monitor-page {
    margin-top: 16px;
  }

  .monitor-toolbar {
    align-items: flex-start;
    flex-direction: column;
  }

  .monitor-toolbar-actions {
    width: 100%;
    justify-content: space-between;
  }

  .monitor-toolbar-actions .n-button,
  .model-card-footer .n-button {
    min-height: 44px;
  }

  .section-heading {
    align-items: flex-start;
    flex-direction: column;
    gap: 6px;
  }

  .section-heading > span {
    text-align: left;
  }

  .model-paths > div {
    grid-template-columns: 1fr;
    gap: 5px;
  }

  .model-card-footer {
    align-items: stretch;
    flex-direction: column;
  }

  .runtime-metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 480px) {
  .runtime-facts,
  .runtime-metrics {
    grid-template-columns: 1fr;
  }

  .runtime-facts > div {
    border-top: 1px solid rgba(64, 158, 255, 0.16);
    border-left: 0;
  }

  .runtime-facts > div:first-child {
    border-top: 0;
  }
}
</style>
