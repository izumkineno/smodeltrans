<script setup lang="ts">
import { computed, onActivated, onBeforeUnmount, onDeactivated, ref } from "vue";
import type { UnlistenFn } from "@tauri-apps/api/event";
import {
  NAlert,
  NButton,
  NCard,
  NEmpty,
  NInput,
  NInputNumber,
  NSelect,
  NSpin,
  NSwitch,
  NTag,
  NScrollbar,
  useMessage,
} from "naive-ui";
import {
  LIVE_MEMORY_TOKENS_MAX,
  LIVE_MEMORY_TOKENS_MIN,
  LIVE_MEMORY_TURNS_MAX,
  LIVE_MEMORY_TURNS_MIN,
  LIVE_SUPPLEMENTAL_PROMPT_MAX_CHARS,
  beginLiveRoiUpdate,
  startLiveSession,
  cancelLiveSelection,
  getLiveSessionStatus,
  listCaptureWindows,
  listenLiveDebugRecord,
  listenLiveStatus,
  pauseLiveSession,
  resumeLiveSession,
  stopLiveSession,
  setLiveRegionBoxesVisible,
} from "../services/live-translation-provider";
import type {
  CaptureWindowInfo,
  LiveDebugOutcome,
  LiveDebugRecord,
  LiveMetrics,
  LiveSessionState,
  LiveSessionStatus,
} from "../services/live-translation-provider";
import {
  liveOverlaySettings,
  liveSubtitleStyleSettings,
  liveRecognitionSettings,
  liveTranslationSettings,
  KEY_TRIGGER_TIMEOUT_MAX_MS,
  KEY_TRIGGER_TIMEOUT_MIN_MS,
  LIVE_STABILITY_WAIT_MAX_MS,
  LIVE_STABILITY_WAIT_MIN_MS,
  loadPersistedLiveOverlaySettings,
  loadPersistedLiveSubtitleStyleSettings,
  loadPersistedLiveRecognitionSettings,
  loadPersistedLiveTranslationSettings,
  loadPersistedTargetLanguage,
  savePersistedLiveOverlaySettings,
  savePersistedLiveSubtitleStyleSettings,
  LIVE_SUBTITLE_FONT_SIZE_MIN,
  LIVE_SUBTITLE_FONT_SIZE_MAX,
  LIVE_SUBTITLE_BACKGROUND_OPACITY_MIN,
  LIVE_SUBTITLE_BACKGROUND_OPACITY_MAX,
  savePersistedLiveRecognitionSettings,
  savePersistedLiveTranslationSettings,
  savePersistedTargetLanguage,
  targetLanguage,
} from "../services/workspace-settings";
import { showWorkspaceToast } from "../services/workspace-toast";

type TagType = "default" | "success" | "warning" | "error" | "info";
type LiveAction = "refresh" | "start" | "pause" | "resume" | "reselect" | "cancel" | "stop";

const MAX_DEBUG_RECORDS = 200;
const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const toast = useMessage();
const captureWindows = ref<CaptureWindowInfo[]>([]);
const selectedTargetId = ref<string | null>(null);
const windowsLoaded = ref(false);
const activeAction = ref<LiveAction | null>(null);
const commandError = ref("");
const debugRecords = ref<LiveDebugRecord[]>([]);
let debugSessionId: string | undefined;
let statusUnlisten: UnlistenFn | undefined;
let debugUnlisten: UnlistenFn | undefined;
let bindingVersion = 0;
const isCapturingTriggerKey = ref(false);
const triggerKeyCaptureHint = ref("");
const showAdvancedConfig = ref(false);
const showDiagnostics = ref(false);

function emptyMetrics(): LiveMetrics {
  return {
    framesCaptured: 0,
    framesDropped: 0,
    framesSkippedUnchanged: 0,
    ocrRuns: 0,
    translationRuns: 0,
    subtitlePublishes: 0,
    lastOcrMs: 0,
    lastTranslationMs: 0,
    gpuName: "",
    gpuTotalMemoryMib: 0,
    gpuFreeMemoryMib: 0,
    gpuExecutionMode: "",
  };
}

const liveOverlayModeOptions = [
  { label: "字幕框：附着在目标窗口外侧", value: "subtitle" },
  { label: "逐区坐标替换：覆盖窗口 1:1 贴合客户区", value: "region_replace" },
];
const liveOverlayAttachmentOptions = [
  { label: "上边", value: "top" },
  { label: "下边", value: "bottom" },
  { label: "左边", value: "left" },
  { label: "右边", value: "right" },
];

const liveRecognitionModeOptions = [
  { label: "自动识别：按字幕变化自动 OCR", value: "automatic" },
  { label: "按键触发：每次按下或松开只 OCR 一次", value: "key_trigger" },
];
const liveRecognitionTriggerOptions = [
  { label: "按下时触发", value: "press" },
  { label: "松开时触发", value: "release" },
];
const liveTriggerKeyLabels: Record<string, string> = {
  Escape: "Esc",
  Tab: "Tab",
  CapsLock: "Caps Lock",
  ShiftLeft: "左 Shift",
  ShiftRight: "右 Shift",
  ControlLeft: "左 Ctrl",
  ControlRight: "右 Ctrl",
  AltLeft: "左 Alt",
  AltRight: "右 Alt",
  MetaLeft: "左 Win",
  MetaRight: "右 Win",
  ContextMenu: "菜单键",
  Space: "空格",
  Enter: "Enter",
  Backspace: "Backspace",
  Insert: "Insert",
  Delete: "Delete",
  Home: "Home",
  End: "End",
  PageUp: "Page Up",
  PageDown: "Page Down",
  ArrowUp: "↑",
  ArrowDown: "↓",
  ArrowLeft: "←",
  ArrowRight: "→",
  PrintScreen: "Print Screen",
  ScrollLock: "Scroll Lock",
  Pause: "Pause",
  NumLock: "Num Lock",
};

const liveStatus = ref<LiveSessionStatus>({
  state: "idle",
  message: isDesktopRuntime ? "请选择目标窗口并开始实时翻译。" : "实时翻译仅在 Windows 桌面端可用。",
  latestRevision: 0,
  metrics: emptyMetrics(),
});

const windowOptions = computed(() =>
  captureWindows.value.map((target) => ({
    label: `${target.title || "未命名窗口"} — ${target.processName} · ${
      target.isMinimized ? "已最小化，可直接恢复" : `${target.width}×${target.height}`
    }`,
    value: target.id,
  })),
);

const selectedTarget = computed(() =>
  captureWindows.value.find((target) => target.id === selectedTargetId.value) ?? liveStatus.value.target,
);

const canConfigure = computed(
  () => liveStatus.value.state === "idle" || liveStatus.value.state === "error",
);
const canStart = computed(
  () =>
    isDesktopRuntime &&
    canConfigure.value &&
    activeAction.value === null &&
    selectedTargetId.value !== null &&
    targetLanguage.value.trim().length > 0,
);
const hasActiveSession = computed(
  () => liveStatus.value.state !== "idle" && liveStatus.value.state !== "error",
);

const stateLabel = computed(() => {
  const labels: Record<LiveSessionState, string> = {
    idle: "未启动",
    selecting: "选择字幕区域",
    warming: "模型预热",
    running: "实时运行",
    paused: "已暂停",
    stopping: "正在停止",
    error: "运行错误",
  };
  return labels[liveStatus.value.state];
});

const stateTagType = computed<TagType>(() => {
  switch (liveStatus.value.state) {
    case "running":
      return "success";
    case "selecting":
    case "warming":
    case "stopping":
      return "warning";
    case "paused":
      return "info";
    case "error":
      return "error";
    default:
      return "default";
  }
});

const roiLabel = computed(() => {
  const roi = liveStatus.value.roi;
  return roi ? `${roi.x}, ${roi.y} · ${roi.width}×${roi.height} px` : "尚未选择";
});

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function normalizeRecordedTriggerKey(event: KeyboardEvent): string | null {
  const virtualKey = event.keyCode || event.which;
  if (!Number.isInteger(virtualKey) || virtualKey <= 0 || virtualKey > 0xff) {
    return null;
  }
  const code = event.code.trim();
  return code && code !== "Unidentified"
    ? `vk:${virtualKey}|${code}`
    : `vk:${virtualKey}`;
}

function formatRecordedTriggerLabel(value: string): string {
  if (/^Key[A-Z]$/.test(value)) {
    return value.slice(3);
  }
  if (/^Digit[0-9]$/.test(value)) {
    return value.slice(5);
  }
  if (/^Numpad[0-9]$/.test(value)) {
    return `数字键盘 ${value.slice(6)}`;
  }
  if (/^F(?:[1-9]|1[0-9]|2[0-4])$/.test(value)) {
    return value;
  }
  return liveTriggerKeyLabels[value] ?? value;
}

function formatTriggerKey(value: string): string {
  const key = value.trim();
  if (!key) {
    return "尚未录入";
  }
  const recordedKey = /^vk:([0-9]+)(?:\|(.+))?$/.exec(key);
  if (recordedKey) {
    return recordedKey[2] ? formatRecordedTriggerLabel(recordedKey[2]) : `VK ${recordedKey[1]}`;
  }
  return formatRecordedTriggerLabel(key);
}

function stopTriggerKeyCapture(): void {
  if (typeof window !== "undefined") {
    window.removeEventListener("keydown", handleTriggerKeyCapture, true);
  }
  isCapturingTriggerKey.value = false;
}

function handleTriggerKeyCapture(event: KeyboardEvent): void {
  event.preventDefault();
  event.stopPropagation();
  const key = normalizeRecordedTriggerKey(event);
  if (!key) {
    triggerKeyCaptureHint.value =
      "当前环境没有返回 Windows virtual-key code，请在 Windows 桌面端重新录入。";
    return;
  }
  liveRecognitionSettings.value.triggerKey = key;
  triggerKeyCaptureHint.value = `已录入 ${formatTriggerKey(key)}，保存识别设置后生效。`;
  stopTriggerKeyCapture();
}


function toggleTriggerKeyCapture(): void {
  if (isCapturingTriggerKey.value) {
    stopTriggerKeyCapture();
    triggerKeyCaptureHint.value = "已取消按键录入。";
    return;
  }
  if (typeof window === "undefined") {
    return;
  }
  triggerKeyCaptureHint.value = "请按下要作为触发键的按键……";
  isCapturingTriggerKey.value = true;
  window.addEventListener("keydown", handleTriggerKeyCapture, true);
}

function applyStatus(status: LiveSessionStatus): void {
  if (status.sessionId && status.sessionId !== debugSessionId) {
    debugSessionId = status.sessionId;
    debugRecords.value = [];
  }
  liveStatus.value = status;
  if (status.target) {
    selectedTargetId.value = status.target.id;
  }
  if (status.state !== "error") {
    commandError.value = "";
  }
}

function applyDebugRecord(record: LiveDebugRecord): void {
  if (record.sessionId !== debugSessionId) {
    return;
  }
  debugRecords.value = [record, ...debugRecords.value].slice(0, MAX_DEBUG_RECORDS);
}


function setCommandError(error: unknown): void {
  const message = errorText(error);
  commandError.value = message;
  showWorkspaceToast(toast, "error", message);
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(value);
}

function formatDuration(value: number): string {
  return value > 0 ? `${formatInteger(value)} ms` : "暂无";
}

function gpuExecutionModeLabel(value: string): string {
  const labels: Record<string, string> = {
    cpu: "CPU",
    gpu_resident: "GPU 常驻",
    gpu_balanced: "GPU 均衡",
    gpu_constrained: "GPU 分层",
  };
  return labels[value] ?? "未初始化";
}

function formatDebugTime(epochMillis: number): string {
  const time = new Date(epochMillis);
  const formatted = new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(time);
  return `${formatted}.${String(time.getMilliseconds()).padStart(3, "0")}`;
}


function debugOutcomeLabel(outcome: LiveDebugOutcome): string {
  const labels: Record<LiveDebugOutcome, string> = {
    confirmed: "识别已确认",
    completed: "翻译完成",
    skipped_empty_source: "空文本跳过",
    failed: "处理失败",
  };
  return labels[outcome];
}

async function refreshWindows(notifyOnError = true): Promise<void> {
  if (!isDesktopRuntime || activeAction.value || !canConfigure.value) {
    return;
  }
  activeAction.value = "refresh";
  try {
    const windows = await listCaptureWindows();
    captureWindows.value = windows;
    windowsLoaded.value = true;
    if (!windows.some((target) => target.id === selectedTargetId.value)) {
      selectedTargetId.value = null;
    }
    commandError.value = "";
  } catch (error) {
    if (notifyOnError) {
      setCommandError(error);
    }
  } finally {
    activeAction.value = null;
  }
}

function refreshWindowsOnTargetWindowInteraction(): void {
  void refreshWindows(false);
}

async function startSession(): Promise<void> {
  const targetId = selectedTargetId.value;
  const language = targetLanguage.value.trim();
  if (!canStart.value || !targetId || !language) {
    return;
  }
  stopTriggerKeyCapture();
  activeAction.value = "start";
  try {
    loadPersistedLiveOverlaySettings();
    loadPersistedLiveSubtitleStyleSettings();
    loadPersistedLiveRecognitionSettings();
    loadPersistedLiveTranslationSettings();
    targetLanguage.value = language;
    const persistenceError = savePersistedTargetLanguage();
    if (persistenceError) {
      showWorkspaceToast(toast, "warning", persistenceError);
    }
    const recognitionPersistenceError = savePersistedLiveRecognitionSettings();
    if (recognitionPersistenceError) {
      showWorkspaceToast(toast, "warning", recognitionPersistenceError);
    }
    const translationPersistenceError = savePersistedLiveTranslationSettings();
    if (translationPersistenceError) {
      showWorkspaceToast(toast, "warning", translationPersistenceError);
    }
    applyStatus(
      await startLiveSession(
        targetId,
        language,
        liveOverlaySettings.value,
        liveRecognitionSettings.value,
        liveTranslationSettings.value,
      ),
    );
  } catch (error) {
    setCommandError(error);
  } finally {
    activeAction.value = null;
  }
}

function saveLiveOverlayPreferences(): void {
  const persistError = savePersistedLiveOverlaySettings();
  if (persistError) {
    showWorkspaceToast(toast, "error", persistError);
    return;
  }
  const stylePersistenceError = savePersistedLiveSubtitleStyleSettings();
  if (stylePersistenceError) {
    showWorkspaceToast(toast, "error", stylePersistenceError);
    return;
  }
  showWorkspaceToast(toast, "success", "实时显示设置已保存，下次开始抓取时生效。");
}
async function toggleLiveRegionBoxes(): Promise<void> {
  const visible = !liveOverlaySettings.value.showRegionBoxes;
  liveOverlaySettings.value.showRegionBoxes = visible;
  const persistError = savePersistedLiveOverlaySettings();
  if (isDesktopRuntime) {
    try {
      await setLiveRegionBoxesVisible(visible);
    } catch (error) {
      setCommandError(error);
      return;
    }
  }
  if (persistError) {
    showWorkspaceToast(toast, "warning", persistError);
    return;
  }
  showWorkspaceToast(toast, "success", visible ? "已显示译文框体。" : "已隐藏译文框体。");
}


function saveLiveRecognitionPreferences(): void {
  const persistError = savePersistedLiveRecognitionSettings();
  if (persistError) {
    showWorkspaceToast(toast, "error", persistError);
    return;
  }
  showWorkspaceToast(toast, "success", "实时识别设置已保存，下次开始抓取时生效。");
}

function saveLiveTranslationPreferences(): void {
  const persistError = savePersistedLiveTranslationSettings();
  if (persistError) {
    showWorkspaceToast(toast, "error", persistError);
    return;
  }
  showWorkspaceToast(toast, "success", "实时翻译补充提示已保存，下次开始抓取时生效。");
}

async function runSessionAction(action: Exclude<LiveAction, "refresh" | "start">): Promise<void> {
  if (!isDesktopRuntime || activeAction.value) {
    return;
  }
  const sessionId = liveStatus.value.sessionId;
  if (action !== "stop" && !sessionId) {
    setCommandError(new Error("当前实时会话标识不可用，请刷新状态后重试。"));
    return;
  }
  activeAction.value = action;
  try {
    let status: LiveSessionStatus;
    if (action === "pause") {
      status = await pauseLiveSession(sessionId!);
    } else if (action === "resume") {
      status = await resumeLiveSession(sessionId!);
    } else if (action === "reselect") {
      status = await beginLiveRoiUpdate(sessionId!);
    } else if (action === "cancel") {
      status = await cancelLiveSelection(sessionId!);
    } else {
      status = await stopLiveSession(sessionId);
    }
    applyStatus(status);
  } catch (error) {
    setCommandError(error);
  } finally {
    activeAction.value = null;
  }
}

function stopStatusBinding(): void {
  bindingVersion += 1;
  statusUnlisten?.();
  debugUnlisten?.();
  statusUnlisten = undefined;
  debugUnlisten = undefined;
}

async function startStatusBinding(): Promise<void> {
  if (!isDesktopRuntime) {
    return;
  }
  stopStatusBinding();
  const version = bindingVersion;
  let nextStatusUnlisten: UnlistenFn | undefined;
  let nextDebugUnlisten: UnlistenFn | undefined;
  try {
    nextStatusUnlisten = await listenLiveStatus(applyStatus);
    if (version !== bindingVersion) {
      nextStatusUnlisten();
      return;
    }
    nextDebugUnlisten = await listenLiveDebugRecord(applyDebugRecord);
    if (version !== bindingVersion) {
      nextStatusUnlisten();
      nextDebugUnlisten();
      return;
    }
    statusUnlisten = nextStatusUnlisten;
    debugUnlisten = nextDebugUnlisten;
    applyStatus(await getLiveSessionStatus());
    if (canConfigure.value) {
      await refreshWindows(false);
    }
  } catch (error) {
    nextStatusUnlisten?.();
    nextDebugUnlisten?.();
    if (version === bindingVersion) {
      setCommandError(error);
    }
  }
}

function cleanupPage(): void {
  stopTriggerKeyCapture();
  stopStatusBinding();
}

onActivated(() => {
  loadPersistedTargetLanguage();
  loadPersistedLiveOverlaySettings();
  loadPersistedLiveSubtitleStyleSettings();
  loadPersistedLiveRecognitionSettings();
  loadPersistedLiveTranslationSettings();
  void startStatusBinding();
});
onDeactivated(cleanupPage);
onBeforeUnmount(cleanupPage);
</script>

<template>
  <section class="live-translation-page" aria-labelledby="live-translation-page-title">
    <p class="sr-only" aria-live="polite">{{ liveStatus.message }}</p>

    <header class="live-page-intro">
      <div>
        <p class="panel-kicker">Windows Graphics Capture</p>
        <h2 id="live-translation-page-title">窗口字幕实时翻译</h2>
        <p>选择一个窗口，默认翻译整个目标客户区，可在字幕工具栏中手动框选区域。</p>
      </div>
      <n-tag :type="stateTagType" round>{{ stateLabel }}</n-tag>
    </header>

    <n-alert v-if="!isDesktopRuntime" type="info" title="桌面端功能" :show-icon="true">
      浏览器预览仅展示完整操作界面。窗口捕获、区域选择和字幕浮层需要在 Windows Tauri 桌面端运行。
    </n-alert>

    <section class="live-primary-section" aria-labelledby="live-primary-section-title">
      <div class="live-section-heading">
        <div>
          <p class="panel-kicker">Primary workflow</p>
          <h3 id="live-primary-section-title">开始实时翻译</h3>
          <p>先选择目标窗口与目标语言，再开始实时会话；运行中的暂停、重选区和停止操作集中在会话控制中。</p>
        </div>
      </div>
    <div class="live-workspace-grid" :class="{ 'has-active-session': hasActiveSession }">
      <n-card class="live-card" :bordered="false">
        <div class="card-heading">
          <div>
            <span class="step-index">01</span>
            <div>
              <h3>捕获目标</h3>
              <p>选择一个窗口；开始后默认捕获整个客户区，最小化窗口会自动恢复。</p>
            </div>
          </div>
          <n-button
            secondary
            size="small"
            :loading="activeAction === 'refresh'"
            :disabled="!isDesktopRuntime || !canConfigure || activeAction !== null"
            @click="refreshWindows()"
          >
            刷新窗口
          </n-button>
        </div>

        <div class="live-form">
          <label class="live-field">
            <span>目标窗口</span>
            <n-select
              v-model:value="selectedTargetId"
              :options="windowOptions"
              :disabled="!isDesktopRuntime || !canConfigure"
              filterable
              placeholder="请选择要捕获的窗口"
              aria-label="实时翻译目标窗口"
              @focus="refreshWindowsOnTargetWindowInteraction"
              @click="refreshWindowsOnTargetWindowInteraction"
            />
            <span class="live-settings-note">点击或聚焦目标窗口时自动刷新，也可使用右侧“刷新窗口”按钮。</span>
          </label>
          <n-empty
            v-if="isDesktopRuntime && windowsLoaded && captureWindows.length === 0"
            size="small"
            description="没有找到可用于捕获的窗口。请打开目标程序，然后刷新列表。"
          />
          <label class="live-field">
            <span>目标语言</span>
            <n-input
              v-model:value="targetLanguage"
              maxlength="64"
              :disabled="!canConfigure"
              placeholder="例如：Chinese、English、Japanese"
              aria-label="实时翻译目标语言"
            />
          </label>
        </div>

        <div v-if="selectedTarget" class="target-summary">
          <div>
            <strong>{{ selectedTarget.title || "未命名窗口" }}</strong>
            <span>{{ selectedTarget.processName }} · PID {{ selectedTarget.processId }}</span>
          </div>
          <code>{{
            selectedTarget.isMinimized ? "已最小化（开始时自动恢复）" : `${selectedTarget.width}×${selectedTarget.height}`
          }}</code>
        </div>

        <n-alert type="warning" title="捕获模式" :show-icon="true">
          请将游戏或应用切换为<strong>无边框全屏</strong>或<strong>窗口模式</strong>。独占全屏与受保护内容可能无法提供画面。
        </n-alert>
        <n-alert type="info" title="PP-OCR 识别边界" :show-icon="true">
          当前 server recognizer 不覆盖韩文；P1 建议用于中文、英文、日文与常见拉丁字符字幕。目标语言只控制 Hy-MT2 的翻译输出。
        </n-alert>

        <n-button type="primary" :loading="activeAction === 'start'" :disabled="!canStart" @click="startSession">
          开始实时翻译
        </n-button>
      </n-card>

      <n-card class="live-card live-card--session" :bordered="false">
        <div class="card-heading">
          <div>
            <span class="step-index">02</span>
            <div>
              <h3>会话控制</h3>
              <p>启动后默认翻译整个目标客户区；可在字幕工具栏或此处重新框选区域。</p>
            </div>
          </div>
          <n-spin v-if="liveStatus.state === 'warming' || liveStatus.state === 'stopping'" size="small" />
        </div>

        <n-alert
          v-if="commandError || liveStatus.state === 'error'"
          type="error"
          title="实时翻译未能继续"
          :show-icon="true"
        >
          {{ commandError || liveStatus.message }}
        </n-alert>

        <div class="session-status" :class="`session-status-${liveStatus.state}`">
          <span class="session-pulse" aria-hidden="true"></span>
          <div>
            <strong>{{ stateLabel }}</strong>
            <p>{{ liveStatus.message }}</p>
          </div>
        </div>

        <dl class="session-details">
          <div>
            <dt>会话 ID</dt>
            <dd>{{ liveStatus.sessionId || "—" }}</dd>
          </div>
          <div>
            <dt>当前选区</dt>
            <dd>{{ roiLabel }}</dd>
          </div>
          <div>
            <dt>字幕修订</dt>
            <dd>{{ liveStatus.latestRevision }}</dd>
          </div>
        </dl>

        <div class="session-actions">
          <n-button
            v-if="liveStatus.state === 'running'"
            secondary
            :loading="activeAction === 'pause'"
            :disabled="activeAction !== null"
            @click="runSessionAction('pause')"
          >暂停</n-button>
          <n-button
            v-if="liveStatus.state === 'paused'"
            type="primary"
            :loading="activeAction === 'resume'"
            :disabled="activeAction !== null"
            @click="runSessionAction('resume')"
          >继续</n-button>
          <n-button
            v-if="liveStatus.state === 'running' || liveStatus.state === 'paused'"
            secondary
            :loading="activeAction === 'reselect'"
            :disabled="activeAction !== null"
            @click="runSessionAction('reselect')"
          >重新选择区域</n-button>
          <n-button
            v-if="liveStatus.state === 'selecting'"
            secondary
            :loading="activeAction === 'cancel'"
            :disabled="activeAction !== null"
            @click="runSessionAction('cancel')"
          >取消区域选择</n-button>
          <n-button
            v-if="hasActiveSession || liveStatus.state === 'error'"
            type="error"
            secondary
            :loading="activeAction === 'stop'"
            :disabled="activeAction !== null || liveStatus.state === 'stopping'"
            @click="runSessionAction('stop')"
          >停止会话</n-button>
        </div>
      </n-card>
    </div>
    </section>


    <section class="live-secondary-config" aria-labelledby="live-secondary-config-title">
      <div class="live-section-heading">
        <div>
          <p class="panel-kicker">Secondary configuration</p>
          <h3 id="live-secondary-config-title">启动前配置</h3>
          <p>显示、翻译与识别设置按需调整，并在下一次开始抓取字幕时应用。运行中仅显示设置可即时保存。</p>
        </div>
        <div class="live-section-heading-actions">
          <n-tag v-if="hasActiveSession" size="small" type="warning">运行中</n-tag>
          <n-button secondary size="small" @click="showAdvancedConfig = !showAdvancedConfig">
            {{ showAdvancedConfig ? "收起" : "展开配置" }}
          </n-button>
        </div>
      </div>
      <div v-show="showAdvancedConfig" class="live-config-collapsible">
        <div class="live-config-grid">
    <n-card class="live-card live-settings-card" :bordered="false">
      <div class="card-heading">
        <div>
          <span class="step-index">03</span>
          <div>
            <h3>实时显示设置</h3>
            <p>配置字幕浮层的位置、显示模式、OCR 原文与译文框体调试。</p>
          </div>
        </div>
        <n-button secondary size="small" @click="saveLiveOverlayPreferences">
          保存显示设置
        </n-button>
      </div>

      <div class="live-settings-grid">
        <label class="live-field">
          <span>显示模式</span>
          <n-select
            v-model:value="liveOverlaySettings.mode"
            :options="liveOverlayModeOptions"
            aria-label="实时翻译显示模式"
          />
        </label>
        <template v-if="liveOverlaySettings.mode === 'subtitle'">
          <label class="live-field">
            <span>贴附边</span>
            <n-select
              v-model:value="liveOverlaySettings.attachment"
              :options="liveOverlayAttachmentOptions"
              aria-label="实时翻译框贴附边"
            />
          </label>
          <label class="live-field live-number-field">
            <span>外侧偏移（像素）</span>
            <n-input-number
              v-model:value="liveOverlaySettings.offset"
              :min="0"
              :max="2048"
              :step="4"
              aria-label="实时翻译框外侧偏移"
            />
          </label>
        </template>
        <label class="live-field">
          <span>显示 OCR 原文</span>
          <n-switch v-model:value="liveOverlaySettings.showSource" aria-label="显示实时 OCR 原文">
            <template #checked>显示</template>
            <template #unchecked>隐藏</template>
          </n-switch>
        </label>
        <label v-if="liveOverlaySettings.mode === 'region_replace'" class="live-field">
          <span>译文框体调试</span>
          <n-button
            :type="liveOverlaySettings.showRegionBoxes ? 'primary' : 'default'"
            secondary
            @click="toggleLiveRegionBoxes"
          >
            {{ liveOverlaySettings.showRegionBoxes ? "隐藏译文框体" : "显示译文框体" }}
          </n-button>
        </label>
      </div>

      <section class="live-subtitle-style-settings" aria-labelledby="live-subtitle-style-title">
        <div class="live-subtitle-style-heading">
          <div>
            <span class="panel-kicker">Subtitle appearance</span>
            <h4 id="live-subtitle-style-title">字幕样式</h4>
            <p>调整字幕文字与背景，保存后在下一次实时翻译会话中应用。</p>
          </div>
        </div>
        <div class="live-subtitle-style-grid">
          <label class="live-field">
            <span>字体颜色</span>
            <div class="live-color-control">
              <input
                v-model="liveSubtitleStyleSettings.fontColor"
                class="live-color-input"
                type="color"
                aria-label="字幕字体颜色"
              />
              <code>{{ liveSubtitleStyleSettings.fontColor }}</code>
            </div>
          </label>
          <label class="live-field live-number-field">
            <span>字体大小（像素）</span>
            <n-input-number
              v-model:value="liveSubtitleStyleSettings.fontSize"
              :min="LIVE_SUBTITLE_FONT_SIZE_MIN"
              :max="LIVE_SUBTITLE_FONT_SIZE_MAX"
              :step="1"
              :precision="0"
              aria-label="字幕字体大小"
            />
          </label>
          <label class="live-field">
            <span>背景颜色</span>
            <div class="live-color-control">
              <input
                v-model="liveSubtitleStyleSettings.backgroundColor"
                class="live-color-input"
                type="color"
                aria-label="字幕背景颜色"
              />
              <code>{{ liveSubtitleStyleSettings.backgroundColor }}</code>
            </div>
          </label>
          <label class="live-field live-number-field">
            <span>背景透明度（%）</span>
            <n-input-number
              v-model:value="liveSubtitleStyleSettings.backgroundOpacity"
              :min="LIVE_SUBTITLE_BACKGROUND_OPACITY_MIN"
              :max="LIVE_SUBTITLE_BACKGROUND_OPACITY_MAX"
              :step="1"
              :precision="0"
              aria-label="字幕背景透明度"
            />
          </label>
        </div>
      </section>


      <n-alert v-if="liveOverlaySettings.mode === 'region_replace'" type="info" :show-icon="true">
        坐标替换模式固定覆盖目标客户区；贴附边与外侧偏移仅用于字幕框模式。
      </n-alert>
      <p class="live-settings-note">
        译文框体按钮会立即同步到当前浮层；其他显示设置在下一次开始抓取字幕时应用。
      </p>
    </n-card>

    <n-card class="live-card live-settings-card" :bordered="false">
      <div class="card-heading">
        <div>
          <span class="step-index">04</span>
          <div>
            <h3>翻译补充提示</h3>
            <p>把 OCR 内容和此处的要求一起传给 Hy-MT2，只作用于新的实时会话。</p>
          </div>
        </div>
        <n-button
          secondary
          size="small"
          :disabled="hasActiveSession"
          @click="saveLiveTranslationPreferences"
        >
          保存翻译提示
        </n-button>
      </div>

      <label class="live-field">
        <span>补充提示词</span>
        <n-input
          v-model:value="liveTranslationSettings.supplementalPrompt"
          type="textarea"
          :maxlength="LIVE_SUPPLEMENTAL_PROMPT_MAX_CHARS"
          :autosize="{ minRows: 3, maxRows: 8 }"
          :disabled="hasActiveSession"
          placeholder="例如：这是游戏对白；修复明显的 OCR 断词，保留角色名和引号，译文简洁自然。"
          aria-label="实时 OCR 翻译补充提示词"
        />
      </label>

      <div class="live-translation-memory-grid">
        <label class="live-field">
          <span>保留近期翻译上下文</span>
          <n-switch
            v-model:value="liveTranslationSettings.memoryEnabled"
            :disabled="hasActiveSession"
            aria-label="启用实时翻译上下文记忆"
          >
            <template #checked>启用</template>
            <template #unchecked>关闭</template>
          </n-switch>
        </label>
        <label class="live-field live-number-field">
          <span>上下文 token 预算</span>
          <n-input-number
            v-model:value="liveTranslationSettings.memoryMaxTokens"
            :min="LIVE_MEMORY_TOKENS_MIN"
            :max="LIVE_MEMORY_TOKENS_MAX"
            :step="256"
            :precision="0"
            :disabled="hasActiveSession || !liveTranslationSettings.memoryEnabled"
            aria-label="实时翻译上下文 token 预算"
          />
        </label>
        <label class="live-field live-number-field">
          <span>保留轮数</span>
          <n-input-number
            v-model:value="liveTranslationSettings.memoryMaxTurns"
            :min="LIVE_MEMORY_TURNS_MIN"
            :max="LIVE_MEMORY_TURNS_MAX"
            :step="1"
            :precision="0"
            :disabled="hasActiveSession || !liveTranslationSettings.memoryEnabled"
            aria-label="实时翻译上下文记忆轮数"
          />
        </label>
      </div>

      <p class="live-settings-note">
        提示词会和 OCR 待译文本放在同一条 user 消息中；启用记忆后会在同一实时会话中保留近期翻译上下文，并过滤相似度超过 80% 的重复记忆。
      </p>
    </n-card>

    <n-card class="live-card live-recognition-card" :bordered="false">
      <div class="card-heading">
        <div>
          <span class="step-index">05</span>
          <div>
            <h3>识别与稳定</h3>
            <p>选择识别方式、稳定等待、按键触发超时，以及相邻 OCR 文本的处理策略。</p>
        </div>
        </div>
        <n-button
          secondary
          size="small"
          :disabled="isCapturingTriggerKey"
          @click="saveLiveRecognitionPreferences"
        >
          保存识别设置
        </n-button>
      </div>

      <div class="live-recognition-grid">
        <label class="live-field">
          <span>识别模式</span>
          <n-select
            v-model:value="liveRecognitionSettings.mode"
            :options="liveRecognitionModeOptions"
            :disabled="hasActiveSession"
            aria-label="实时翻译识别模式"
          />
        </label>
        <label class="live-field live-number-field">
          <span>字幕稳定等待（毫秒）</span>
          <n-input-number
            v-model:value="liveRecognitionSettings.stabilityWaitMs"
            :min="LIVE_STABILITY_WAIT_MIN_MS"
            :max="LIVE_STABILITY_WAIT_MAX_MS"
            :step="100"
            :precision="0"
            :disabled="hasActiveSession"
            aria-label="OCR 字幕稳定等待"
          />
        </label>
        <label
          v-if="liveRecognitionSettings.mode === 'key_trigger'"
          class="live-field live-number-field"
        >
          <span>按键触发超时（毫秒）</span>
          <n-input-number
            v-model:value="liveRecognitionSettings.keyTriggerTimeoutMs"
            :min="KEY_TRIGGER_TIMEOUT_MIN_MS"
            :max="KEY_TRIGGER_TIMEOUT_MAX_MS"
            :step="100"
            :precision="0"
            :disabled="hasActiveSession"
            aria-label="按键触发 OCR 超时"
          />
        </label>
        <label class="live-field">
          <span>合并相邻文本</span>
          <n-switch
            v-model:value="liveRecognitionSettings.textGroupingEnabled"
            :disabled="hasActiveSession"
            aria-label="合并实时 OCR 相邻文本"
          >
            <template #checked>开启</template>
            <template #unchecked>关闭</template>
          </n-switch>
        </label>
        <div v-if="liveRecognitionSettings.mode === 'key_trigger'" class="live-field">
          <span>触发按键</span>
          <div class="live-trigger-key-control">
            <div
              class="live-trigger-key-value"
              role="status"
              :title="liveRecognitionSettings.triggerKey"
            >
              {{ formatTriggerKey(liveRecognitionSettings.triggerKey) }}
            </div>
            <n-button
              secondary
              size="small"
              :disabled="hasActiveSession"
              @click="toggleTriggerKeyCapture"
            >
              {{ isCapturingTriggerKey ? "取消录入" : "录入按键" }}
            </n-button>
          </div>
          <span v-if="triggerKeyCaptureHint" class="live-settings-note">
            {{ triggerKeyCaptureHint }}
          </span>
        </div>
        <label v-if="liveRecognitionSettings.mode === 'key_trigger'" class="live-field">
          <span>触发时机</span>
          <n-select
            v-model:value="liveRecognitionSettings.triggerEvent"
            :options="liveRecognitionTriggerOptions"
            :disabled="hasActiveSession"
            aria-label="实时翻译触发时机"
          />
        </label>
      </div>

      <n-alert type="info" :show-icon="true">
        <template v-if="liveRecognitionSettings.mode === 'key_trigger'">
          {{
            liveRecognitionSettings.triggerEvent === "press"
              ? `按下 ${formatTriggerKey(liveRecognitionSettings.triggerKey)}`
              : `松开 ${formatTriggerKey(liveRecognitionSettings.triggerKey)}`
          }}
          后等待字幕画面稳定 {{ liveRecognitionSettings.stabilityWaitMs }} ms；若超过
          {{ liveRecognitionSettings.keyTriggerTimeoutMs }} ms 仍未稳定，则使用最新画面执行一次 OCR 和翻译。
        </template>
        <template v-else>
          自动模式会根据 ROI 画面变化，等待字幕连续稳定
          {{ liveRecognitionSettings.stabilityWaitMs }} ms 后再执行 OCR 与翻译；字幕逐字出现时可调大该数值。
        </template>
        {{
          liveRecognitionSettings.textGroupingEnabled
            ? " 同一视觉行的 OCR 碎片会先合并重识别；重识别失败时仍保留合并文本，字幕模式会将完整内容一次翻译。"
            : " 相邻文本将保持原始 OCR 区域，不做合并重识别。"
        }}
      </n-alert>
    </n-card>
        </div>
      </div>
    </section>

    <section class="live-observability" aria-labelledby="live-observability-title">
      <div class="live-section-heading">
        <div>
          <p class="panel-kicker">Runtime details</p>
          <h3 id="live-observability-title">运行数据与诊断</h3>
          <p>指标和调试记录用于会话运行中的状态观察与问题排查，位于主要流程之后。</p>
        </div>
        <div class="live-section-heading-actions">
          <n-tag v-if="debugRecords.length" size="small" round>{{ debugRecords.length }}</n-tag>
          <n-button secondary size="small" @click="showDiagnostics = !showDiagnostics">
            {{ showDiagnostics ? "收起" : "展开诊断" }}
          </n-button>
        </div>
      </div>
      <div v-show="showDiagnostics" class="live-observability-collapsible">
        <div class="live-observability-grid">
    <n-card class="live-card" :bordered="false">
      <div class="metrics-heading">
        <div>
          <p class="panel-kicker">Live Metrics</p>
          <h3>实时指标</h3>
        </div>
        <span>状态事件权威快照</span>
      </div>
      <dl class="metrics-grid">
        <div><dt>捕获帧</dt><dd>{{ formatInteger(liveStatus.metrics.framesCaptured) }}</dd></div>
        <div><dt>丢弃帧</dt><dd>{{ formatInteger(liveStatus.metrics.framesDropped) }}</dd></div>
        <div><dt>ROI 未变化跳过</dt><dd>{{ formatInteger(liveStatus.metrics.framesSkippedUnchanged) }}</dd></div>
        <div><dt>OCR 次数</dt><dd>{{ formatInteger(liveStatus.metrics.ocrRuns) }}</dd></div>
        <div><dt>翻译次数</dt><dd>{{ formatInteger(liveStatus.metrics.translationRuns) }}</dd></div>
        <div><dt>字幕发布</dt><dd>{{ formatInteger(liveStatus.metrics.subtitlePublishes) }}</dd></div>
        <div><dt>最近 OCR</dt><dd>{{ formatDuration(liveStatus.metrics.lastOcrMs) }}</dd></div>
        <div><dt>最近翻译</dt><dd>{{ formatDuration(liveStatus.metrics.lastTranslationMs) }}</dd></div>
        <div><dt>GPU</dt><dd>{{ liveStatus.metrics.gpuName || "未初始化" }}</dd></div>
        <div>
          <dt>可用 / 总显存</dt>
          <dd>
            {{
              liveStatus.metrics.gpuTotalMemoryMib > 0
                ? `${formatInteger(liveStatus.metrics.gpuFreeMemoryMib)} / ${formatInteger(liveStatus.metrics.gpuTotalMemoryMib)} MiB`
                : "暂无"
            }}
          </dd>
        </div>
        <div><dt>执行策略</dt><dd>{{ gpuExecutionModeLabel(liveStatus.metrics.gpuExecutionMode) }}</dd></div>
      </dl>
    </n-card>

    <n-card class="live-card" :bordered="false">
      <div class="debug-heading">
        <div>
          <p class="panel-kicker">Live Debug</p>
          <h3>OCR 与翻译记录</h3>
          <p>当前会话仅保留最新 {{ MAX_DEBUG_RECORDS }} 条记录；文本过长时会截断。</p>
        </div>
        <n-button text size="small" :disabled="debugRecords.length === 0" @click="debugRecords = []">
          清空记录
        </n-button>
      </div>

      <n-empty v-if="debugRecords.length === 0" size="small" description="等待 OCR 或翻译事件。" />
      <n-scrollbar v-else class="debug-records-scrollbar" content-class="debug-records-scroll-content">
        <ol class="debug-records" aria-label="OCR 与翻译调试记录">
          <li v-for="record in debugRecords" :key="`${record.sessionId}-${record.sequence}`" class="debug-record">
            <header class="debug-record-header">
              <span class="debug-record-stage">{{ record.stage === "ocr" ? "OCR" : "翻译" }}</span>
              <strong>{{ debugOutcomeLabel(record.outcome) }}</strong>
              <time>{{ formatDebugTime(record.observedAtEpochMs) }}</time>
            </header>
            <dl class="debug-record-meta">
              <div><dt>ROI 版本</dt><dd>{{ record.roiVersion }}</dd></div>
              <div><dt>文本区域</dt><dd>{{ record.regionCount }}</dd></div>
              <div><dt>耗时</dt><dd>{{ record.durationMs }} ms</dd></div>
              <div><dt>目标语言</dt><dd>{{ record.targetLanguage }}</dd></div>
            </dl>
            <div v-if="record.sourceText" class="debug-record-text">
              <span>OCR 输出</span>
              <pre>{{ record.sourceText }}</pre>
            </div>
            <div v-if="record.translatedText !== undefined" class="debug-record-text">
              <span>翻译输出</span>
              <pre>{{ record.translatedText || "（空）" }}</pre>
            </div>
            <p v-if="record.message" class="debug-record-error">{{ record.message }}</p>
          </li>
        </ol>
      </n-scrollbar>
    </n-card>
        </div>
      </div>
    </section>


    <footer class="live-footer">
      <span>单窗口 · 单 ROI</span>
      <span aria-hidden="true">·</span>
      <span>PP-OCR + Hy-MT2</span>
      <span class="footer-spacer"></span>
      <span>Windows 桌面端</span>
    </footer>
  </section>
</template>

<style scoped src="../styles/live-translation-page.css"></style>
