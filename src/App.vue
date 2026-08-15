<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, ref } from "vue";
import type { CSSProperties } from "vue";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  lightTheme,
  NButton,
  NConfigProvider,
  NGlobalStyle,
  NIcon,
  NLayout,
  NLayoutContent,
  NLayoutSider,
  NMessageProvider,
  NMenu,
  NTag,
  zhCN,
} from "naive-ui";
import type { GlobalThemeOverrides, MenuOption } from "naive-ui";
import { RouterView, useRoute, useRouter } from "vue-router";
import { isWorkspaceRouteName } from "./app-router";
import type { WorkspaceRouteName } from "./app-router";
import {
  backendStatus,
  fetchSharedBackendStatus,
  fetchSharedModelRuntimeStatus,
  loadPersistedTargetLanguage,
  modelRuntimeStatus,
} from "./services/workspace-settings";
import { stopLiveSession } from "./services/live-translation-provider";
import LiveSelectionWindow from "./components/LiveSelectionWindow.vue";
import LiveSubtitleOverlay from "./components/LiveSubtitleOverlay.vue";

type TagType = "default" | "success" | "warning" | "error" | "info";

const themePalette = {
  appBg: "#f5f7fa",
  surface: "#ffffff",
  surfaceRaised: "#ffffff",
  surfaceSoft: "#f8fafc",
  border: "#dcdfe6",
  borderStrong: "#c0c4cc",
  text: "#303133",
  textSoft: "#606266",
  textMuted: "#909399",
  placeholder: "#a8abb2",
  divider: "#ebeef5",
  input: "#ffffff",
  progressRail: "#e4e7ed",
  primary: "#409eff",
  primaryHover: "#66b1ff",
  primaryPressed: "#3a8ee6",
  primarySuppl: "#79bbff",
  success: "#67c23a",
  successHover: "#85ce61",
  successPressed: "#5daf34",
  successSuppl: "#b3e19d",
  warning: "#e6a23c",
  error: "#f56c6c",
} as const;

const themeOverrides: GlobalThemeOverrides = {
  common: {
    baseColor: "#ffffff",
    primaryColor: themePalette.primary,
    primaryColorHover: themePalette.primaryHover,
    primaryColorPressed: themePalette.primaryPressed,
    primaryColorSuppl: themePalette.primarySuppl,
    successColor: themePalette.success,
    successColorHover: themePalette.successHover,
    successColorPressed: themePalette.successPressed,
    successColorSuppl: themePalette.successSuppl,
    warningColor: themePalette.warning,
    errorColor: themePalette.error,
    textColorBase: themePalette.text,
    textColor1: themePalette.text,
    textColor2: themePalette.textSoft,
    textColor3: themePalette.textMuted,
    placeholderColor: themePalette.placeholder,
    dividerColor: themePalette.divider,
    borderColor: themePalette.border,
    cardColor: themePalette.surface,
    modalColor: themePalette.surfaceRaised,
    popoverColor: themePalette.surfaceRaised,
    bodyColor: themePalette.appBg,
    inputColor: themePalette.input,
    progressRailColor: themePalette.progressRail,
    railColor: themePalette.progressRail,
    fontFamily:
      '"Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", ui-sans-serif, system-ui, sans-serif',
    fontFamilyMono: 'Consolas, "Cascadia Code", ui-monospace, monospace',
    borderRadius: "4px",
    borderRadiusSmall: "3px",
  },
  Button: {
    heightMedium: "36px",
    borderRadiusMedium: "4px",
    fontSizeMedium: "14px",
    fontWeight: "500",
    fontWeightStrong: "600",
  },
  Card: {
    color: themePalette.surface,
    colorEmbedded: themePalette.surfaceSoft,
    borderColor: themePalette.divider,
    borderRadius: "4px",
    paddingMedium: "16px",
    boxShadow: "0 2px 12px rgba(0, 0, 0, 0.04)",
  },
  Empty: {
    fontSizeSmall: "12px",
    iconSizeSmall: "16px",
    textColor: themePalette.textMuted,
    iconColor: themePalette.textMuted,
    extraTextColor: themePalette.textMuted,
  },
  Menu: {
    color: "#0000",
    borderRadius: "4px",
    fontSize: "14px",
    itemHeight: "40px",
    itemTextColor: themePalette.textSoft,
    itemTextColorHover: themePalette.text,
    itemTextColorActive: themePalette.primary,
    itemTextColorActiveHover: themePalette.primary,
    itemIconColor: themePalette.textMuted,
    itemIconColorHover: themePalette.text,
    itemIconColorActive: themePalette.primary,
    itemIconColorActiveHover: themePalette.primary,
    itemColorHover: "#f5f7fa",
    itemColorActive: "#ecf5ff",
    itemColorActiveHover: "#ecf5ff",
  },
  Tag: {
    fontSizeSmall: "12px",
    fontSizeMedium: "12px",
  },
  Input: {
    fontSizeSmall: "12px",
    fontSizeMedium: "14px",
  },
  Alert: {
    fontSize: "14px",
    borderRadius: "4px",
  },
};

const appThemeStyle = {
  "--app-bg": themePalette.appBg,
  "--surface": themePalette.surface,
  "--surface-raised": themePalette.surfaceRaised,
  "--surface-soft": themePalette.surfaceSoft,
  "--border": themePalette.border,
  "--border-strong": themePalette.borderStrong,
  "--divider": themePalette.divider,
  "--text": themePalette.text,
  "--text-soft": themePalette.textSoft,
  "--text-muted": themePalette.textMuted,
  "--green": themePalette.primary,
  "--green-soft": themePalette.primarySuppl,
} as CSSProperties;

function renderTranslationIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("path", {
            d: "M4 5.5h7M4 10h12M4 14.5h7",
            stroke: "currentColor",
            "stroke-width": "1.4",
            "stroke-linecap": "round",
          }),
          h("path", {
            d: "m13 4 3 1.5-3 1.5M11 13l-3 1.5 3 1.5",
            stroke: "currentColor",
            "stroke-width": "1.4",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
        ]),
    },
  );
}

function renderOcrIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("rect", {
            x: "3.25",
            y: "4.25",
            width: "13.5",
            height: "11.5",
            rx: "1.5",
            stroke: "currentColor",
            "stroke-width": "1.2",
          }),
          h("circle", {
            cx: "7",
            cy: "8",
            r: "1.35",
            stroke: "currentColor",
            "stroke-width": "1.2",
          }),
          h("path", {
            d: "m5 13 3-2.8 2.5 2 2.5-2.2 2 3",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
        ]),
    },
  );
}

function renderOcrTranslationIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("rect", {
            x: "3.25",
            y: "3.25",
            width: "7.5",
            height: "7.5",
            rx: "1.2",
            stroke: "currentColor",
            "stroke-width": "1.2",
          }),
          h("path", {
            d: "m4.8 9 1.8-1.8L8 8.4l1.7-1.7",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
          h("path", {
            d: "M12.5 12.5h4M14.5 10.5v4M11.5 16h5",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linecap": "round",
          }),
        ]),
    },
  );
}

function renderSettingsIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("path", {
            d: "M8.7 2.2h2.6l.5 1.8a6.7 6.7 0 0 1 1.3.8l1.8-.5 1.3 2.2-1.3 1.3c.1.5.2.9.2 1.4s-.1.9-.2 1.4l1.3 1.3-1.3 2.2-1.8-.5a6.7 6.7 0 0 1-1.3.8l-.5 1.8H8.7l-.5-1.8a6.7 6.7 0 0 1-1.3-.8l-1.8.5-1.3-2.2 1.3-1.3a6.7 6.7 0 0 1-.2-1.4c0-.5.1-.9.2-1.4L3.8 6.5l1.3-2.2 1.8.5a6.7 6.7 0 0 1 1.3-.8l.5-1.8Z",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linejoin": "round",
          }),
          h("circle", {
            cx: "10",
            cy: "10",
            r: "2.2",
            stroke: "currentColor",
            "stroke-width": "1.2",
          }),
        ]),
    },
  );
}

function renderLiveTranslationIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("rect", {
            x: "3",
            y: "4",
            width: "14",
            height: "10",
            rx: "1.5",
            stroke: "currentColor",
            "stroke-width": "1.2",
          }),
          h("path", {
            d: "M6 17h8M8.5 14v3M11.5 14v3M5.5 9h2l1.3-2.2L11 11l1.4-1.8 2.1 1.3",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
          }),
        ]),
    },
  );
}

function renderModelMonitorIcon() {
  return h(
    NIcon,
    { size: 16 },
    {
      default: () =>
        h("svg", { viewBox: "0 0 20 20", fill: "none", "aria-hidden": "true" }, [
          h("path", {
            d: "M4 15.5V10M8 15.5V6.5M12 15.5V8.5M16 15.5V4.5",
            stroke: "currentColor",
            "stroke-width": "1.4",
            "stroke-linecap": "round",
          }),
          h("path", {
            d: "M3 17h14",
            stroke: "currentColor",
            "stroke-width": "1.2",
            "stroke-linecap": "round",
          }),
        ]),
    },
  );
}

function renderNavOptionLabel(title: string, description: string) {
  return () =>
    h("span", { class: "nav-option-label" }, [
      h("span", { class: "nav-option-title" }, title),
      h("span", { class: "nav-option-description" }, description),
    ]);
}

const primaryMenuOptions: MenuOption[] = [
  {
    label: "实时翻译",
    key: "live-translation",
    icon: renderLiveTranslationIcon,
  },
];

const oneShotMenuOptions: MenuOption[] = [
  {
    label: renderNavOptionLabel("文本翻译", "粘贴文本，快速完成一次翻译"),
    key: "translate",
    icon: renderTranslationIcon,
  },
  {
    label: renderNavOptionLabel("OCR", "上传图片，提取其中的文字"),
    key: "ocr",
    icon: renderOcrIcon,
  },
  {
    label: renderNavOptionLabel("OCR翻译", "上传图片，识别并翻译文字"),
    key: "ocr-translate",
    icon: renderOcrTranslationIcon,
  },
];

const operationsMenuOptions: MenuOption[] = [
  {
    label: "设置",
    key: "settings",
    icon: renderSettingsIcon,
  },
  {
    label: "模型监控",
    key: "model-monitor",
    icon: renderModelMonitorIcon,
  },
];


const route = useRoute();
const router = useRouter();
const activeMenu = computed<WorkspaceRouteName>(() =>
  isWorkspaceRouteName(route.name) ? route.name : "ocr-translate",
);

function handleMenuUpdate(value: string) {
  if (!isWorkspaceRouteName(value) || value === activeMenu.value) {
    return;
  }
  void router.push({ name: value });
}

const isDesktopRuntime = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
const appWindow = isDesktopRuntime ? getCurrentWindow() : null;
const windowLabel = appWindow?.label ?? "main";
const isLiveSelectorWindow = windowLabel === "live-selector";
const isLiveOverlayWindow = windowLabel === "live-overlay";
const isMainWorkspaceWindow = !isLiveSelectorWindow && !isLiveOverlayWindow;
const isWindowMaximized = ref(false);
let windowStateUnlisten: (() => void) | undefined;
let windowStateListenerActive = true;
let windowCloseUnlisten: (() => void) | undefined;
let closeCleanupInFlight = false;

const settingsStatusLabel = computed(() => {
  if (!backendStatus.value) {
    return isDesktopRuntime ? "读取中" : "浏览器预览";
  }
  return backendStatus.value.ready ? "模型已就绪" : "需要检查";
});

const modelProviderLabel = computed(() => {
  const status = modelRuntimeStatus.value;
  if (!status) {
    return settingsStatusLabel.value;
  }
  if (status.busy) {
    return "正在推理";
  }
  const loadedCount = Number(status.ocrLoaded) + Number(status.translatorLoaded);
  return loadedCount > 0 ? `已加载 ${loadedCount}/2` : "按需加载";
});

type PageMetadata = {
  title: string;
  titleId: string;
  statusLabel: string;
  statusType: TagType;
  statusAriaLabel: string;
};

const pageMetadata = computed<PageMetadata>(() => {
  switch (activeMenu.value) {
    case "translate":
      return {
        title: "翻译",
        titleId: "translate-title",
        statusLabel: "文本翻译",
        statusType: "info",
        statusAriaLabel: "文本翻译状态",
      };
    case "ocr":
      return {
        title: "OCR",
        titleId: "ocr-title",
        statusLabel: "OCR 识别",
        statusType: "info",
        statusAriaLabel: "OCR 状态",
      };
    case "ocr-translate":
      return {
        title: "OCR翻译",
        titleId: "ocr-translate-title",
        statusLabel: "OCR 翻译",
        statusType: "info",
        statusAriaLabel: "OCR 翻译流程状态",
      };
    case "live-translation":
      return {
        title: "实时翻译",
        titleId: "live-translation-title",
        statusLabel: "窗口 OCR 翻译",
        statusType: "info",
        statusAriaLabel: "实时翻译状态",
      };
    case "settings":
      return {
        title: "设置",
        titleId: "settings-title",
        statusLabel: settingsStatusLabel.value,
        statusType: backendStatus.value?.ready ? "success" : "warning",
        statusAriaLabel: "后端状态",
      };
    case "model-monitor":
      return {
        title: "模型监控",
        titleId: "model-monitor-title",
        statusLabel: modelProviderLabel.value,
        statusType: !modelRuntimeStatus.value
          ? "warning"
          : modelRuntimeStatus.value.busy
            ? "warning"
            : "success",
        statusAriaLabel: "模型运行状态",
      };
  }
});

async function syncWindowState() {
  if (!appWindow) {
    isWindowMaximized.value = typeof document !== "undefined" && document.fullscreenElement !== null;
    return;
  }
  try {
    isWindowMaximized.value = await appWindow.isMaximized();
  } catch {
    // Window state is optional in the browser preview.
  }
}

async function bindWindowStateListener() {
  if (!appWindow) {
    if (typeof document === "undefined") {
      return;
    }
    const syncBrowserFullscreenState = () => {
      void syncWindowState();
    };
    document.addEventListener("fullscreenchange", syncBrowserFullscreenState);
    windowStateUnlisten = () => {
      document.removeEventListener("fullscreenchange", syncBrowserFullscreenState);
    };
    return;
  }
  try {
    const unlisten = await appWindow.onResized(() => {
      void syncWindowState();
    });
    if (!windowStateListenerActive) {
      unlisten();
      return;
    }
    windowStateUnlisten = unlisten;
  } catch {
    // Window state events are optional in the browser preview.
  }
}

async function bindMainWindowCloseListener(): Promise<void> {
  if (!appWindow || !isMainWorkspaceWindow) {
    return;
  }
  try {
    const unlisten = await appWindow.onCloseRequested((event) => {
      event.preventDefault();
      if (closeCleanupInFlight) {
        return;
      }
      closeCleanupInFlight = true;
      // Let the close event return first so backend window cleanup can use
      // the Tauri main thread before destroying the main window.
      void stopLiveSession()
        .catch(() => undefined)
        .finally(() => {
          void appWindow.destroy().catch(() => undefined);
        });
    });
    if (!windowStateListenerActive) {
      unlisten();
      return;
    }
    windowCloseUnlisten = unlisten;
  } catch {
    // Window close events are optional in the browser preview.
  }
}

function minimizeWindow() {
  if (appWindow) {
    void appWindow.minimize().catch(() => undefined);
  }
}

async function toggleWindowMaximize() {
  if (!appWindow) {
    if (typeof document === "undefined") {
      return;
    }
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await document.documentElement.requestFullscreen();
      }
    } catch {
      return;
    } finally {
      void syncWindowState();
    }
    return;
  }
  try {
    await appWindow.toggleMaximize();
  } catch {
    return;
  } finally {
    void syncWindowState();
  }
}

function closeWindow() {
  if (appWindow) {
    void appWindow.close().catch(() => undefined);
  }
}

function openModelMonitor() {
  if (activeMenu.value !== "model-monitor") {
    void router.push({ name: "model-monitor" });
  }
}

function handleTitlebarMouseDown(event: MouseEvent) {
  if (!appWindow || event.button !== 0) {
    return;
  }
  const target = event.target as HTMLElement | null;
  if (target?.closest("button")) {
    return;
  }
  void appWindow.startDragging().catch(() => undefined);
}

function handleTitlebarDoubleClick(event: MouseEvent) {
  const target = event.target as HTMLElement | null;
  if (!target?.closest("button")) {
    void toggleWindowMaximize();
  }
}

onMounted(() => {
  if (!isMainWorkspaceWindow) {
    return;
  }
  loadPersistedTargetLanguage();
  void syncWindowState();
  void bindWindowStateListener();
  void bindMainWindowCloseListener();
  if (isDesktopRuntime) {
    void fetchSharedBackendStatus().catch(() => undefined);
    void fetchSharedModelRuntimeStatus().catch(() => undefined);
  }
});

onBeforeUnmount(() => {
  windowStateListenerActive = false;
  windowStateUnlisten?.();
  windowStateUnlisten = undefined;
  windowCloseUnlisten?.();
  windowCloseUnlisten = undefined;
});
</script>

<template>
  <n-config-provider :locale="zhCN" :theme="lightTheme" :theme-overrides="themeOverrides">
    <n-message-provider>
      <n-global-style />
      <a v-if="isMainWorkspaceWindow" class="skip-link" href="#main-content">跳转到主要内容</a>

      <LiveSelectionWindow v-if="isLiveSelectorWindow" />
      <LiveSubtitleOverlay v-else-if="isLiveOverlayWindow" />
      <div v-else class="app-shell" :style="appThemeStyle">
        <header
          class="titlebar"
          data-tauri-drag-region
          @mousedown="handleTitlebarMouseDown"
          @dblclick="handleTitlebarDoubleClick"
        >
          <div class="titlebar-brand" data-tauri-drag-region>
            <span class="brand-mark" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M5 5h5v5H5zM14 5h5v5h-5zM5 14h5v5H5z" fill="currentColor" />
                <path d="M14 14h5M16.5 11.5V19M14 17h5" stroke="currentColor" stroke-width="1.7" />
              </svg>
            </span>
            <span class="titlebar-name">smodeltrans</span>
            <span class="titlebar-divider" aria-hidden="true">/</span>
            <span class="titlebar-context">{{ pageMetadata.title }}</span>
          </div>

          <div class="window-controls" aria-label="窗口控制">
            <n-button
              class="window-control"
              quaternary
              circle
              size="small"
              aria-label="最小化窗口"
              title="最小化"
              @click.stop="minimizeWindow"
            >
              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M3 8h10" stroke="currentColor" stroke-width="1.2" />
              </svg>
            </n-button>
            <n-button
              class="window-control"
              quaternary
              circle
              size="small"
              :aria-label="isWindowMaximized ? '恢复窗口' : '最大化窗口'"
              :title="isWindowMaximized ? '恢复' : '最大化'"
              @click.stop="toggleWindowMaximize"
            >
              <svg v-if="!isWindowMaximized" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <rect x="3.25" y="3.25" width="9.5" height="9.5" stroke="currentColor" stroke-width="1.2" />
              </svg>
              <svg v-else viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M5.5 5.5h6v6h-6z" stroke="currentColor" stroke-width="1.2" />
                <path d="M4.5 10.5h-1v-7h7v1" stroke="currentColor" stroke-width="1.2" />
              </svg>
            </n-button>
            <n-button
              class="window-control window-control-close"
              quaternary
              circle
              size="small"
              aria-label="关闭窗口"
              title="关闭"
              @click.stop="closeWindow"
            >
              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="m4 4 8 8M12 4l-8 8" stroke="currentColor" stroke-width="1.2" />
              </svg>
            </n-button>
          </div>
        </header>

        <n-layout has-sider class="workspace-shell">
          <n-layout-sider class="sidebar" bordered :width="208" :native-scrollbar="false">

            <nav class="sidebar-nav" aria-label="主导航">
              <section class="nav-section nav-section-primary" aria-labelledby="nav-primary-heading">
                <p id="nav-primary-heading" class="nav-heading">主要功能</p>
                <p class="nav-context">先选择窗口并开始实时翻译。</p>
                <n-menu
                  :value="activeMenu"
                  :icon-size="16"
                  :options="primaryMenuOptions"
                  @update:value="handleMenuUpdate"
                />
              </section>

              <section class="nav-section" aria-labelledby="nav-one-shot-heading">
                <p id="nav-one-shot-heading" class="nav-heading">一次性工具</p>
                <p class="nav-context">按内容选择文本、OCR 或 OCR 翻译。</p>
                <n-menu
                  :value="activeMenu"
                  :icon-size="16"
                  :options="oneShotMenuOptions"
                  @update:value="handleMenuUpdate"
                />
              </section>

              <section class="nav-section" aria-labelledby="nav-operations-heading">
                <p id="nav-operations-heading" class="nav-heading">操作</p>
                <p class="nav-context">准备模型和目标语言，或查看运行状态。</p>
                <n-menu
                  :value="activeMenu"
                  :icon-size="16"
                  :options="operationsMenuOptions"
                  @update:value="handleMenuUpdate"
                />
              </section>
            </nav>

            <div class="sidebar-bottom">
              <button class="provider-card" type="button" @click="openModelMonitor">
                <span
                  class="provider-indicator"
                  :class="{
                    'provider-indicator-active':
                      modelRuntimeStatus?.ocrLoaded || modelRuntimeStatus?.translatorLoaded,
                    'provider-indicator-busy': modelRuntimeStatus?.busy,
                  }"
                  aria-hidden="true"
                ></span>
                <div>
                  <strong>本地模型</strong>
                  <span>{{ modelProviderLabel }}</span>
                </div>
              </button>
              <p class="sidebar-build">ver. 0.1.0</p>
            </div>
          </n-layout-sider>

          <n-layout class="workspace-main">
            <n-layout-content id="main-content" class="content">
              <section class="workspace-header" :aria-labelledby="pageMetadata.titleId">
                <h1 :id="pageMetadata.titleId">{{ pageMetadata.title }}</h1>
                <n-tag
                  class="state-tag"
                  :type="pageMetadata.statusType"
                  round
                  size="small"
                  :aria-label="pageMetadata.statusAriaLabel"
                >
                  {{ pageMetadata.statusLabel }}
                </n-tag>
              </section>

              <RouterView v-slot="{ Component }">
                <KeepAlive>
                  <component :is="Component" />
                </KeepAlive>
              </RouterView>
            </n-layout-content>
          </n-layout>
        </n-layout>
      </div>
    </n-message-provider>
  </n-config-provider>
</template>

<style>
:root {
  color: #303133;
  background: #f5f7fa;
  color-scheme: light;
  font-family:
    "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", ui-sans-serif, system-ui, sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  --app-bg: #f5f7fa;
  --surface: #ffffff;
  --surface-raised: #ffffff;
  --surface-soft: #f8fafc;
  --border: #dcdfe6;
  --border-strong: #c0c4cc;
  --divider: #ebeef5;
  --text: #303133;
  --text-soft: #606266;
  --text-muted: #909399;
  --primary: #409eff;
  --primary-soft: #79bbff;
  --green: #409eff;
  --green-soft: #79bbff;
  --font-size-meta: 12px;
  --font-size-body: 14px;
  --font-size-heading: 16px;
}

* {
  box-sizing: border-box;
}

html,
body,
#app {
  min-width: 320px;
  min-height: 100%;
  margin: 0;
}

body {
  height: 100vh;
  overflow: hidden;
  background: var(--app-bg);
}

button,
input,
textarea {
  font: inherit;
}

button {
  cursor: pointer;
}

button:focus-visible,
[role="button"]:focus-visible,
a:focus-visible {
  outline: 2px solid var(--green);
}

.drop-zone:focus {
  outline: 2px solid var(--green);
  outline-offset: 3px;
}

::selection {
  color: #ffffff;
  background: var(--green);
}

.skip-link {
  position: fixed;
  z-index: 50;
  top: 12px;
  left: 12px;
  padding: 8px 12px;
  border-radius: 4px;
  color: #ffffff;
  background: var(--green);
  font-size: 14px;
  font-weight: 600;
  transform: translateY(-160%);
  transition: transform 180ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

.app-shell {
  display: flex;
  height: 100dvh;
  min-height: 560px;
  flex-direction: column;
  overflow: hidden;
  background: var(--app-bg);
}

.titlebar {
  display: flex;
  height: 44px;
  flex: 0 0 44px;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--divider);
  color: var(--text-soft);
  background: var(--surface);
  user-select: none;
  -webkit-app-region: drag;
}

.titlebar-brand {
  display: flex;
  min-width: 0;
  height: 100%;
  align-items: center;
  gap: 9px;
  padding: 0 16px;
  -webkit-app-region: drag;
}

.brand-mark {
  display: grid;
  width: 21px;
  height: 21px;
  flex: 0 0 21px;
  place-items: center;
  border: 1px solid rgba(64, 158, 255, 0.38);
  border-radius: 4px;
  color: var(--green);
  background: #ecf5ff;
}

.brand-mark svg {
  width: 15px;
  height: 15px;
}

.titlebar-name,
.titlebar-context,
.titlebar-divider {
  white-space: nowrap;
}

.titlebar-name {
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0;
}

.titlebar-divider {
  color: #c0c4cc;
  font-family: inherit;
  font-size: 12px;
}

.titlebar-context {
  overflow: hidden;
  color: var(--text-muted);
  font-family: inherit;
  font-size: 12px;
  letter-spacing: 0;
  text-overflow: ellipsis;
}

.window-controls {
  display: flex;
  height: 100%;
  -webkit-app-region: no-drag;
}

.titlebar .n-button.window-control {
  display: grid;
  width: 44px;
  min-width: 44px;
  height: 44px;
  min-height: 44px;
  padding: 0;
  place-items: center;
  border: 0;
  border-radius: 0;
  color: var(--text-muted);
  background: transparent;
  -webkit-app-region: no-drag;
}

.titlebar .n-button.window-control svg {
  width: 16px;
  height: 16px;
}

.titlebar .n-button.window-control:hover {
  color: var(--text);
  background: #f5f7fa;
}

.titlebar .n-button.window-control-close:hover {
  color: #f56c6c;
  background: #fef0f0;
}

.workspace-shell {
  display: flex;
  min-height: 0;
  flex: 1;
  background: var(--app-bg);
}

.workspace-main {
  display: flex;
  min-width: 0;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  background: var(--app-bg);
}

.workspace-main > .n-layout-scroll-container {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
}

.workspace-main > .n-layout-scroll-container > .content {
  flex: 1;
  min-height: 0;
}

.sidebar {
  display: flex;
  min-height: 0;
  flex: 0 0 208px;
  flex-direction: column;
  border-right: 1px solid var(--divider);
  background: var(--surface);
  overflow: hidden;
}

.sidebar .n-layout-sider__content {
  display: flex;
  min-height: 100%;
  flex-direction: column;
}

.sidebar > .n-scrollbar,
.sidebar > .n-scrollbar > .n-scrollbar-container,
.sidebar > .n-scrollbar > .n-scrollbar-container > .n-scrollbar-content {
  display: flex;
  height: 100%;
  min-height: 100%;
  flex-direction: column;
}

.sidebar > .n-scrollbar {
  min-height: 0;
  flex: 1;
}

.sidebar-header {
  padding: 24px 20px 22px;
}

.sidebar-kicker,
.nav-heading,
.sidebar-build {
  margin: 0;
  color: var(--text-muted);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0;
}

.sidebar-title {
  margin: 9px 0 0;
  color: var(--text);
  font-family: inherit;
  font-size: 14px;
  font-weight: 700;
  letter-spacing: 0;
}

.sidebar-subtitle {
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 12px;
}

.sidebar-nav {
  display: grid;
  gap: 18px;
  padding: 0 10px 12px;
}

.nav-section {
  min-width: 0;
}

.nav-section-primary {
  padding: 10px 8px 8px;
  border: 1px solid #d9ecff;
  border-radius: 6px;
  background: #f8fbff;
}

.sidebar-nav .n-menu {
  width: 100%;
  background: transparent;
}

.nav-heading {
  padding: 0 9px 8px;
}

.nav-section-primary .nav-heading {
  color: var(--green);
}

.nav-context {
  margin: -2px 9px 8px;
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}
.nav-option-label {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 1px;
  line-height: 1.2;
}

.nav-option-title,
.nav-option-description {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.nav-option-title {
  color: inherit;
  font-size: 13px;
  font-weight: 600;
}

.nav-option-description {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 400;
  line-height: 1.25;
}

.sidebar-bottom {
  margin-top: auto;
  padding: 16px 14px 18px;
}

.provider-card {
  display: flex;
  width: 100%;
  flex: 0 0 auto;
  align-items: center;
  gap: 9px;
  padding: 11px;
  border: 1px solid var(--divider);
  border-radius: 4px;
  color: inherit;
  background: var(--surface);
  box-shadow: none;
  font: inherit;
  text-align: left;
  cursor: pointer;
  transition:
    border-color 150ms ease,
    background-color 150ms ease;
}

.provider-card:hover {
  border-color: var(--primary);
  background: #ecf5ff;
}

.provider-card:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.provider-card strong,
.provider-card span {
  display: block;
}

.provider-card strong {
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 650;
}

.provider-card div span {
  margin-top: 3px;
  color: var(--text-muted);
  font-size: 12px;
}

.provider-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  flex: 0 0 8px;
  border-radius: 50%;
  background: var(--text-muted);
}

.provider-indicator-active {
  background: var(--success);
  box-shadow: 0 0 0 3px rgba(103, 194, 58, 0.14);
}

.provider-indicator-busy {
  background: var(--warning);
  box-shadow: 0 0 0 3px rgba(230, 162, 60, 0.14);
}

.sidebar-build {
  margin: 14px 2px 0;
  color: var(--text-muted);
  font-size: 12px;
  letter-spacing: 0;
}

.content {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: auto;
  padding: 24px clamp(24px, 3vw, 48px) 20px;
}

.workspace-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  margin: 0 auto;
  width: 100%;
  max-width: none;
}

.settings-page {
  width: 100%;
  max-width: none;
  margin: 24px auto 0;
  padding-bottom: 88px;
}

.settings-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.settings-card {
  min-width: 0;
  border: 1px solid var(--divider);
  border-radius: 6px;
  background: var(--surface);
}

.settings-card-wide {
  grid-column: 1 / -1;
}

.settings-card .n-card-content {
  display: flex;
  min-height: 190px;
  flex-direction: column;
  gap: 14px;
}

.settings-card-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.settings-card-heading h2 {
  margin: 6px 0 0;
  font-size: 16px;
}

.settings-card-copy {
  max-width: 620px;
  margin: 0;
  color: var(--text-muted);
  font-size: 13px;
  line-height: 1.6;
}

.settings-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.settings-metrics div {
  padding: 12px;
  border: 1px solid var(--divider);
  border-radius: 4px;
  background: var(--surface-soft);
}

.settings-metrics span,
.settings-metrics strong {
  display: block;
}

.settings-metrics span {
  color: var(--text-muted);
  font-size: 12px;
}

.settings-metrics strong {
  margin-top: 6px;
  color: var(--text);
  font-size: 14px;
  font-weight: 650;
}

.settings-field {
  display: grid;
  gap: 7px;
  max-width: 360px;
  color: var(--text-soft);
  font-size: 13px;
  font-weight: 600;
}

.settings-field-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.settings-field-wide {
  grid-column: 1 / -1;
  max-width: none;
}

.settings-help {
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 400;
  line-height: 1.5;
}

.settings-textarea .n-input {
  width: 100%;
}

.settings-alert {
  margin-top: auto;
}

.settings-path-list {
  display: grid;
  gap: 10px;
  margin: 0;
}

.settings-path-list > div {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--divider);
}

.settings-path-list > div:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.settings-path-list dt {
  color: var(--text-soft);
  font-size: 12px;
  font-weight: 650;
}

.settings-path-list dd {
  min-width: 0;
  margin: 0;
  color: var(--text-muted);
  font-family: var(--font-mono, Consolas, monospace);
  font-size: 12px;
  overflow-wrap: anywhere;
}

.settings-path-list dd {
  display: flex;
  align-items: center;
  gap: 12px;
}

.settings-path-value {
  min-width: 0;
  flex: 1;
  overflow-wrap: anywhere;
}

.settings-model-select-row {
  flex-wrap: wrap;
}

.settings-model-select {
  min-width: 200px;
  flex: 1;
  max-width: 380px;
}

.settings-model-help {
  flex-basis: 100%;
  min-width: 0;
  color: var(--text-muted);
  font-size: 12px;
  overflow-wrap: anywhere;
}

.model-dialog-fields {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.model-dialog-path-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.model-dialog-path-value {
  min-width: 0;
  flex: 1;
  overflow: hidden;
  color: var(--text-muted);
  font-size: 12px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.settings-actions-feedback {
  min-width: 0;
  flex: 1;
  margin-right: auto;
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}

.settings-card-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  margin-top: auto;
}

.settings-page-actions {
  position: sticky;
  z-index: 20;
  bottom: 0;
  align-items: center;
  justify-content: flex-end;
  margin-top: 0;
  padding: 12px 16px;
  border: 1px solid rgba(220, 223, 230, 0.86);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 14px 36px rgba(0, 0, 0, 0.12);
  backdrop-filter: blur(14px);
}

.settings-number-field .n-input-number {
  width: 180px;
}

.settings-number-field .n-input-number .n-input__suffix {
  align-items: center;
}

.settings-number-field .n-input-number .n-input__suffix > .n-button {
  align-self: center;
}

.breadcrumb,
.panel-kicker,
.detail-label,
.preview-frame-bar,
.result-toolbar,
.input-helper,
.workspace-footer,
.progress-meta {
  font-family: inherit;
  letter-spacing: 0;
  text-transform: none;
}

.breadcrumb {
  margin: 0;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.breadcrumb span {
  padding: 0 6px;
  color: #c0c4cc;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1,
h2,
h3 {
  color: var(--text);
}

h1 {
  margin-bottom: 0;
  margin-top: 0;
  font-family: inherit;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0;
  line-height: 1.5;
}


.workflow-grid {
  display: grid;
  width: 100%;
  max-width: none;
  margin: 24px auto 0;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.panel {
  min-height: max(360px, calc(100dvh - 360px));
}

.panel .n-card-content {
  display: flex;
  min-height: max(360px, calc(100dvh - 360px));
  flex-direction: column;
}

.panel-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.panel-title {
  display: flex;
  min-width: 0;
  gap: 12px;
}

.section-number {
  display: grid;
  width: 28px;
  height: 28px;
  flex: 0 0 28px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--green);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
}

.panel-kicker,
.detail-label {
  margin-bottom: 4px;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 600;
}

.panel-heading h2 {
  margin-bottom: 0;
  font-size: 16px;
  font-weight: 650;
  letter-spacing: -0.02em;
  line-height: 1.35;
}

.panel-copy {
  max-width: 340px;
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.drop-zone {
  position: relative;
  display: flex;
  min-height: clamp(220px, 34dvh, 380px);
  flex: 1;
  align-items: center;
  justify-content: center;
  border: 1px dashed var(--border-strong);
  border-radius: 4px;
  color: var(--text);
  background: var(--surface-soft);
  cursor: pointer;
  transition:
    border-color 180ms ease,
    background-color 180ms ease,
    box-shadow 180ms ease;
}

.drop-zone:hover,
.drop-zone-active {
  border-color: var(--green);
  background: #ecf5ff;
  box-shadow: inset 0 0 0 1px rgba(64, 158, 255, 0.12);
}

.drop-zone-label {
  display: flex;
  max-width: 260px;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 24px;
  cursor: pointer;
  text-align: center;
}

.drop-icon {
  display: grid;
  width: 48px;
  height: 48px;
  margin-bottom: 6px;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 4px;
  color: var(--green);
  background: #f5f7fa;
}

.drop-icon svg {
  width: 27px;
  height: 27px;
}

.drop-zone-kicker {
  color: var(--text);
  font-size: 14px;
  font-weight: 500;
  letter-spacing: 0;
}

.drop-zone-copy {
  color: var(--text-muted);
  font-size: 12px;
}
.drop-zone-action {
  margin-top: 8px;
}

.file-input {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  clip-path: inset(50%);
}

.preview-layout {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 14px;
}

.image-preview-frame {
  display: flex;
  min-height: 260px;
  flex: 1;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface-soft);
}
.result-preview-frame {
  display: flex;
  min-height: 150px;
  max-height: 240px;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: var(--surface-soft);
}

.result-preview-canvas {
  display: grid;
  min-height: 0;
  flex: 1;
  place-items: center;
  padding: 8px;
  overflow: auto;
}

.result-image {
  display: block;
  width: 100%;
}

.input-image {
  display: block;
  width: auto;
  max-width: 100%;
  align-self: center;
}

.input-image img {
  display: block;
  width: auto;
  max-width: 100%;
  max-height: 290px;
  object-fit: contain;
  cursor: zoom-in;
}

.result-image img {
  display: block;
  width: auto;
  max-width: 100%;
  max-height: 190px;
  object-fit: contain;
  cursor: zoom-in;
}


.preview-frame-bar,
.result-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  min-height: 32px;
  padding: 0 12px;
  border-bottom: 1px solid var(--divider);
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.preview-frame-state {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: var(--green-soft);
}

.preview-frame-state .provider-indicator {
  width: 5px;
  height: 5px;
  flex-basis: 5px;
  box-shadow: none;
}

.preview-frame-actions {
  display: inline-flex;
  align-items: center;
  gap: 2px;
}



.preview-canvas {
  display: grid;
  min-height: 0;
  flex: 1;
  place-items: center;
  padding: 12px;
  background:
    linear-gradient(45deg, rgba(48, 49, 51, 0.035) 25%, transparent 25%),
    linear-gradient(-45deg, rgba(48, 49, 51, 0.035) 25%, transparent 25%),
    linear-gradient(45deg, transparent 75%, rgba(48, 49, 51, 0.035) 75%),
    linear-gradient(-45deg, transparent 75%, rgba(48, 49, 51, 0.035) 75%),
    #f5f7fa;
  background-position:
    0 0,
    0 8px,
    8px -8px,
    -8px 0;
  background-size: 16px 16px;
}

.preview-canvas img {
  display: block;
  max-width: 100%;
  max-height: 290px;
  object-fit: contain;
}

.preview-details {
  display: flex;
  flex-direction: column;
}

.file-identity {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.file-type-mark {
  display: grid;
  width: 34px;
  height: 34px;
  flex: 0 0 34px;
  place-items: center;
  border: 1px solid #b3d8ff;
  border-radius: 4px;
  color: var(--green);
  background: #ecf5ff;
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
}

.file-identity-copy {
  min-width: 0;
}

.file-identity-copy .detail-label {
  margin-bottom: 3px;
}

.preview-details h3 {
  overflow: hidden;
  margin-bottom: 0;
  color: var(--text-soft);
  font-family: inherit;
  font-size: 12px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-meta {
  margin: 4px 0 0;
  color: var(--text-muted);
  font-family: inherit;
  font-size: 12px;
}

.button-row,
.result-actions {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 9px;
}

.button-row {
  margin-top: 14px;
}

.inline-alert {
  margin-top: 14px;
  border-radius: 7px;
}

.input-helper {
  margin: 12px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.processing-state,
.result-state,
.output-message,
.empty-output {
  min-height: 280px;
  flex: 1;
}

.processing-state {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 14px;
}

.processing-visual {
  position: relative;
  display: grid;
  width: 52px;
  height: 52px;
  margin-bottom: 2px;
  place-items: center;
}

.processing-copy .detail-label {
  margin-bottom: 0;
}

.processing-copy p:last-child {
  max-width: 330px;
  margin: 8px 0 0;
  color: var(--text-soft);
  font-size: 14px;
  line-height: 1.5;
}

.progress-meta {
  display: flex;
  justify-content: space-between;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
}

.progress-meta strong {
  color: var(--green-soft);
  font-weight: 700;
}

.result-state {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.result-toolbar {
  min-height: auto;
  padding: 0 1px 8px;
  border-bottom: 1px solid var(--divider);
}

.result-toolbar-meta {
  display: inline-flex;
  align-items: center;
  gap: 12px;
}

.result-duration {
  color: var(--text-soft);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.result-mode-note {
  margin: -2px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.result-input textarea {
  min-height: 210px !important;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
}

.result-actions {
  justify-content: space-between;
  gap: 10px;
}

.settings-card .n-button {
  align-self: flex-start;
}


.action-feedback {
  margin: 0;
  color: var(--green);
  font-size: 12px;
  line-height: 1.5;
  text-align: right;
}

.output-message {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  gap: 14px;
}

.output-message .inline-alert {
  width: 100%;
  margin-top: 0;
}

.empty-output {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 20px;
  border: 1px dashed var(--border);
  border-radius: 4px;
  color: var(--text-muted);
  text-align: center;
  background: var(--surface-soft);
}



.app-shell .n-button {
  cursor: pointer;
}

.app-shell .n-button--disabled {
  cursor: not-allowed;
}

.workspace-footer {
  display: flex;
  width: 100%;
  max-width: none;
  align-items: center;
  gap: 10px;
  margin: 16px auto 0;
  color: var(--text-muted);
  font-size: 12px;
  font-weight: 500;
  line-height: 1.5;
}

.footer-separator {
  width: 4px;
  height: 4px;
  flex: 0 0 4px;
  border-radius: 50%;
  background: #c0c4cc;
}

.footer-spacer {
  flex: 1;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
  clip-path: inset(50%);
}


@media (max-width: 1040px) {

  .content {
    padding-right: 24px;
    padding-left: 24px;
  }

  .panel .n-card-content {
    padding: 16px;
  }
}

@media (max-width: 900px) {
  .settings-grid {
    grid-template-columns: 1fr;
  }

  .settings-card-wide {
    grid-column: auto;
  }

  .settings-field-grid {
    grid-template-columns: 1fr;
  }
  .workflow-grid {
    grid-template-columns: 1fr;
  }

  .panel,
  .panel .n-card-content {
    min-height: 0;
  }

  .drop-zone,
  .processing-state,
  .result-state,
  .output-message,
  .empty-output {
    min-height: clamp(220px, 34dvh, 320px);
  }

}

@media (max-width: 720px) {
  .settings-page {
    margin-top: 16px;
  }

  .settings-page-actions {
    bottom: 8px;
    flex-direction: column;
    align-items: stretch;
  }

  .settings-page-actions .n-button {
    width: 100%;
  }

  .settings-path-list > div {
    grid-template-columns: 1fr;
    gap: 4px;
  }
  .app-shell {
    min-height: 100dvh;
  }


  .sidebar {
    display: none;
  }

  .content {
    padding: 22px 16px 16px;
  }

  .workspace-header {
    align-items: center;
    flex-wrap: wrap;
  }

}

@media (max-width: 560px) {
  .titlebar-brand {
    padding-left: 12px;
  }

  .titlebar-context,
  .titlebar-divider {
    display: none;
  }

  .window-control {
    width: 42px;
  }

  .content {
    padding-right: 16px;
    padding-left: 16px;
  }

  h1 {
    font-size: 16px;
  }


  .panel .n-card-content {
    padding: 15px;
  }

  .panel-heading {
    flex-direction: column;
    gap: 10px;
  }

  .state-tag {
    align-self: flex-start;
  }

  .drop-zone {
    min-height: clamp(200px, 32dvh, 280px);
  }

  .image-preview-frame {
    min-height: 220px;
  }

  .result-actions {
    align-items: flex-start;
    flex-direction: column;
  }

  .action-feedback {
    text-align: left;
  }

  .workspace-footer {
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .footer-spacer {
    display: none;
  }
}


@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    scroll-behavior: auto !important;
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
</style>