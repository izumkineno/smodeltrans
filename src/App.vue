<script setup lang="ts">
import { computed, h, onBeforeUnmount, onMounted, ref } from "vue";
import type { CSSProperties } from "vue";
import { getCurrentWindow, Window } from "@tauri-apps/api/window";
import {
  darkTheme,
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
import { loadPersistedThemeMode, resolvedTheme } from "./services/theme-settings";
import { stopLiveSession } from "./services/live-translation-provider";
import { initializeQuickTranslationSettings } from "./services/quick-translation-settings";
import LiveSelectionWindow from "./components/LiveSelectionWindow.vue";
import LiveSubtitleOverlay from "./components/LiveSubtitleOverlay.vue";
import QuickTranslationOverlay from "./components/QuickTranslationOverlay.vue";

loadPersistedThemeMode();

type TagType = "default" | "success" | "warning" | "error" | "info";

type ThemePalette = {
  baseColor: string;
  appBg: string;
  surface: string;
  surfaceRaised: string;
  surfaceSoft: string;
  surfaceHover: string;
  surfaceActive: string;
  surfaceInfo: string;
  surfaceSuccess: string;
  surfaceError: string;
  surfaceLatency: string;
  border: string;
  borderStrong: string;
  borderInfo: string;
  divider: string;
  text: string;
  textSoft: string;
  textMuted: string;
  placeholder: string;
  textDivider: string;
  textOnPrimary: string;
  input: string;
  progressRail: string;
  progressDot: string;
  primary: string;
  primaryHover: string;
  primaryPressed: string;
  primarySuppl: string;
  success: string;
  successHover: string;
  successPressed: string;
  successSuppl: string;
  warning: string;
  error: string;
  primaryRgb: string;
  successRgb: string;
  warningRgb: string;
  errorRgb: string;
  cardShadow: string;
};

const themePalettes: Record<"light" | "dark", ThemePalette> = {
  light: {
    baseColor: "#ffffff",
    appBg: "#f5f7fa",
    surface: "#ffffff",
    surfaceRaised: "#ffffff",
    surfaceSoft: "#f8fafc",
    surfaceHover: "#f5f7fa",
    surfaceActive: "#ecf5ff",
    surfaceInfo: "#f8fbff",
    surfaceSuccess: "#f6fbf3",
    surfaceError: "#fef7f7",
    surfaceLatency: "#f5f9ff",
    border: "#dcdfe6",
    borderStrong: "#c0c4cc",
    borderInfo: "#d9ecff",
    divider: "#ebeef5",
    text: "#303133",
    textSoft: "#606266",
    textMuted: "#909399",
    placeholder: "#a8abb2",
    textDivider: "#c0c4cc",
    textOnPrimary: "#ffffff",
    input: "#ffffff",
    progressRail: "#e4e7ed",
    progressDot: "#c0c4cc",
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
    primaryRgb: "64, 158, 255",
    successRgb: "103, 194, 58",
    warningRgb: "230, 162, 60",
    errorRgb: "245, 108, 108",
    cardShadow: "0 2px 12px rgba(0, 0, 0, 0.04)",
  },
  dark: {
    baseColor: "#101014",
    appBg: "#18181c",
    surface: "#202024",
    surfaceRaised: "#25252b",
    surfaceSoft: "#24242a",
    surfaceHover: "#2b2b31",
    surfaceActive: "#1d3852",
    surfaceInfo: "#1e2d3d",
    surfaceSuccess: "#213327",
    surfaceError: "#3b2729",
    surfaceLatency: "#1f3042",
    border: "#3f3f46",
    borderStrong: "#52525b",
    borderInfo: "#365878",
    divider: "#303038",
    text: "#f1f1f3",
    textSoft: "#c5c5cc",
    textMuted: "#9999a5",
    placeholder: "#777783",
    textDivider: "#696973",
    textOnPrimary: "#111827",
    input: "#1e1e23",
    progressRail: "#393942",
    progressDot: "#71717a",
    primary: "#70b8ff",
    primaryHover: "#8ac8ff",
    primaryPressed: "#579fdf",
    primarySuppl: "#afd8ff",
    success: "#83d76a",
    successHover: "#9de584",
    successPressed: "#6abb55",
    successSuppl: "#c3efb1",
    warning: "#f0b95a",
    error: "#ff8585",
    primaryRgb: "112, 184, 255",
    successRgb: "131, 215, 106",
    warningRgb: "240, 185, 90",
    errorRgb: "255, 133, 133",
    cardShadow: "0 2px 16px rgba(0, 0, 0, 0.24)",
  },
};

const activeThemePalette = computed(() => themePalettes[resolvedTheme.value]);

function createThemeOverrides(palette: ThemePalette): GlobalThemeOverrides {
  return {
    common: {
      baseColor: palette.baseColor,
      primaryColor: palette.primary,
      primaryColorHover: palette.primaryHover,
      primaryColorPressed: palette.primaryPressed,
      primaryColorSuppl: palette.primarySuppl,
      successColor: palette.success,
      successColorHover: palette.successHover,
      successColorPressed: palette.successPressed,
      successColorSuppl: palette.successSuppl,
      warningColor: palette.warning,
      errorColor: palette.error,
      textColorBase: palette.text,
      textColor1: palette.text,
      textColor2: palette.textSoft,
      textColor3: palette.textMuted,
      placeholderColor: palette.placeholder,
      dividerColor: palette.divider,
      borderColor: palette.border,
      cardColor: palette.surface,
      modalColor: palette.surfaceRaised,
      popoverColor: palette.surfaceRaised,
      bodyColor: palette.appBg,
      inputColor: palette.input,
      progressRailColor: palette.progressRail,
      railColor: palette.progressRail,
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
      fontWeightMedium: "500",
      fontWeightStrong: "600",
    },
    Card: {
      color: palette.surface,
      colorEmbedded: palette.surfaceSoft,
      borderColor: palette.divider,
      borderRadius: "4px",
      paddingMedium: "16px",
      boxShadow: palette.cardShadow,
    },
    Empty: {
      fontSizeSmall: "12px",
      iconSizeSmall: "16px",
      textColor: palette.textMuted,
      iconColor: palette.textMuted,
      extraTextColor: palette.textMuted,
    },
    Menu: {
      color: "#0000",
      borderRadius: "4px",
      fontSize: "14px",
      itemHeight: "40px",
      itemTextColor: palette.textSoft,
      itemTextColorHover: palette.text,
      itemTextColorActive: palette.primary,
      itemTextColorActiveHover: palette.primary,
      itemIconColor: palette.textMuted,
      itemIconColorHover: palette.text,
      itemIconColorActive: palette.primary,
      itemIconColorActiveHover: palette.primary,
      itemColorHover: palette.surfaceHover,
      itemColorActive: palette.surfaceActive,
      itemColorActiveHover: palette.surfaceActive,
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
}

const naiveTheme = computed(() => (resolvedTheme.value === "dark" ? darkTheme : lightTheme));
const themeOverrides = computed(() => createThemeOverrides(activeThemePalette.value));

function createAppThemeStyle(palette: ThemePalette): CSSProperties {
  return {
    "--app-bg": palette.appBg,
    "--surface": palette.surface,
    "--surface-raised": palette.surfaceRaised,
    "--surface-soft": palette.surfaceSoft,
    "--surface-hover": palette.surfaceHover,
    "--surface-active": palette.surfaceActive,
    "--surface-info": palette.surfaceInfo,
    "--surface-success": palette.surfaceSuccess,
    "--surface-error": palette.surfaceError,
    "--surface-latency": palette.surfaceLatency,
    "--border": palette.border,
    "--border-strong": palette.borderStrong,
    "--border-info": palette.borderInfo,
    "--divider": palette.divider,
    "--text": palette.text,
    "--text-soft": palette.textSoft,
    "--text-muted": palette.textMuted,
    "--text-divider": palette.textDivider,
    "--text-on-primary": palette.textOnPrimary,
    "--green": palette.primary,
    "--green-soft": palette.primarySuppl,
    "--primary": palette.primary,
    "--primary-soft": palette.primarySuppl,
    "--primary-rgb": palette.primaryRgb,
    "--success": palette.success,
    "--success-rgb": palette.successRgb,
    "--warning": palette.warning,
    "--warning-rgb": palette.warningRgb,
    "--error": palette.error,
    "--error-rgb": palette.errorRgb,
    "--progress-dot": palette.progressDot,
  } as CSSProperties;
}

const appThemeStyle = computed(() => createAppThemeStyle(activeThemePalette.value));

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
    label: renderNavOptionLabel("文本翻译", "快速完成一次翻译"),
    key: "translate",
    icon: renderTranslationIcon,
  },
  {
    label: renderNavOptionLabel("OCR", "提取其中的文字"),
    key: "ocr",
    icon: renderOcrIcon,
  },
  {
    label: renderNavOptionLabel("OCR翻译", "识别并翻译文字"),
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
const isQuickTranslationWindow = windowLabel === "quick-translation";
const isMainWorkspaceWindow =
  !isLiveSelectorWindow && !isLiveOverlayWindow && !isQuickTranslationWindow;
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

async function destroyWorkspaceWindows(): Promise<void> {
  const quickTranslationWindow = await Window.getByLabel("quick-translation").catch(() => null);
  await quickTranslationWindow?.destroy().catch(() => undefined);
  await appWindow?.destroy().catch(() => undefined);
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
      // the Tauri main thread before destroying every remaining app window.
      void stopLiveSession()
        .catch(() => undefined)
        .finally(() => {
          void destroyWorkspaceWindows();
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
  if (isDesktopRuntime) {
    void initializeQuickTranslationSettings().catch((error) => {
      console.error("初始化快捷翻译设置失败", error);
    });
  }
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
  <n-config-provider :locale="zhCN" :theme="naiveTheme" :theme-overrides="themeOverrides">
    <n-message-provider>
      <n-global-style />
      <a v-if="isMainWorkspaceWindow" class="skip-link" href="#main-content">跳转到主要内容</a>

      <LiveSelectionWindow v-if="isLiveSelectorWindow" />
      <LiveSubtitleOverlay v-else-if="isLiveOverlayWindow" />
      <QuickTranslationOverlay v-else-if="isQuickTranslationWindow" />
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
                <p id="nav-one-shot-heading" class="nav-heading">工具</p>
                <n-menu
                  :value="activeMenu"
                  :icon-size="16"
                  :options="oneShotMenuOptions"
                  @update:value="handleMenuUpdate"
                />
              </section>

              <section class="nav-section" aria-labelledby="nav-operations-heading">
                <p id="nav-operations-heading" class="nav-heading">操作</p>
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
            <n-layout-content
              id="main-content"
              class="content"
              :native-scrollbar="false"
              content-class="content-scroll"
            >
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

<style src="./styles/app.css"></style>