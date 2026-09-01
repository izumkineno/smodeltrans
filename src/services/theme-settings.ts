import { ref } from "vue";

const THEME_LOG_PREFIX = " [theme-settings]";

export type ThemeMode = "light" | "dark" | "system";
export type ResolvedTheme = Exclude<ThemeMode, "system">;

const THEME_MODE_STORAGE_KEY = "smodeltrans.themeMode";
const SYSTEM_THEME_QUERY = "(prefers-color-scheme: dark)";

export const themeMode = ref<ThemeMode>("system");
export const resolvedTheme = ref<ResolvedTheme>("light");

let persistedThemeModeLoaded = false;
let systemThemeQuery: MediaQueryList | null = null;
let systemThemeListener: ((event: MediaQueryListEvent) => void) | null = null;

function isThemeMode(value: unknown): value is ThemeMode {
  return value === "light" || value === "dark" || value === "system";
}

function readSystemTheme(): ResolvedTheme {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
    return "light";
  }
  return window.matchMedia(SYSTEM_THEME_QUERY).matches ? "dark" : "light";
}

function syncDocumentTheme(): void {
  if (typeof document === "undefined") {
    return;
  }
  console.debug(`${THEME_LOG_PREFIX} syncDocumentTheme`, { resolvedTheme: resolvedTheme.value });
  document.documentElement.dataset.theme = resolvedTheme.value;
  document.documentElement.style.colorScheme = resolvedTheme.value;
}

function resolveTheme(): void {
  const prev = resolvedTheme.value;
  resolvedTheme.value = themeMode.value === "system" ? readSystemTheme() : themeMode.value;
  console.debug(`${THEME_LOG_PREFIX} resolveTheme`, { themeMode: themeMode.value, prevResolved: prev, resolvedTheme: resolvedTheme.value });
  syncDocumentTheme();
}

function bindSystemThemeListener(): void {
  if (
    systemThemeQuery ||
    typeof window === "undefined" ||
    typeof window.matchMedia !== "function"
  ) {
    return;
  }
  console.debug(`${THEME_LOG_PREFIX} bindSystemThemeListener start`);

  systemThemeQuery = window.matchMedia(SYSTEM_THEME_QUERY);
  systemThemeListener = (event) => {
    if (themeMode.value !== "system") {
      return;
    }
    resolvedTheme.value = event.matches ? "dark" : "light";
    syncDocumentTheme();
  };

  if (typeof systemThemeQuery.addEventListener === "function") {
    systemThemeQuery.addEventListener("change", systemThemeListener);
  } else {
    const legacySystemThemeQuery = systemThemeQuery as unknown as {
      addListener: (listener: (event: MediaQueryListEvent) => void) => void;
    };
    legacySystemThemeQuery.addListener(systemThemeListener);
  }
}

export function loadPersistedThemeMode(): string | null {
  console.info(`${THEME_LOG_PREFIX} loadPersistedThemeMode start`, { alreadyLoaded: persistedThemeModeLoaded });
  if (persistedThemeModeLoaded || typeof window === "undefined") {
    console.debug(`${THEME_LOG_PREFIX} loadPersistedThemeMode skip`, { alreadyLoaded: persistedThemeModeLoaded, hasWindow: typeof window !== "undefined" });
    resolveTheme();
    return null;
  }
  persistedThemeModeLoaded = true;

  let persistedMode: string | null = null;
  try {
    persistedMode = window.localStorage.getItem(THEME_MODE_STORAGE_KEY);
    console.debug(`${THEME_LOG_PREFIX} loadPersistedThemeMode raw`, { persistedMode });
  } catch (error) {
    console.warn(`${THEME_LOG_PREFIX} loadPersistedThemeMode localStorage read failed`, { error: error instanceof Error ? error.message : String(error) });
    bindSystemThemeListener();
    resolveTheme();
    console.info(`${THEME_LOG_PREFIX} loadPersistedThemeMode fallback`, { themeMode: themeMode.value, resolvedTheme: resolvedTheme.value });
    return "无法读取界面主题设置，将使用系统主题。";
  }

  const nextMode = isThemeMode(persistedMode) ? persistedMode : "system";
  themeMode.value = nextMode;
  bindSystemThemeListener();
  resolveTheme();
  console.info(`${THEME_LOG_PREFIX} loadPersistedThemeMode success`, { persistedMode, themeMode: themeMode.value, resolvedTheme: resolvedTheme.value });
  return null;
}

export function setThemeMode(nextMode: ThemeMode): string | null {
  console.info(`${THEME_LOG_PREFIX} setThemeMode start`, { nextMode, prevMode: themeMode.value });
  if (!isThemeMode(nextMode)) {
    console.warn(`${THEME_LOG_PREFIX} setThemeMode invalid`, { nextMode });
    return "界面主题设置无效。";
  }

  themeMode.value = nextMode;
  bindSystemThemeListener();
  resolveTheme();
  console.debug(`${THEME_LOG_PREFIX} setThemeMode resolved`, { themeMode: themeMode.value, resolvedTheme: resolvedTheme.value });

  try {
    window.localStorage.setItem(THEME_MODE_STORAGE_KEY, nextMode);
    console.info(`${THEME_LOG_PREFIX} setThemeMode persisted success`, { nextMode, resolvedTheme: resolvedTheme.value });
    return null;
  } catch (error) {
    console.warn(`${THEME_LOG_PREFIX} setThemeMode persist failed`, { nextMode, error: error instanceof Error ? error.message : String(error) });
    return "无法保存界面主题设置，请检查应用存储权限。";
  }
}
