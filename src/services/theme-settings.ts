import { ref } from "vue";

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
  document.documentElement.dataset.theme = resolvedTheme.value;
  document.documentElement.style.colorScheme = resolvedTheme.value;
}

function resolveTheme(): void {
  resolvedTheme.value = themeMode.value === "system" ? readSystemTheme() : themeMode.value;
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
  if (persistedThemeModeLoaded || typeof window === "undefined") {
    resolveTheme();
    return null;
  }
  persistedThemeModeLoaded = true;

  let persistedMode: string | null = null;
  try {
    persistedMode = window.localStorage.getItem(THEME_MODE_STORAGE_KEY);
  } catch {
    bindSystemThemeListener();
    resolveTheme();
    return "无法读取界面主题设置，将使用系统主题。";
  }

  themeMode.value = isThemeMode(persistedMode) ? persistedMode : "system";
  bindSystemThemeListener();
  resolveTheme();
  return null;
}

export function setThemeMode(nextMode: ThemeMode): string | null {
  if (!isThemeMode(nextMode)) {
    return "界面主题设置无效。";
  }

  themeMode.value = nextMode;
  bindSystemThemeListener();
  resolveTheme();

  try {
    window.localStorage.setItem(THEME_MODE_STORAGE_KEY, nextMode);
    return null;
  } catch {
    return "无法保存界面主题设置，请检查应用存储权限。";
  }
}
