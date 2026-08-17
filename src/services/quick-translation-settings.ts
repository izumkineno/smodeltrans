import { invoke } from "@tauri-apps/api/core";
import { ref } from "vue";

const QUICK_TRANSLATION_STORAGE_KEY = "smodeltrans.quickTranslationSettings";

export interface QuickTranslationSettings {
  enabled: boolean;
  shortcut: string;
}

export const quickTranslationShortcutOptions: Array<{ label: string; value: string }> = [
  { label: "Ctrl + Alt + E", value: "CommandOrControl+Alt+E" },
  { label: "Ctrl + Alt + T", value: "CommandOrControl+Alt+T" },
  { label: "Ctrl + Shift + E", value: "CommandOrControl+Shift+E" },
  { label: "Alt + Shift + E", value: "Alt+Shift+E" },
];

const defaultSettings: QuickTranslationSettings = {
  enabled: true,
  shortcut: quickTranslationShortcutOptions[0].value,
};

export const quickTranslationSettings = ref<QuickTranslationSettings>({ ...defaultSettings });
let persistedSettingsLoaded = false;

function isSupportedShortcut(value: unknown): value is string {
  return quickTranslationShortcutOptions.some((option) => option.value === value);
}

function normalizeSettings(value: unknown): QuickTranslationSettings | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as Partial<QuickTranslationSettings>;
  if (typeof candidate.enabled !== "boolean" || !isSupportedShortcut(candidate.shortcut)) {
    return null;
  }
  return {
    enabled: candidate.enabled,
    shortcut: candidate.shortcut,
  };
}

export function loadPersistedQuickTranslationSettings(): string | null {
  if (persistedSettingsLoaded || typeof window === "undefined") {
    return null;
  }
  persistedSettingsLoaded = true;
  try {
    const raw = window.localStorage.getItem(QUICK_TRANSLATION_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const settings = normalizeSettings(JSON.parse(raw));
    if (!settings) {
      return "快捷翻译设置无效，已恢复默认值。";
    }
    quickTranslationSettings.value = settings;
    return null;
  } catch {
    return "无法读取快捷翻译设置，已使用默认值。";
  }
}

export function quickTranslationShortcutLabel(shortcut: string): string {
  return (
    quickTranslationShortcutOptions.find((option) => option.value === shortcut)?.label ?? shortcut
  );
}

async function applyShortcutSettings(settings: QuickTranslationSettings): Promise<void> {
  if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
    return;
  }
  await invoke("configure_quick_translation", {
    settings: {
      enabled: settings.enabled,
      shortcut: settings.shortcut,
    },
  });
}

export async function initializeQuickTranslationSettings(): Promise<string | null> {
  const loadMessage = loadPersistedQuickTranslationSettings();
  await applyShortcutSettings(quickTranslationSettings.value);
  return loadMessage;
}

export async function saveQuickTranslationSettings(
  value: QuickTranslationSettings,
): Promise<QuickTranslationSettings> {
  const settings = normalizeSettings(value);
  if (!settings) {
    throw new Error("快捷翻译设置不完整。");
  }

  const previousSettings = { ...quickTranslationSettings.value };
  await applyShortcutSettings(settings);
  try {
    window.localStorage.setItem(QUICK_TRANSLATION_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    await applyShortcutSettings(previousSettings).catch(() => undefined);
    throw new Error("无法保存快捷翻译设置，请检查应用存储权限。");
  }
  quickTranslationSettings.value = settings;
  return settings;
}
