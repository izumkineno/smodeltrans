import { invoke } from "@tauri-apps/api/core";
import { ref } from "vue";

const QT_LOG_PREFIX = " [quick-translation-settings]";

const QUICK_TRANSLATION_STORAGE_KEY = "smodeltrans.quickTranslationSettings";
const QUICK_TRANSLATION_MODIFIER_ALIASES: Record<string, string> = {
  ALT: "Alt",
  CMD: "Command",
  CMDORCONTROL: "CommandOrControl",
  CMDORCTRL: "CommandOrControl",
  COMMAND: "Command",
  COMMANDORCONTROL: "CommandOrControl",
  COMMANDORCTRL: "CommandOrControl",
  CONTROL: "Control",
  CTRL: "Control",
  OPTION: "Alt",
  SHIFT: "Shift",
  SUPER: "Super",
};
const QUICK_TRANSLATION_MODIFIER_CODES: Record<string, true> = {
  AltLeft: true,
  AltRight: true,
  ControlLeft: true,
  ControlRight: true,
  MetaLeft: true,
  MetaRight: true,
  ShiftLeft: true,
  ShiftRight: true,
};
const QUICK_TRANSLATION_KEY_PATTERN = /^(?:[A-Z]|[0-9]|Key[A-Z]|Digit[0-9]|Backquote|Backslash|BracketLeft|BracketRight|Pause|Comma|Equal|Minus|Period|Quote|Semicolon|Slash|Backspace|CapsLock|Enter|Space|Tab|Delete|End|Home|Insert|PageDown|PageUp|PrintScreen|ScrollLock|Arrow(?:Down|Left|Right|Up)|NumLock|Numpad(?:[0-9]|Add|Decimal|Divide|Enter|Equal|Multiply|Subtract)|Escape|F(?:[1-9]|1[0-9]|2[0-4])|AudioVolume(?:Down|Up|Mute)|Media(?:Play|Pause|PlayPause|Stop|TrackNext|TrackPrevious))$/;
const QUICK_TRANSLATION_KEY_LABELS: Record<string, string> = {
  ARROWDOWN: "Arrow Down",
  ARROWLEFT: "Arrow Left",
  ARROWRIGHT: "Arrow Right",
  ARROWUP: "Arrow Up",
  ALT: "Alt",
  BACKQUOTE: "`",
  BACKSLASH: "\\",
  BACKSPACE: "Backspace",
  BRACKETLEFT: "[",
  BRACKETRIGHT: "]",
  CAPSLOCK: "Caps Lock",
  CMD: "Win",
  CMDORCONTROL: "Ctrl",
  CMDORCTRL: "Ctrl",
  COMMAND: "Win",
  COMMANDORCONTROL: "Ctrl",
  COMMANDORCTRL: "Ctrl",
  COMMA: ",",
  CONTROL: "Ctrl",
  CTRL: "Ctrl",
  DELETE: "Delete",
  END: "End",
  ENTER: "Enter",
  ESCAPE: "Esc",
  EQUAL: "=",
  HOME: "Home",
  INSERT: "Insert",
  MINUS: "-",
  OPTION: "Alt",
  PAGEDOWN: "Page Down",
  PAGEUP: "Page Up",
  PERIOD: ".",
  PRINTSCREEN: "Print Screen",
  QUOTE: "'",
  SHIFT: "Shift",
  SLASH: "/",
  SPACE: "Space",
  SUPER: "Win",
  TAB: "Tab",
};
const DEFAULT_QUICK_TRANSLATION_SHORTCUT = "CommandOrControl+Alt+E";

export interface QuickTranslationSettings {
  enabled: boolean;
  shortcut: string;
}

const defaultSettings: QuickTranslationSettings = {
  enabled: true,
  shortcut: DEFAULT_QUICK_TRANSLATION_SHORTCUT,
};

export const quickTranslationSettings = ref<QuickTranslationSettings>({ ...defaultSettings });
let persistedSettingsLoaded = false;

function normalizeQuickTranslationShortcut(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const tokens = value
    .trim()
    .split("+")
    .map((token) => token.trim())
    .filter(Boolean);
  if (tokens.length < 2 || tokens.length > 5) {
    return null;
  }

  const modifiers = tokens
    .slice(0, -1)
    .map((token) => QUICK_TRANSLATION_MODIFIER_ALIASES[token.toUpperCase()] ?? null);
  const key = tokens[tokens.length - 1];
  if (
    modifiers.some((modifier) => modifier === null) ||
    new Set(modifiers).size !== modifiers.length ||
    !QUICK_TRANSLATION_KEY_PATTERN.test(key)
  ) {
    return null;
  }
  const normalizedModifiers = modifiers.filter(
    (modifier): modifier is string => modifier !== null,
  );
  return [...normalizedModifiers, key].join("+");
}

export function isQuickTranslationShortcutModifierCode(code: string): boolean {
  return QUICK_TRANSLATION_MODIFIER_CODES[code] === true;
}

export function quickTranslationShortcutFromKeyboardEvent(event: KeyboardEvent): string | null {
  const code = event.code.trim();
  if (
    !code ||
    isQuickTranslationShortcutModifierCode(code) ||
    !QUICK_TRANSLATION_KEY_PATTERN.test(code)
  ) {
    return null;
  }

  const modifiers: string[] = [];
  if (event.ctrlKey) {
    modifiers.push("CommandOrControl");
  }
  if (event.altKey) {
    modifiers.push("Alt");
  }
  if (event.shiftKey) {
    modifiers.push("Shift");
  }
  if (event.metaKey) {
    modifiers.push("Command");
  }
  if (modifiers.length === 0) {
    return null;
  }
  return [...modifiers, code].join("+");
}

function normalizeSettings(value: unknown): QuickTranslationSettings | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const candidate = value as Partial<QuickTranslationSettings>;
  const shortcut = normalizeQuickTranslationShortcut(candidate.shortcut);
  if (typeof candidate.enabled !== "boolean" || !shortcut) {
    return null;
  }
  return {
    enabled: candidate.enabled,
    shortcut,
  };
}

export function loadPersistedQuickTranslationSettings(): string | null {
  console.info(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings start`, { alreadyLoaded: persistedSettingsLoaded });
  if (persistedSettingsLoaded || typeof window === "undefined") {
    console.debug(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings skip`, { alreadyLoaded: persistedSettingsLoaded, hasWindow: typeof window !== "undefined" });
    return null;
  }
  persistedSettingsLoaded = true;
  try {
    const raw = window.localStorage.getItem(QUICK_TRANSLATION_STORAGE_KEY);
    console.debug(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings raw`, { raw });
    if (!raw) {
      console.info(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings no persisted value, using default`, { defaultShortcut: defaultSettings.shortcut });
      return null;
    }
    const settings = normalizeSettings(JSON.parse(raw));
    if (!settings) {
      console.warn(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings invalid, fallback to default`, { raw });
      return "快捷翻译设置无效，已恢复默认值。";
    }
    quickTranslationSettings.value = settings;
    console.info(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings success`, { enabled: settings.enabled, shortcut: settings.shortcut });
    return null;
  } catch (error) {
    console.warn(`${QT_LOG_PREFIX} loadPersistedQuickTranslationSettings failed`, { error: error instanceof Error ? error.message : String(error) });
    return "无法读取快捷翻译设置，已使用默认值。";
  }
}

function formatShortcutToken(token: string): string {
  const normalized = token.trim();
  const namedLabel = QUICK_TRANSLATION_KEY_LABELS[normalized.toUpperCase()];
  if (namedLabel) {
    return namedLabel;
  }
  const uppercase = normalized.toUpperCase();
  if (/^KEY[A-Z]$/.test(uppercase)) {
    return uppercase.slice(3);
  }
  if (/^DIGIT[0-9]$/.test(uppercase)) {
    return uppercase.slice(5);
  }
  if (/^NUMPAD[0-9]$/.test(uppercase)) {
    return `Numpad ${uppercase.slice(6)}`;
  }
  return normalized;
}

export function quickTranslationShortcutLabel(shortcut: string): string {
  return shortcut
    .split("+")
    .map(formatShortcutToken)
    .join(" + ");
}

async function applyShortcutSettings(settings: QuickTranslationSettings): Promise<void> {
  console.debug(`${QT_LOG_PREFIX} applyShortcutSettings start`, { enabled: settings.enabled, shortcut: settings.shortcut });
  if (!(typeof window !== "undefined" && "__TAURI_INTERNALS__" in window)) {
    console.debug(`${QT_LOG_PREFIX} applyShortcutSettings skip not Tauri`, { enabled: settings.enabled, shortcut: settings.shortcut });
    return;
  }
  try {
    await invoke("configure_quick_translation", {
      settings: {
        enabled: settings.enabled,
        shortcut: settings.shortcut,
      },
    });
    console.info(`${QT_LOG_PREFIX} applyShortcutSettings success`, { enabled: settings.enabled, shortcut: settings.shortcut });
  } catch (error) {
    console.error(`${QT_LOG_PREFIX} applyShortcutSettings failed`, { enabled: settings.enabled, shortcut: settings.shortcut, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function initializeQuickTranslationSettings(): Promise<string | null> {
  console.info(`${QT_LOG_PREFIX} initializeQuickTranslationSettings start`);
  const loadMessage = loadPersistedQuickTranslationSettings();
  if (loadMessage) {
    console.warn(`${QT_LOG_PREFIX} initializeQuickTranslationSettings loadMessage`, { loadMessage });
  }
  console.debug(`${QT_LOG_PREFIX} initializeQuickTranslationSettings applying`, { enabled: quickTranslationSettings.value.enabled, shortcut: quickTranslationSettings.value.shortcut });
  try {
    await applyShortcutSettings(quickTranslationSettings.value);
    console.info(`${QT_LOG_PREFIX} initializeQuickTranslationSettings success`, { enabled: quickTranslationSettings.value.enabled, shortcut: quickTranslationSettings.value.shortcut });
  } catch (error) {
    console.error(`${QT_LOG_PREFIX} initializeQuickTranslationSettings apply failed`, { error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
  return loadMessage;
}

export async function saveQuickTranslationSettings(
  value: QuickTranslationSettings,
): Promise<QuickTranslationSettings> {
  console.info(`${QT_LOG_PREFIX} saveQuickTranslationSettings start`, { enabled: value.enabled, shortcut: value.shortcut });
  const settings = normalizeSettings(value);
  if (!settings) {
    console.error(`${QT_LOG_PREFIX} saveQuickTranslationSettings invalid input`, { enabled: value.enabled, shortcut: value.shortcut });
    throw new Error("快捷翻译设置不完整。");
  }

  const previousSettings = { ...quickTranslationSettings.value };
  console.debug(`${QT_LOG_PREFIX} saveQuickTranslationSettings normalized`, { enabled: settings.enabled, shortcut: settings.shortcut });
  try {
    await applyShortcutSettings(settings);
    console.debug(`${QT_LOG_PREFIX} saveQuickTranslationSettings applied to backend`, { enabled: settings.enabled, shortcut: settings.shortcut });
  } catch (error) {
    console.error(`${QT_LOG_PREFIX} saveQuickTranslationSettings backend apply failed`, { error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
  try {
    window.localStorage.setItem(QUICK_TRANSLATION_STORAGE_KEY, JSON.stringify(settings));
    console.info(`${QT_LOG_PREFIX} saveQuickTranslationSettings persisted success`, { enabled: settings.enabled, shortcut: settings.shortcut });
  } catch (error) {
    console.warn(`${QT_LOG_PREFIX} saveQuickTranslationSettings localStorage failed, rolling back`, { error: error instanceof Error ? error.message : String(error) });
    await applyShortcutSettings(previousSettings).catch(() => undefined);
    throw new Error("无法保存快捷翻译设置，请检查应用存储权限。");
  }
  quickTranslationSettings.value = settings;
  console.info(`${QT_LOG_PREFIX} saveQuickTranslationSettings success`, { enabled: settings.enabled, shortcut: settings.shortcut });
  return settings;
}
