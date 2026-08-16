import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import {
  loadPersistedThemeMode,
  resolvedTheme,
  setThemeMode,
  themeMode,
} from "./theme-settings";

const stored = new Map<string, string>();
const originalWindow = globalThis.window;
const originalDocument = globalThis.document;
let systemThemeListener: ((event: MediaQueryListEvent) => void) | undefined;

beforeAll(() => {
  const mediaQuery = {
    matches: false,
    media: "(prefers-color-scheme: dark)",
    onchange: null,
    addEventListener: (_type: string, listener: (event: MediaQueryListEvent) => void) => {
      systemThemeListener = listener;
    },
    removeEventListener: () => undefined,
    addListener: (listener: (event: MediaQueryListEvent) => void) => {
      systemThemeListener = listener;
    },
    removeListener: () => undefined,
    dispatchEvent: () => false,
  } as unknown as MediaQueryList;

  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem: (key: string) => stored.get(key) ?? null,
        setItem: (key: string, value: string) => stored.set(key, value),
      },
      matchMedia: () => mediaQuery,
    },
  });
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: {
      documentElement: {
        dataset: {} as DOMStringMap,
        style: { colorScheme: "" },
      },
    },
  });
});

afterAll(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: originalWindow,
  });
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: originalDocument,
  });
});

describe("theme mode persistence", () => {
  test("loads system mode and follows operating system changes", () => {
    stored.set("smodeltrans.themeMode", "system");

    expect(loadPersistedThemeMode()).toBeNull();
    expect(themeMode.value).toBe("system");
    expect(resolvedTheme.value).toBe("light");

    systemThemeListener?.({ matches: true } as MediaQueryListEvent);
    expect(resolvedTheme.value).toBe("dark");

    systemThemeListener?.({ matches: false } as MediaQueryListEvent);
    expect(resolvedTheme.value).toBe("light");
  });

  test("persists explicit modes and ignores system changes", () => {
    expect(setThemeMode("dark")).toBeNull();
    expect(stored.get("smodeltrans.themeMode")).toBe("dark");
    expect(resolvedTheme.value).toBe("dark");

    systemThemeListener?.({ matches: false } as MediaQueryListEvent);
    expect(resolvedTheme.value).toBe("dark");

    expect(setThemeMode("light")).toBeNull();
    expect(stored.get("smodeltrans.themeMode")).toBe("light");
    expect(resolvedTheme.value).toBe("light");
  });
});
