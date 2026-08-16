import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import {
  DEFAULT_KEY_TRIGGER_TIMEOUT_MS,
  liveRecognitionSettings,
  liveSubtitleStyleSettings,
  liveTranslationSettings,
  loadPersistedLiveRecognitionSettings,
  loadPersistedLiveSubtitleStyleSettings,
  loadPersistedLiveTranslationSettings,
  savePersistedLiveRecognitionSettings,
  savePersistedLiveSubtitleStyleSettings,
  savePersistedLiveTranslationSettings,
} from "./workspace-settings";

const stored = new Map<string, string>();
const originalWindow = globalThis.window;

beforeAll(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      localStorage: {
        getItem: (key: string) => stored.get(key) ?? null,
        setItem: (key: string, value: string) => stored.set(key, value),
      },
    },
  });
});

afterAll(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: originalWindow,
  });
});

describe("live translation settings persistence", () => {
  test("migrates prior prompt-only settings and persists memory controls", () => {
    stored.set(
      "smodeltrans.liveTranslationSettings",
      JSON.stringify({
        supplementalPrompt: "  Keep dialogue punctuation.  ",
      }),
    );

    expect(loadPersistedLiveTranslationSettings()).toBeNull();
    expect(liveTranslationSettings.value).toMatchObject({
      supplementalPrompt: "  Keep dialogue punctuation.  ",
      memoryEnabled: true,
      memoryMaxTokens: 4_096,
      memoryMaxTurns: 16,
    });

    liveTranslationSettings.value = {
      supplementalPrompt: "  Preserve names.  ",
      memoryEnabled: true,
      memoryMaxTokens: 8_192,
      memoryMaxTurns: 8,
    };
    expect(savePersistedLiveTranslationSettings()).toBeNull();
    expect(
      JSON.parse(stored.get("smodeltrans.liveTranslationSettings") ?? "{}"),
    ).toEqual({
      supplementalPrompt: "Preserve names.",
      memoryEnabled: true,
      memoryMaxTokens: 8_192,
      memoryMaxTurns: 8,
    });
  });
});

describe("live recognition settings persistence", () => {
  test("migrates legacy settings and persists the configurable trigger timeout", () => {
    stored.set(
      "smodeltrans.liveRecognitionSettings",
      JSON.stringify({
        mode: "key_trigger",
        triggerKey: "F8",
        triggerEvent: "press",
        stabilityWaitMs: 300,
        textGroupingEnabled: true,
      }),
    );

    expect(loadPersistedLiveRecognitionSettings()).toBeNull();
    expect(liveRecognitionSettings.value.keyTriggerTimeoutMs).toBe(
      DEFAULT_KEY_TRIGGER_TIMEOUT_MS,
    );

    liveRecognitionSettings.value.keyTriggerTimeoutMs = 1_500;
    expect(savePersistedLiveRecognitionSettings()).toBeNull();
    expect(
      JSON.parse(stored.get("smodeltrans.liveRecognitionSettings") ?? "{}"),
    ).toMatchObject({ keyTriggerTimeoutMs: 1_500 });
  });
});

describe("live subtitle style persistence", () => {
  test("loads, validates, and saves subtitle appearance controls", () => {
    stored.set(
      "smodeltrans.liveSubtitleStyleSettings",
      JSON.stringify({
        fontColor: "#123456",
        fontSize: 32,
        backgroundColor: "#abcdef",
        backgroundOpacity: 64,
      }),
    );

    expect(loadPersistedLiveSubtitleStyleSettings()).toBeNull();
    expect(liveSubtitleStyleSettings.value).toEqual({
      fontColor: "#123456",
      fontSize: 32,
      backgroundColor: "#abcdef",
      backgroundOpacity: 64,
    });

    liveSubtitleStyleSettings.value = {
      fontColor: "#0f172a",
      fontSize: 20,
      backgroundColor: "#e2e8f0",
      backgroundOpacity: 80,
    };
    expect(savePersistedLiveSubtitleStyleSettings()).toBeNull();
    expect(
      JSON.parse(stored.get("smodeltrans.liveSubtitleStyleSettings") ?? "{}"),
    ).toEqual({
      fontColor: "#0f172a",
      fontSize: 20,
      backgroundColor: "#e2e8f0",
      backgroundOpacity: 80,
    });
  });
});
