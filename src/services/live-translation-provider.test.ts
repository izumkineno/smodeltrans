import { describe, expect, test } from "bun:test";
import {
  LIVE_DEBUG_RECORD_EVENT,
  LIVE_REGION_BOX_VISIBILITY_EVENT,
  LIVE_STATUS_EVENT,
  LIVE_SUBTITLE_EVENT,
  beginLiveRoiUpdate,
  startLiveSession,
  cancelLiveSelection,
  confirmLiveSelection,
  getLiveSessionStatus,
  getLiveSubtitle,
  listCaptureWindows,
  listenLiveDebugRecord,
  listenLiveRegionBoxesVisible,
  listenLiveStatus,
  listenLiveSubtitle,
  interruptLiveTranslation,
  pauseLiveSession,
  resumeLiveSession,
  setLiveRegionBoxesVisible,
  shouldApplyLiveSubtitle,
  hasRenderableLiveSubtitleContent,
  mapLiveSubtitleRegionToOverlay,
  stopLiveSession,
  beginLiveOverlayDrag,
  beginLiveOverlayResize,
  finishLiveOverlayResize,
  updateLiveOverlayPosition,
  updateLiveOverlayLayout,
} from "./live-translation-provider";
import type {
  CaptureWindowInfo,
  LiveDebugRecord,
  LiveOverlaySettings,
  LiveRecognitionSettings,
  LiveRoi,
  LiveSessionStatus,
  LiveSubtitle,
  LiveSubtitleRegion,
  LiveTranslationSettings,
} from "./live-translation-provider";
const target: CaptureWindowInfo = {
  id: "778899",
  title: "Game Window",
  processName: "game.exe",
  processId: 4242,
  width: 1920,
  height: 1080,
  isMinimized: false,
};

const roi: LiveRoi = {
  x: 100,
  y: 700,
  width: 900,
  height: 220,
  clientWidth: 1920,
  clientHeight: 1080,
};

const overlaySettings: LiveOverlaySettings = {
  mode: "region_replace",
  attachment: "bottom",
  offset: 24,
  showSource: false,
  showRegionBoxes: true,
  autoWidth: true,
  autoHeight: false,
  manualWidth: 880,
  manualHeight: 144,
};

const recognitionSettings: LiveRecognitionSettings = {
  mode: "key_trigger",
  triggerKey: "F8",
  triggerEvent: "release",
  stabilityWaitMs: 800,
  keyTriggerTimeoutMs: 1_200,
  textGroupingEnabled: true,
};

const translationSettings: LiveTranslationSettings = {
  supplementalPrompt: "Keep character names and dialogue punctuation.",
  memoryEnabled: true,
  memoryMaxTokens: 4_096,
  memoryMaxTurns: 16,
};

const overlaySizing = {
  autoWidth: false,
  autoHeight: true,
  manualWidth: 720,
  manualHeight: 180,
};

const overlayContentSize = {
  width: 512,
  height: 96,
  minimumWidth: 320,
  minimumHeight: 72,
};

const status: LiveSessionStatus = {
  state: "running",
  sessionId: "live-1",
  target,
  roi,
  message: "实时翻译正在运行。",
  latestRevision: 3,
  metrics: {
    framesCaptured: 120,
    framesDropped: 4,
    framesSkippedUnchanged: 92,
    ocrRuns: 8,
    translationRuns: 5,
    subtitlePublishes: 3,
    lastOcrMs: 44,
    lastTranslationMs: 81,
    gpuName: "NVIDIA GeForce RTX 4090",
    gpuTotalMemoryMib: 24_564,
    gpuFreeMemoryMib: 19_200,
    gpuExecutionMode: "gpu_resident",
  },
};

describe("live translation command adapter", () => {
  test("uses exact command names and camel-case payload keys", async () => {
    const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
    const invokeFn = async <T>(command: string, args?: Record<string, unknown>): Promise<T> => {
      calls.push({ command, args });
      if (command === "list_capture_windows") {
        return [target] as T;
      }
      if (command === "get_live_subtitle") {
        return null as T;
      }
      return status as T;
    };

    await listCaptureWindows(invokeFn);
    await startLiveSession(
      target.id,
      "English",
      overlaySettings,
      recognitionSettings,
      translationSettings,
      invokeFn,
    );
    await confirmLiveSelection("live-1", roi, invokeFn);
    await beginLiveRoiUpdate("live-1", invokeFn);
    await cancelLiveSelection("live-1", invokeFn);
    await pauseLiveSession("live-1", invokeFn);
    await resumeLiveSession("live-1", invokeFn);
    await interruptLiveTranslation("live-1", invokeFn);
    await stopLiveSession("live-1", invokeFn);
    await stopLiveSession(undefined, invokeFn);
    await getLiveSessionStatus(invokeFn);
    await getLiveSubtitle(invokeFn);
    await updateLiveOverlayLayout("live-1", overlaySizing, overlayContentSize, invokeFn);
    await beginLiveOverlayDrag("live-1", invokeFn);
    await beginLiveOverlayResize("live-1", invokeFn);
    await finishLiveOverlayResize("live-1", invokeFn);
    await updateLiveOverlayPosition("live-1", { x: 320, y: 240 }, invokeFn);

    expect(calls).toEqual([
      { command: "list_capture_windows", args: undefined },
      {
        command: "start_live_session",
        args: {
          targetId: "778899",
          targetLanguage: "English",
          overlaySettings,
          recognitionSettings,
          translationSettings,
        },
      },
      {
        command: "confirm_live_selection",
        args: { sessionId: "live-1", roi },
      },
      {
        command: "begin_live_roi_update",
        args: { sessionId: "live-1" },
      },
      {
        command: "cancel_live_selection",
        args: { sessionId: "live-1" },
      },
      {
        command: "pause_live_session",
        args: { sessionId: "live-1" },
      },
      {
        command: "resume_live_session",
        args: { sessionId: "live-1" },
      },
      {
        command: "interrupt_live_translation",
        args: { sessionId: "live-1" },
      },
      {
        command: "stop_live_session",
        args: { sessionId: "live-1" },
      },
      { command: "stop_live_session", args: {} },
      { command: "get_live_session_status", args: undefined },
      { command: "get_live_subtitle", args: undefined },
      {
        command: "update_live_overlay_layout",
        args: {
          sessionId: "live-1",
          sizing: overlaySizing,
          contentSize: overlayContentSize,
        },
      },
      {
        command: "begin_live_overlay_drag",
        args: { sessionId: "live-1" },
      },
      {
        command: "begin_live_overlay_resize",
        args: { sessionId: "live-1" },
      },
      {
        command: "finish_live_overlay_resize",
        args: { sessionId: "live-1" },
      },
      {
        command: "update_live_overlay_position",
        args: {
          sessionId: "live-1",
          position: { x: 320, y: 240 },
        },
      },
    ]);
  });

  test("maps the exact event names to typed payload callbacks", async () => {
    const events: string[] = [];
    const receivedStatuses: LiveSessionStatus[] = [];
    const receivedSubtitles: LiveSubtitle[] = [];
    const receivedDebugRecords: LiveDebugRecord[] = [];
    const subtitle: LiveSubtitle = {
      sessionId: "live-1",
      revision: 3,
      sourceText: "原文",
      translatedText: "Translation",
      roi,
      regions: [],
      isStreaming: false,
      observedAtEpochMs: 1_700_000_000_000,
    };
    const debugRecord: LiveDebugRecord = {
      sessionId: "live-1",
      sequence: 1,
      stage: "ocr",
      outcome: "confirmed",
      sourceText: "原文",
      targetLanguage: "English",
      regionCount: 2,
      roiVersion: 1,
      durationMs: 44,
      observedAtEpochMs: 1_700_000_000_000,
    };
    const listenFn = async <T>(
      event: string,
      handler: (event: { event: string; id: number; payload: T }) => void,
    ): Promise<() => void> => {
      events.push(event);
      handler({
        event,
        id: events.length,
        payload: (
          event === LIVE_STATUS_EVENT
            ? status
            : event === LIVE_SUBTITLE_EVENT
              ? subtitle
              : debugRecord
        ) as T,
      });
    };

    await listenLiveStatus((payload) => receivedStatuses.push(payload), listenFn);
    await listenLiveSubtitle((payload) => receivedSubtitles.push(payload), listenFn);
    await listenLiveDebugRecord((payload) => receivedDebugRecords.push(payload), listenFn);

    expect(events).toEqual([LIVE_STATUS_EVENT, LIVE_SUBTITLE_EVENT, LIVE_DEBUG_RECORD_EVENT]);
    expect(receivedStatuses).toEqual([status]);
    expect(receivedSubtitles).toEqual([subtitle]);
    expect(receivedDebugRecords).toEqual([debugRecord]);
  });

  test("broadcasts and listens for region box visibility", async () => {
    const emitted: Array<{ event: string; payload?: unknown }> = [];
    const emitFn = async <T>(event: string, payload?: T): Promise<void> => {
      emitted.push({ event, payload });
    };
    const received: boolean[] = [];
    const listenFn = async <T>(
      event: string,
      handler: (event: { event: string; id: number; payload: T }) => void,
    ): Promise<() => void> => {
      handler({ event, id: 1, payload: true as T });
      return () => {};
    };

    await setLiveRegionBoxesVisible(true, emitFn);
    await listenLiveRegionBoxesVisible((visible) => received.push(visible), listenFn);

    expect(emitted).toEqual([
      { event: LIVE_REGION_BOX_VISIBILITY_EVENT, payload: true },
    ]);
    expect(received).toEqual([true]);
  });
});

describe("live subtitle revision filtering", () => {
  const subtitle: LiveSubtitle = {
    sessionId: "live-1",
    revision: 7,
    sourceText: "原文",
    translatedText: "Translation",
    roi,
    regions: [],
    isStreaming: false,
    observedAtEpochMs: 1_700_000_000_000,
  };

  test("accepts the active session at the same or a newer revision", () => {
    expect(shouldApplyLiveSubtitle(subtitle, "live-1", 7)).toBe(true);
    expect(shouldApplyLiveSubtitle({ ...subtitle, revision: 8 }, "live-1", 7)).toBe(true);
  });

  test("rejects lower revisions and events from another session", () => {
    expect(shouldApplyLiveSubtitle({ ...subtitle, revision: 6 }, "live-1", 7)).toBe(false);
    expect(shouldApplyLiveSubtitle(subtitle, "live-2", 1)).toBe(false);
    expect(shouldApplyLiveSubtitle(subtitle, undefined, -1)).toBe(false);
  });
  test("keeps region payload when aggregate subtitle text is empty", () => {
    expect(
      hasRenderableLiveSubtitleContent({
        ...subtitle,
        sourceText: "",
        translatedText: "",
        regions: [
          {
            bounds: { left: 10, top: 20, width: 100, height: 24 },
            sourceText: "原文",
            translatedText: "译文",
          },
        ],
      }),
    ).toBe(true);
    expect(
      hasRenderableLiveSubtitleContent({
        ...subtitle,
        sourceText: "",
        translatedText: "",
        regions: [],
        isStreaming: false,
      }),
    ).toBe(false);
  });
});

describe("live subtitle region coordinate mapping", () => {
  const region: LiveSubtitleRegion = {
    bounds: { left: 960, top: 270, width: 480, height: 54 },
    sourceText: "原文",
    translatedText: "译文",
  };

  test("maps source-image pixels into the measured overlay viewport", () => {
    expect(mapLiveSubtitleRegionToOverlay(region, roi, 960, 540)).toEqual({
      x: 480,
      y: 135,
      width: 240,
      height: 27,
    });
  });

  test("rejects invalid source or viewport dimensions", () => {
    expect(
      mapLiveSubtitleRegionToOverlay(region, { ...roi, clientWidth: 0 }, 960, 540),
    ).toBeUndefined();
    expect(mapLiveSubtitleRegionToOverlay(region, roi, 0, 540)).toBeUndefined();
    expect(
      mapLiveSubtitleRegionToOverlay(
        { ...region, bounds: { ...region.bounds, width: 0 } },
        roi,
        960,
        540,
      ),
    ).toBeUndefined();
  });
});
