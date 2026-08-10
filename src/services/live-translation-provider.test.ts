import { describe, expect, test } from "bun:test";
import {
  LIVE_DEBUG_RECORD_EVENT,
  LIVE_STATUS_EVENT,
  LIVE_SUBTITLE_EVENT,
  beginLiveRoiUpdate,
  beginLiveSelection,
  cancelLiveSelection,
  confirmLiveSelection,
  getLiveSessionStatus,
  listCaptureWindows,
  listenLiveDebugRecord,
  listenLiveStatus,
  listenLiveSubtitle,
  pauseLiveSession,
  resolveLiveSubtitleRegionStyle,
  resumeLiveSession,
  shouldApplyLiveSubtitle,
  stopLiveSession,
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
} from "./live-translation-provider";
const target: CaptureWindowInfo = {
  id: "778899",
  title: "Game Window",
  processName: "game.exe",
  processId: 4242,
  width: 1920,
  height: 1080,
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
};

const recognitionSettings: LiveRecognitionSettings = {
  mode: "key_trigger",
  triggerKey: "F8",
  triggerEvent: "release",
  stabilityWaitMs: 800,
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
    ocrRuns: 8,
    translationRuns: 5,
    cacheHits: 2,
    subtitlePublishes: 3,
    lastOcrMs: 44,
    lastTranslationMs: 81,
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
      return status as T;
    };

    await listCaptureWindows(invokeFn);
    await beginLiveSelection(target.id, "English", overlaySettings, recognitionSettings, invokeFn);
    await confirmLiveSelection("live-1", roi, invokeFn);
    await beginLiveRoiUpdate("live-1", invokeFn);
    await cancelLiveSelection("live-1", invokeFn);
    await pauseLiveSession("live-1", invokeFn);
    await resumeLiveSession("live-1", invokeFn);
    await stopLiveSession("live-1", invokeFn);
    await stopLiveSession(undefined, invokeFn);
    await getLiveSessionStatus(invokeFn);

    expect(calls).toEqual([
      { command: "list_capture_windows", args: undefined },
      {
        command: "begin_live_selection",
        args: {
          targetId: "778899",
          targetLanguage: "English",
          overlaySettings,
          recognitionSettings,
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
        command: "stop_live_session",
        args: { sessionId: "live-1" },
      },
      { command: "stop_live_session", args: {} },
      { command: "get_live_session_status", args: undefined },
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
      cacheHit: false,
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
});

describe("live subtitle revision filtering", () => {
  const subtitle: LiveSubtitle = {
    sessionId: "live-1",
    revision: 7,
    sourceText: "原文",
    translatedText: "Translation",
    roi,
    regions: [],
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
});

describe("live subtitle region styles", () => {
  const roi: LiveRoi = {
    x: 100,
    y: 700,
    width: 900,
    height: 220,
    clientWidth: 1920,
    clientHeight: 1080,
  };
  const region: LiveSubtitleRegion = {
    quad: [
      [10, 20],
      [210, 20],
      [210, 80],
      [10, 80],
    ],
    sourceText: "字幕",
    translatedText: "Subtitle",
  };

  test("maps OCR coordinates to a fixed region box", () => {
    const style = resolveLiveSubtitleRegionStyle(region, roi);

    expect(Number.parseFloat(style.left)).toBeCloseTo(5.7291666667, 8);
    expect(Number.parseFloat(style.top)).toBeCloseTo(66.6666666667, 8);
    expect(Number.parseFloat(style.width)).toBeCloseTo(10.4166666667, 8);
    expect(Number.parseFloat(style.height)).toBeCloseTo(5.5555555556, 8);
  });

  test("clamps regions to the overlay and rejects invalid ROI sizes", () => {
    const outsideRegion: LiveSubtitleRegion = {
      ...region,
      quad: [
        [-1000, -1000],
        [2500, -1000],
        [2500, 1000],
        [-1000, 1000],
      ],
    };

    expect(resolveLiveSubtitleRegionStyle(outsideRegion, roi)).toEqual({
      left: "0%",
      top: "0%",
      width: "100%",
      height: "100%",
    });
  });
  test("rejects invalid ROI sizes", () => {
    expect(
      resolveLiveSubtitleRegionStyle(region, { ...roi, clientWidth: 0 }),
    ).toEqual({});
  });

});
