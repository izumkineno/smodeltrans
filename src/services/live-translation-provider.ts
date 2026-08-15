import { invoke } from "@tauri-apps/api/core";
import { emit, listen } from "@tauri-apps/api/event";
import type { EventCallback, UnlistenFn } from "@tauri-apps/api/event";

export const LIVE_STATUS_EVENT = "live-status";
export const LIVE_SUBTITLE_EVENT = "live-subtitle";
export const LIVE_DEBUG_RECORD_EVENT = "live-debug-record";
export const LIVE_REGION_BOX_VISIBILITY_EVENT = "live-region-box-visibility";

export interface CaptureWindowInfo {
  id: string;
  title: string;
  processName: string;
  processId: number;
  width: number;
  height: number;
  isMinimized: boolean;
}

export interface LiveRoi {
  x: number;
  y: number;
  width: number;
  height: number;
  clientWidth: number;
  clientHeight: number;
}

export interface LiveMetrics {
  framesCaptured: number;
  framesDropped: number;
  framesSkippedUnchanged: number;
  ocrRuns: number;
  translationRuns: number;
  subtitlePublishes: number;
  lastOcrMs: number;
  lastTranslationMs: number;
  gpuName: string;
  gpuTotalMemoryMib: number;
  gpuFreeMemoryMib: number;
  gpuExecutionMode: string;
}

export type LiveSessionState =
  | "idle"
  | "selecting"
  | "warming"
  | "running"
  | "paused"
  | "stopping"
  | "error";

export interface LiveSessionStatus {
  state: LiveSessionState;
  sessionId?: string;
  target?: CaptureWindowInfo;
  roi?: LiveRoi;
  message: string;
  latestRevision: number;
  metrics: LiveMetrics;
}

export type LiveOverlayMode = "subtitle" | "region_replace";

export type LiveOverlayAttachment = "top" | "bottom" | "left" | "right";

export interface LiveOverlaySettings {
  mode: LiveOverlayMode;
  attachment: LiveOverlayAttachment;
  offset: number;
  showSource: boolean;
  showRegionBoxes: boolean;
  autoWidth: boolean;
  autoHeight: boolean;
  manualWidth: number;
  manualHeight: number;
}

export type LiveOverlaySizing = Pick<
  LiveOverlaySettings,
  "autoWidth" | "autoHeight" | "manualWidth" | "manualHeight"
>;

export interface LiveOverlayContentSize {
  width: number;
  height: number;
}

export type LiveRecognitionMode = "automatic" | "key_trigger";

export type LiveRecognitionTrigger = "press" | "release";

export interface LiveRecognitionSettings {
  mode: LiveRecognitionMode;
  triggerKey: string;
  triggerEvent: LiveRecognitionTrigger;
  stabilityWaitMs: number;
  keyTriggerTimeoutMs: number;
  textGroupingEnabled: boolean;
}

export const LIVE_SUPPLEMENTAL_PROMPT_MAX_CHARS = 4_096;
export const LIVE_MEMORY_TOKENS_MIN = 1;
export const LIVE_MEMORY_TOKENS_MAX = 262_144;
export const LIVE_MEMORY_TURNS_MIN = 1;
export const LIVE_MEMORY_TURNS_MAX = 1_024;
export const DEFAULT_LIVE_MEMORY_ENABLED = true;
export const DEFAULT_LIVE_MEMORY_TOKENS = 4_096;
export const DEFAULT_LIVE_MEMORY_TURNS = 16;

export interface LiveTranslationSettings {
  supplementalPrompt: string;
  memoryEnabled: boolean;
  memoryMaxTokens: number;
  memoryMaxTurns: number;
}

export interface LiveSubtitleRegion {
  bounds: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
  sourceText: string;
  translatedText: string;
}

export interface LiveSubtitleRegionFlowItem {
  id: string;
  index: number;
  region: LiveSubtitleRegion;
  leftOffset: number;
  width: number;
  gapAbove: number;
}

export interface LiveSubtitleRegionFlowGroup {
  id: string;
  left: number;
  top: number;
  bottom: number;
  width: number;
  items: LiveSubtitleRegionFlowItem[];
}

export interface LiveSubtitleRegionVerticalAnchor {
  edge: "top" | "bottom";
  offset: number;
}

export interface LiveSubtitle {
  sessionId: string;
  revision: number;
  sourceText: string;
  translatedText: string;
  roi: LiveRoi;
  regions: LiveSubtitleRegion[];
  isStreaming: boolean;
  observedAtEpochMs: number;
}

export type LiveDebugStage = "ocr" | "translation";

export type LiveDebugOutcome =
  | "confirmed"
  | "completed"
  | "skipped_empty_source"
  | "failed";

export interface LiveDebugRecord {
  sessionId: string;
  sequence: number;
  stage: LiveDebugStage;
  outcome: LiveDebugOutcome;
  sourceText: string;
  translatedText?: string;
  targetLanguage: string;
  regionCount: number;
  roiVersion: number;
  durationMs: number;
  message?: string;
  observedAtEpochMs: number;
}

type InvokeFn = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
type ListenFn = <T>(event: string, handler: EventCallback<T>) => Promise<UnlistenFn>;
type EmitFn = <T>(event: string, payload?: T) => Promise<void>;

export function listCaptureWindows(invokeFn: InvokeFn = invoke): Promise<CaptureWindowInfo[]> {
  return invokeFn<CaptureWindowInfo[]>("list_capture_windows");
}

export function beginLiveSelection(
  targetId: string,
  targetLanguage: string,
  overlaySettings: LiveOverlaySettings,
  recognitionSettings: LiveRecognitionSettings,
  translationSettings: LiveTranslationSettings,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("begin_live_selection", {
    targetId,
    targetLanguage,
    overlaySettings,
    recognitionSettings,
    translationSettings,
  });
}

export function confirmLiveSelection(
  sessionId: string,
  roi: LiveRoi,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("confirm_live_selection", { sessionId, roi });
}


export function beginLiveRoiUpdate(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("begin_live_roi_update", { sessionId });
}

export function cancelLiveSelection(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("cancel_live_selection", { sessionId });
}

export function pauseLiveSession(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("pause_live_session", { sessionId });
}

export function resumeLiveSession(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("resume_live_session", { sessionId });
}

export function stopLiveSession(
  sessionId?: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>(
    "stop_live_session",
    sessionId === undefined ? {} : { sessionId },
  );
}
export function interruptLiveTranslation(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("interrupt_live_translation", { sessionId });
}

export function getLiveSessionStatus(invokeFn: InvokeFn = invoke): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("get_live_session_status");
}

export function updateLiveOverlayLayout(
  sessionId: string,
  sizing: LiveOverlaySizing,
  contentSize: LiveOverlayContentSize,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  return invokeFn<void>("update_live_overlay_layout", { sessionId, sizing, contentSize });
}

export function listenLiveStatus(
  handler: (status: LiveSessionStatus) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  return listenFn<LiveSessionStatus>(LIVE_STATUS_EVENT, (event) => handler(event.payload));
}

export function listenLiveSubtitle(
  handler: (subtitle: LiveSubtitle) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  return listenFn<LiveSubtitle>(LIVE_SUBTITLE_EVENT, (event) => handler(event.payload));
}
export function setLiveRegionBoxesVisible(
  visible: boolean,
  emitFn: EmitFn = emit,
): Promise<void> {
  return emitFn<boolean>(LIVE_REGION_BOX_VISIBILITY_EVENT, visible);
}

export function listenLiveRegionBoxesVisible(
  handler: (visible: boolean) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  return listenFn<boolean>(LIVE_REGION_BOX_VISIBILITY_EVENT, (event) =>
    handler(event.payload),
  );
}


export function listenLiveDebugRecord(
  handler: (record: LiveDebugRecord) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  return listenFn<LiveDebugRecord>(LIVE_DEBUG_RECORD_EVENT, (event) => handler(event.payload));
}

export function shouldApplyLiveSubtitle(
  subtitle: LiveSubtitle,
  activeSessionId: string | undefined,
  lastAppliedRevision: number,
): boolean {
  return (
    activeSessionId !== undefined &&
    subtitle.sessionId === activeSessionId &&
    subtitle.revision >= lastAppliedRevision
  );
}
export function groupLiveSubtitleRegions(
  regions: LiveSubtitleRegion[],
): LiveSubtitleRegionFlowGroup[] {
  const prepared = regions
    .map((region, index) => ({
      region,
      index,
      left: region.bounds.left,
      top: region.bounds.top,
      right: region.bounds.left + region.bounds.width,
      bottom: region.bounds.top + region.bounds.height,
    }))
    .filter(
      ({ left, top, right, bottom }) =>
        [left, top, right, bottom].every(Number.isFinite) &&
        right > left &&
        bottom > top,
    )
    .sort(
      (left, right) =>
        left.left - right.left || left.top - right.top || left.index - right.index,
    );
  const columns: Array<{
    right: number;
    items: typeof prepared;
  }> = [];
  for (const item of prepared) {
    const column = columns[columns.length - 1];
    if (!column || item.left >= column.right) {
      columns.push({ right: item.right, items: [item] });
      continue;
    }
    column.items.push(item);
    column.right = Math.max(column.right, item.right);
  }

  const blocks: Array<typeof prepared> = [];
  for (const column of columns) {
    const ordered = column.items.sort(
      (left, right) =>
        left.top - right.top || left.left - right.left || left.index - right.index,
    );
    let block: typeof prepared = [];
    let blockBottom = Number.NEGATIVE_INFINITY;
    let blockLineHeight = 0;
    for (const item of ordered) {
      const itemHeight = item.bottom - item.top;
      const gap = item.top - blockBottom;
      const splitThreshold = Math.max(24, Math.max(blockLineHeight, itemHeight) * 1.5);
      if (block.length > 0 && gap > splitThreshold) {
        blocks.push(block);
        block = [];
        blockBottom = Number.NEGATIVE_INFINITY;
        blockLineHeight = 0;
      }
      block.push(item);
      blockBottom = Math.max(blockBottom, item.bottom);
      blockLineHeight = Math.max(blockLineHeight, itemHeight);
    }
    if (block.length > 0) {
      blocks.push(block);
    }
  }

  return blocks.map((block, groupIndex) => {
    const left = Math.min(...block.map((item) => item.left));
    const right = Math.max(...block.map((item) => item.right));
    const top = Math.min(...block.map((item) => item.top));
    const bottom = Math.max(...block.map((item) => item.bottom));
    let previousBottom: number | undefined;
    const items = block.map((item) => {
      const gapAbove =
        previousBottom === undefined ? 0 : Math.max(4, item.top - previousBottom);
      previousBottom =
        previousBottom === undefined
          ? item.bottom
          : Math.max(previousBottom, item.bottom);
      return {
        id: `${item.index}-${item.left}-${item.top}-${item.region.sourceText}`,
        index: item.index,
        region: item.region,
        leftOffset: item.left - left,
        width: item.right - item.left,
        gapAbove,
      };
    });
    return {
      id: `group-${groupIndex}-${left}-${top}`,
      left,
      top,
      bottom,
      width: right - left,
      items,
    };
  });
}

export function resolveLiveSubtitleRegionVerticalAnchor(
  group: LiveSubtitleRegionFlowGroup,
  clientHeight: number,
): LiveSubtitleRegionVerticalAnchor | undefined {
  if (
    !Number.isFinite(clientHeight) ||
    clientHeight <= 0 ||
    !Number.isFinite(group.top) ||
    !Number.isFinite(group.bottom) ||
    group.bottom <= group.top
  ) {
    return undefined;
  }
  const top = Math.max(0, Math.min(clientHeight, group.top));
  const bottom = Math.max(top, Math.min(clientHeight, group.bottom));
  return top + bottom <= clientHeight
    ? { edge: "top", offset: top }
    : { edge: "bottom", offset: clientHeight - bottom };
}
