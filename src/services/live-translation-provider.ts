import { invoke } from "@tauri-apps/api/core";
import { emit, listen } from "@tauri-apps/api/event";
import type { EventCallback, UnlistenFn } from "@tauri-apps/api/event";

const LIVE_LOG_PREFIX = " [live-translation-provider]";

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
  minimumWidth: number;
  minimumHeight: number;
}
export interface LiveOverlayPosition {
  x: number;
  y: number;
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
export type SubtitleProgressMode = "translation" | "live";

export interface SubtitleProgress {
  mode: SubtitleProgressMode;
  active: boolean;
  overall: number;
  ocr: number;
  translation: number;
  label: string;
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

export async function listCaptureWindows(invokeFn: InvokeFn = invoke): Promise<CaptureWindowInfo[]> {
  console.info(`${LIVE_LOG_PREFIX} listCaptureWindows start`);
  const start = Date.now();
  try {
    const windows = await invokeFn<CaptureWindowInfo[]>("list_capture_windows");
    console.info(`${LIVE_LOG_PREFIX} listCaptureWindows success`, { count: windows.length, durationMs: Date.now() - start });
    console.debug(`${LIVE_LOG_PREFIX} listCaptureWindows detail`, { windows: windows.slice(0,3).map(w=>({id: (w as any).id ?? (w as any).windowId, title: (w as any).title})) });
    return windows;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} listCaptureWindows failed`, { error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
    throw error;
  }
}

export async function startLiveSession(
  targetId: string,
  targetLanguage: string,
  overlaySettings: LiveOverlaySettings,
  recognitionSettings: LiveRecognitionSettings,
  translationSettings: LiveTranslationSettings,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} startLiveSession start`, { targetId, targetLanguage });
  console.debug(`${LIVE_LOG_PREFIX} startLiveSession config`, { overlaySettings, recognitionSettings, translationSettings });
  const start = Date.now();
  try {
    const status = await invokeFn<LiveSessionStatus>("start_live_session", {
      targetId,
      targetLanguage,
      overlaySettings,
      recognitionSettings,
      translationSettings,
    });
    console.info(`${LIVE_LOG_PREFIX} startLiveSession success`, { targetId, targetLanguage, sessionId: status.sessionId, state: status.state, durationMs: Date.now() - start });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} startLiveSession failed`, { targetId, targetLanguage, error: error instanceof Error ? error.message : String(error), durationMs: Date.now() - start });
    throw error;
  }
}

export async function confirmLiveSelection(
  sessionId: string,
  roi: LiveRoi,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} confirmLiveSelection start`, { sessionId, roi });
  try {
    const status = await invokeFn<LiveSessionStatus>("confirm_live_selection", { sessionId, roi });
    console.info(`${LIVE_LOG_PREFIX} confirmLiveSelection success`, { sessionId, state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} confirmLiveSelection failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}


export async function beginLiveRoiUpdate(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} beginLiveRoiUpdate start`, { sessionId });
  try {
    const status = await invokeFn<LiveSessionStatus>("begin_live_roi_update", { sessionId });
    console.info(`${LIVE_LOG_PREFIX} beginLiveRoiUpdate success`, { sessionId, state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} beginLiveRoiUpdate failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function cancelLiveSelection(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} cancelLiveSelection start`, { sessionId });
  try {
    const status = await invokeFn<LiveSessionStatus>("cancel_live_selection", { sessionId });
    console.info(`${LIVE_LOG_PREFIX} cancelLiveSelection success`, { sessionId, state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} cancelLiveSelection failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function pauseLiveSession(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} pauseLiveSession start`, { sessionId });
  try {
    const status = await invokeFn<LiveSessionStatus>("pause_live_session", { sessionId });
    console.info(`${LIVE_LOG_PREFIX} pauseLiveSession success`, { sessionId, state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} pauseLiveSession failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function resumeLiveSession(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} resumeLiveSession start`, { sessionId });
  try {
    const status = await invokeFn<LiveSessionStatus>("resume_live_session", { sessionId });
    console.info(`${LIVE_LOG_PREFIX} resumeLiveSession success`, { sessionId, state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} resumeLiveSession failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function stopLiveSession(
  sessionId?: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} stopLiveSession start`, { sessionId: sessionId ?? "all" });
  try {
    const status = await invokeFn<LiveSessionStatus>(
      "stop_live_session",
      sessionId === undefined ? {} : { sessionId },
    );
    console.info(`${LIVE_LOG_PREFIX} stopLiveSession success`, { sessionId: sessionId ?? "all", state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} stopLiveSession failed`, { sessionId: sessionId ?? "all", error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}
export async function interruptLiveTranslation(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  console.info(`${LIVE_LOG_PREFIX} interruptLiveTranslation start`, { sessionId });
  try {
    const status = await invokeFn<LiveSessionStatus>("interrupt_live_translation", { sessionId });
    console.info(`${LIVE_LOG_PREFIX} interruptLiveTranslation success`, { sessionId, state: status.state });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} interruptLiveTranslation failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function getLiveSessionStatus(invokeFn: InvokeFn = invoke): Promise<LiveSessionStatus> {
  console.debug(`${LIVE_LOG_PREFIX} getLiveSessionStatus start`);
  try {
    const status = await invokeFn<LiveSessionStatus>("get_live_session_status");
    console.info(`${LIVE_LOG_PREFIX} getLiveSessionStatus success`, { sessionId: status.sessionId, state: status.state });
    console.debug(`${LIVE_LOG_PREFIX} getLiveSessionStatus detail`, { status });
    return status;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} getLiveSessionStatus failed`, { error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function getLiveSubtitle(invokeFn: InvokeFn = invoke): Promise<LiveSubtitle | null> {
  console.debug(`${LIVE_LOG_PREFIX} getLiveSubtitle start`);
  try {
    const subtitle = await invokeFn<LiveSubtitle | null>("get_live_subtitle");
    console.debug(`${LIVE_LOG_PREFIX} getLiveSubtitle result`, { hasSubtitle: !!subtitle, sessionId: subtitle?.sessionId, revision: subtitle?.revision });
    return subtitle;
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} getLiveSubtitle failed`, { error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function updateLiveOverlayLayout(
  sessionId: string,
  sizing: LiveOverlaySizing,
  contentSize: LiveOverlayContentSize,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${LIVE_LOG_PREFIX} updateLiveOverlayLayout start`, { sessionId, sizing, contentSize });
  try {
    await invokeFn<void>("update_live_overlay_layout", { sessionId, sizing, contentSize });
    console.debug(`${LIVE_LOG_PREFIX} updateLiveOverlayLayout success`, { sessionId });
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} updateLiveOverlayLayout failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}
export async function beginLiveOverlayDrag(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${LIVE_LOG_PREFIX} beginLiveOverlayDrag start`, { sessionId });
  try {
    await invokeFn<void>("begin_live_overlay_drag", { sessionId });
    console.debug(`${LIVE_LOG_PREFIX} beginLiveOverlayDrag success`, { sessionId });
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} beginLiveOverlayDrag failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}
export async function beginLiveOverlayResize(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${LIVE_LOG_PREFIX} beginLiveOverlayResize start`, { sessionId });
  try {
    await invokeFn<void>("begin_live_overlay_resize", { sessionId });
    console.debug(`${LIVE_LOG_PREFIX} beginLiveOverlayResize success`, { sessionId });
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} beginLiveOverlayResize failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export async function finishLiveOverlayResize(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${LIVE_LOG_PREFIX} finishLiveOverlayResize start`, { sessionId });
  try {
    await invokeFn<void>("finish_live_overlay_resize", { sessionId });
    console.debug(`${LIVE_LOG_PREFIX} finishLiveOverlayResize success`, { sessionId });
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} finishLiveOverlayResize failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}


export async function updateLiveOverlayPosition(
  sessionId: string,
  position: LiveOverlayPosition,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  console.info(`${LIVE_LOG_PREFIX} updateLiveOverlayPosition start`, { sessionId, position });
  try {
    await invokeFn<void>("update_live_overlay_position", { sessionId, position });
    console.debug(`${LIVE_LOG_PREFIX} updateLiveOverlayPosition success`, { sessionId });
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} updateLiveOverlayPosition failed`, { sessionId, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export function listenLiveStatus(
  handler: (status: LiveSessionStatus) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  console.info(`${LIVE_LOG_PREFIX} listenLiveStatus registering`);
  return listenFn<LiveSessionStatus>(LIVE_STATUS_EVENT, (event) => {
    console.info(`${LIVE_LOG_PREFIX} live-status event`, { state: event.payload.state, sessionId: event.payload.sessionId, hasError: !!(event.payload as any).error });
    console.debug(`${LIVE_LOG_PREFIX} live-status payload`, event.payload as unknown as Record<string, unknown>);
    handler(event.payload);
  });
}

export function listenLiveSubtitle(
  handler: (subtitle: LiveSubtitle) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  console.info(`${LIVE_LOG_PREFIX} listenLiveSubtitle registering`);
  return listenFn<LiveSubtitle>(LIVE_SUBTITLE_EVENT, (event) => {
    console.info(`${LIVE_LOG_PREFIX} live-subtitle event`, { sessionId: event.payload.sessionId, revision: event.payload.revision, isStreaming: event.payload.isStreaming, regions: event.payload.regions.length });
    console.debug(`${LIVE_LOG_PREFIX} live-subtitle payload`, { translatedTextLength: event.payload.translatedText.length, sourceTextLength: event.payload.sourceText.length });
    handler(event.payload);
  });
}
export async function setLiveRegionBoxesVisible(
  visible: boolean,
  emitFn: EmitFn = emit,
): Promise<void> {
  console.info(`${LIVE_LOG_PREFIX} setLiveRegionBoxesVisible start`, { visible });
  try {
    await emitFn<boolean>(LIVE_REGION_BOX_VISIBILITY_EVENT, visible);
    console.debug(`${LIVE_LOG_PREFIX} setLiveRegionBoxesVisible success`, { visible });
  } catch (error) {
    console.error(`${LIVE_LOG_PREFIX} setLiveRegionBoxesVisible failed`, { visible, error: error instanceof Error ? error.message : String(error) });
    throw error;
  }
}

export function listenLiveRegionBoxesVisible(
  handler: (visible: boolean) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  console.info(`${LIVE_LOG_PREFIX} listenLiveRegionBoxesVisible registering`);
  return listenFn<boolean>(LIVE_REGION_BOX_VISIBILITY_EVENT, (event) => {
    console.info(`${LIVE_LOG_PREFIX} live-region-box-visibility event`, { visible: event.payload });
    handler(event.payload);
  });
}


export function listenLiveDebugRecord(
  handler: (record: LiveDebugRecord) => void,
  listenFn: ListenFn = listen,
): Promise<UnlistenFn> {
  console.info(`${LIVE_LOG_PREFIX} listenLiveDebugRecord registering`);
  return listenFn<LiveDebugRecord>(LIVE_DEBUG_RECORD_EVENT, (event) => {
    console.info(`${LIVE_LOG_PREFIX} live-debug-record event`, { stage: event.payload.stage, outcome: event.payload.outcome, sessionId: event.payload.sessionId });
    console.debug(`${LIVE_LOG_PREFIX} live-debug-record payload`, event.payload as unknown as Record<string, unknown>);
    handler(event.payload);
  });
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
export function hasRenderableLiveSubtitleContent(subtitle: LiveSubtitle): boolean {
  return (
    subtitle.isStreaming ||
    subtitle.regions.length > 0 ||
    subtitle.translatedText.trim().length > 0 ||
    subtitle.sourceText.trim().length > 0
  );
}
export function mapLiveSubtitleRegionToOverlay(
  region: LiveSubtitleRegion,
  roi: LiveRoi,
  overlayWidth: number,
  overlayHeight: number,
): { x: number; y: number; width: number; height: number } | undefined {
  if (
    roi.clientWidth <= 0 ||
    roi.clientHeight <= 0 ||
    !Number.isFinite(overlayWidth) ||
    !Number.isFinite(overlayHeight) ||
    overlayWidth <= 0 ||
    overlayHeight <= 0 ||
    !Number.isFinite(region.bounds.left) ||
    !Number.isFinite(region.bounds.top) ||
    !Number.isFinite(region.bounds.width) ||
    !Number.isFinite(region.bounds.height) ||
    region.bounds.width <= 0 ||
    region.bounds.height <= 0
  ) {
    return undefined;
  }
  return {
    x: (region.bounds.left / roi.clientWidth) * overlayWidth,
    y: (region.bounds.top / roi.clientHeight) * overlayHeight,
    width: (region.bounds.width / roi.clientWidth) * overlayWidth,
    height: (region.bounds.height / roi.clientHeight) * overlayHeight,
  };
}
