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

export function listCaptureWindows(invokeFn: InvokeFn = invoke): Promise<CaptureWindowInfo[]> {
  return invokeFn<CaptureWindowInfo[]>("list_capture_windows");
}

export function startLiveSession(
  targetId: string,
  targetLanguage: string,
  overlaySettings: LiveOverlaySettings,
  recognitionSettings: LiveRecognitionSettings,
  translationSettings: LiveTranslationSettings,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("start_live_session", {
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

export function getLiveSubtitle(invokeFn: InvokeFn = invoke): Promise<LiveSubtitle | null> {
  return invokeFn<LiveSubtitle | null>("get_live_subtitle");
}

export function updateLiveOverlayLayout(
  sessionId: string,
  sizing: LiveOverlaySizing,
  contentSize: LiveOverlayContentSize,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  return invokeFn<void>("update_live_overlay_layout", { sessionId, sizing, contentSize });
}
export function beginLiveOverlayDrag(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  return invokeFn<void>("begin_live_overlay_drag", { sessionId });
}
export function beginLiveOverlayResize(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  return invokeFn<void>("begin_live_overlay_resize", { sessionId });
}

export function finishLiveOverlayResize(
  sessionId: string,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  return invokeFn<void>("finish_live_overlay_resize", { sessionId });
}


export function updateLiveOverlayPosition(
  sessionId: string,
  position: LiveOverlayPosition,
  invokeFn: InvokeFn = invoke,
): Promise<void> {
  return invokeFn<void>("update_live_overlay_position", { sessionId, position });
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
