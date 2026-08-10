import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { EventCallback, UnlistenFn } from "@tauri-apps/api/event";

export const LIVE_STATUS_EVENT = "live-status";
export const LIVE_SUBTITLE_EVENT = "live-subtitle";
export const LIVE_DEBUG_RECORD_EVENT = "live-debug-record";

export interface CaptureWindowInfo {
  id: string;
  title: string;
  processName: string;
  processId: number;
  width: number;
  height: number;
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
  ocrRuns: number;
  translationRuns: number;
  cacheHits: number;
  subtitlePublishes: number;
  lastOcrMs: number;
  lastTranslationMs: number;
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
}

export type LiveRecognitionMode = "automatic" | "key_trigger";

export type LiveRecognitionTrigger = "press" | "release";

export interface LiveRecognitionSettings {
  mode: LiveRecognitionMode;
  triggerKey: string;
  triggerEvent: LiveRecognitionTrigger;
  stabilityWaitMs: number;
}

export interface LiveSubtitleRegion {
  quad: [[number, number], [number, number], [number, number], [number, number]];
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
  observedAtEpochMs: number;
}

export type LiveDebugStage = "ocr" | "translation";

export type LiveDebugOutcome =
  | "awaiting_confirmation"
  | "confirmed"
  | "cache_hit"
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
  cacheHit: boolean;
  message?: string;
  observedAtEpochMs: number;
}

type InvokeFn = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;
type ListenFn = <T>(event: string, handler: EventCallback<T>) => Promise<UnlistenFn>;

export function listCaptureWindows(invokeFn: InvokeFn = invoke): Promise<CaptureWindowInfo[]> {
  return invokeFn<CaptureWindowInfo[]>("list_capture_windows");
}

export function beginLiveSelection(
  targetId: string,
  targetLanguage: string,
  overlaySettings: LiveOverlaySettings,
  recognitionSettings: LiveRecognitionSettings,
  invokeFn: InvokeFn = invoke,
): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("begin_live_selection", {
    targetId,
    targetLanguage,
    overlaySettings,
    recognitionSettings,
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

export function getLiveSessionStatus(invokeFn: InvokeFn = invoke): Promise<LiveSessionStatus> {
  return invokeFn<LiveSessionStatus>("get_live_session_status");
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
export function resolveLiveSubtitleRegionStyle(
  region: LiveSubtitleRegion,
  roi: LiveRoi,
): Record<string, string> {
  if (
    roi.clientWidth <= 0 ||
    roi.clientHeight <= 0 ||
    region.quad.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))
  ) {
    return {};
  }
  const horizontal = region.quad.map(([x]) => x);
  const vertical = region.quad.map(([, y]) => y);
  const clamp = (value: number, maximum: number): number =>
    Math.max(0, Math.min(maximum, value));
  const left = clamp(roi.x + Math.min(...horizontal), roi.clientWidth);
  const top = clamp(roi.y + Math.min(...vertical), roi.clientHeight);
  const right = clamp(roi.x + Math.max(...horizontal), roi.clientWidth);
  const bottom = clamp(roi.y + Math.max(...vertical), roi.clientHeight);
  const height = Math.max(1, ((bottom - top) / roi.clientHeight) * 100);
  return {
    left: `${(left / roi.clientWidth) * 100}%`,
    top: `${(top / roi.clientHeight) * 100}%`,
    width: `${Math.max(1, ((right - left) / roi.clientWidth) * 100)}%`,
    height: `${height}%`,
  };
}
