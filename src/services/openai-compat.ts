import { invoke } from "@tauri-apps/api/core";

export interface OpenAiStatus {
  enabled: boolean;
  host: string;
  port: number;
  boundPort: number | null;
  running: boolean;
  hasApiKey: boolean;
  message: string;
}

export interface UpdateOpenAiRequest {
  enabled: boolean;
  host: string;
  port: number;
  apiKey: string | null;
}

type InvokeFn = <T>(command: string, args?: Record<string, unknown>) => Promise<T>;

export async function getOpenAiStatus(invokeFn: InvokeFn = invoke): Promise<OpenAiStatus> {
  return invokeFn<OpenAiStatus>("get_openai_status");
}

export async function updateOpenAiConfig(
  request: UpdateOpenAiRequest,
  invokeFn: InvokeFn = invoke,
): Promise<OpenAiStatus> {
  return invokeFn<OpenAiStatus>("update_openai_config", { request });
}

export function buildBaseUrl(status: Pick<OpenAiStatus, "host" | "port" | "boundPort">): string {
  const port = status.boundPort ?? status.port;
  return `http://${status.host}:${port}`;
}

export function buildOpenAiBaseUrlForSdk(status: OpenAiStatus): string {
  return `${buildBaseUrl(status)}/v1`;
}
