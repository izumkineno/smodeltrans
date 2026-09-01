import { invoke } from "@tauri-apps/api/core";

const isTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

function stringifyArgs(args: unknown[]): string {
  return args
    .map((arg) => {
      if (typeof arg === "string") return arg;
      if (arg instanceof Error) return arg.stack || arg.message;
      try {
        return JSON.stringify(arg);
      } catch {
        return String(arg);
      }
    })
    .join(" ");
}

async function writeToBridge(level: string, args: unknown[]) {
  if (!isTauri) return;
  const message = stringifyArgs(args);
  if (!message.trim()) return;
  try {
    await invoke("frontend_log", { level, message });
  } catch {
    // 忽略桥接失败，避免递归
  }
}

export const appLog = {
  trace: (...args: unknown[]) => void writeToBridge("trace", args),
  debug: (...args: unknown[]) => void writeToBridge("debug", args),
  info: (...args: unknown[]) => void writeToBridge("info", args),
  warn: (...args: unknown[]) => void writeToBridge("warn", args),
  error: (...args: unknown[]) => void writeToBridge("error", args),
};

let consoleForwardInstalled = false;

export function installConsoleForwarding() {
  if (consoleForwardInstalled || typeof window === "undefined" || typeof console === "undefined") return;
  consoleForwardInstalled = true;

  const wrap = (level: string, original: (...args: unknown[]) => void) => {
    return (...args: unknown[]) => {
      original(...args);
      void writeToBridge(level, args);
    };
  };

  // 保留原始引用，避免递归
  const origLog = console.log.bind(console);
  const origInfo = console.info.bind(console);
  const origWarn = console.warn.bind(console);
  const origError = console.error.bind(console);
  const origDebug = (console.debug ?? console.log).bind(console);

  console.log = wrap("info", origLog);
  console.info = wrap("info", origInfo);
  console.warn = wrap("warn", origWarn);
  console.error = wrap("error", origError);
  console.debug = wrap("debug", origDebug);

  // 全局错误兜底
  window.addEventListener("error", (event) => {
    void writeToBridge("error", [event.message, event.filename + ":" + event.lineno, event.error]);
  });
  window.addEventListener("unhandledrejection", (event) => {
    void writeToBridge("error", ["unhandledrejection", event.reason]);
  });

  void writeToBridge("info", [`  console forwarding installed, tauri=${isTauri}`]);
}

export async function listLogFiles(): Promise<string[]> {
  if (!isTauri) return [];
  try {
    return await invoke<string[]>("list_log_files");
  } catch (e) {
    console.warn("[app-log] listLogFiles failed", e);
    return [];
  }
}

export async function readLogFile(fileName: string, lines?: number): Promise<string> {
  if (!isTauri) return "";
  return await invoke<string>("read_log_file", { fileName, lines: lines ?? null });
}

export async function openLogDirectory(): Promise<void> {
  if (!isTauri) return;
  await invoke("open_log_directory");
}
