import { afterEach, describe, expect, test } from "bun:test";
import { createTauriTranslationProvider, isTranslationCancellation } from "./translation-provider";

const originalWindow = globalThis.window;

function installTauriWindow() {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: {
      __TAURI_INTERNALS__: {},
      btoa: globalThis.btoa,
    },
  });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

afterEach(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: originalWindow,
  });
});

describe("TauriTranslationProvider", () => {
  test("sends real base64 payload and preserves response fields", async () => {
    installTauriWindow();
    const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
    const provider = createTauriTranslationProvider(async <T>(command, args): Promise<T> => {
      calls.push({ command, args });
      return {
        text: "translated",
        markdown: "# translated",
        annotatedImageDataUrl: "data:image/png;base64,AAAA",
        providerLabel: "PP-OCRv5 + Hy-MT2 / Candle",
        isTranslated: true,
        durationMs: 1234,
      } as T;
    });

    const result = await provider.translate(
      {
        file: new File([new Uint8Array([1, 2, 3])], "sample.png", { type: "image/png" }),
        targetLanguage: "Chinese",
      },
      new AbortController().signal,
    );

    expect(result.text).toBe("translated");
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
    expect(calls).toHaveLength(1);
    expect(calls[0].command).toBe("translate_image");
    const request = calls[0].args?.request as Record<string, unknown>;
    expect(request.imageBase64).toBe("AQID");
    expect(request.fileName).toBe("sample.png");
    expect(request.targetLanguage).toBe("Chinese");
    expect(typeof request.requestId).toBe("string");
  });

  test("abort sends additive backend cancellation and normalizes AbortError", async () => {
    installTauriWindow();
    const pending = deferred<never>();
    const calls: Array<{ command: string; args?: Record<string, unknown> }> = [];
    const provider = createTauriTranslationProvider(<T>(command: string, args?: Record<string, unknown>): Promise<T> => {
      calls.push({ command, args });
      return command === "translate_image" ? pending.promise : Promise.resolve(undefined as T);
    });
    const controller = new AbortController();
    const promise = provider.translate(
      {
        file: new File([new Uint8Array([1])], "sample.png", { type: "image/png" }),
        targetLanguage: "Chinese",
      },
      controller.signal,
    );

    await new Promise((resolve) => setTimeout(resolve, 0));
    controller.abort();
    const error = await promise.catch((value: unknown) => value);

    expect(isTranslationCancellation(error)).toBe(true);
    expect(calls.map((call) => call.command)).toEqual(["translate_image", "cancel_translation"]);
    const translateRequest = calls[0].args?.request as Record<string, unknown>;
    const cancelRequest = calls[1].args?.request as Record<string, unknown>;
    expect(cancelRequest.requestId).toBe(translateRequest.requestId);
  });

  test("backend cancelled code normalizes to AbortError", async () => {
    installTauriWindow();
    const provider = createTauriTranslationProvider(<T>(command: string): Promise<T> => {
      if (command === "translate_image") {
        return Promise.reject({ code: "cancelled", message: "cancelled" });
      }
      return Promise.resolve(undefined as T);
    });

    const error = await provider
      .translate(
        {
          file: new File([new Uint8Array([1])], "sample.png", { type: "image/png" }),
          targetLanguage: "Chinese",
        },
        new AbortController().signal,
      )
      .catch((value: unknown) => value);

    expect(isTranslationCancellation(error)).toBe(true);
  });
});
