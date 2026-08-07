/// <reference lib="esnext.promise" />
export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 20_000_000;

export const SUPPORTED_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "webp", "gif", "bmp"] as const;

const SUPPORTED_IMAGE_MIME_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  "image/bmp",
]);

export function validateImageFile(file: File): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  const supportedByMime = SUPPORTED_IMAGE_MIME_TYPES.has(file.type);
  const supportedByExtension = SUPPORTED_IMAGE_EXTENSIONS.includes(
    extension as (typeof SUPPORTED_IMAGE_EXTENSIONS)[number],
  );

  if (!supportedByMime && !supportedByExtension) {
    return "请选择 PNG、JPG、WEBP、GIF 或 BMP 图片。";
  }

  if (file.size === 0) {
    return "图片为空，请选择其他文件。";
  }

  if (file.size > MAX_IMAGE_BYTES) {
    return "图片大小不能超过 10 MB。";
  }

  return null;
}

export function validateImagePreview(previewUrl: string): Promise<string | null> {
  const { promise, resolve } = Promise.withResolvers<string | null>();
  const image = new Image();

  image.onload = () => {
    const pixelCount = image.naturalWidth * image.naturalHeight;
    resolve(
      pixelCount > MAX_IMAGE_PIXELS
        ? "图片像素不能超过 20 MP。"
        : null,
    );
  };
  image.onerror = () => {
    resolve("图片无法解码，请选择其他文件。");
  };
  image.src = previewUrl;

  return promise;
}

export function createImagePreview(file: File): string {
  return URL.createObjectURL(file);
}

export function releaseImagePreview(previewUrl: string | null): void {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
}

export async function copyTranslationText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "true");
  textArea.style.position = "fixed";
  textArea.style.opacity = "0";
  document.body.appendChild(textArea);

  try {
    textArea.select();
    if (!document.execCommand("copy")) {
      throw new Error("Clipboard access is unavailable");
    }
  } finally {
    textArea.remove();
  }
}

export function saveTranslationText(text: string, sourceName: string): void {
  const baseName = sourceName.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "-");
  const fileName = `${baseName || "translation"}-preview.txt`;
  const blobUrl = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const anchor = document.createElement("a");

  try {
    anchor.href = blobUrl;
    anchor.download = fileName;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 0);
  }
}

export function saveImageDataUrl(dataUrl: string, sourceName: string): void {
  const match = /^data:(image\/(?:png|jpeg|webp|gif|bmp));base64,(.+)$/.exec(dataUrl);
  if (!match) {
    throw new Error("图片数据格式无效");
  }

  const binary = globalThis.atob(match[2]);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const extension = match[1].split("/")[1] === "jpeg" ? "jpg" : match[1].split("/")[1];
  const baseName = sourceName.replace(/\.[^/.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "-");
  const blobUrl = URL.createObjectURL(new Blob([bytes], { type: match[1] }));
  const anchor = document.createElement("a");

  try {
    anchor.href = blobUrl;
    anchor.download = `${baseName || "translation"}-annotated.${extension}`;
    anchor.rel = "noopener";
    document.body.appendChild(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(blobUrl), 0);
  }
}
