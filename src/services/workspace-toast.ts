import type { MessageApi } from "naive-ui";

export type WorkspaceToastType = "success" | "error" | "warning" | "info";

export function showWorkspaceToast(
  toast: MessageApi,
  type: WorkspaceToastType,
  content: string,
): void {
  const message = content.trim();
  if (!message) {
    return;
  }

  switch (type) {
    case "success":
      toast.success(message);
      return;
    case "error":
      toast.error(message);
      return;
    case "warning":
      toast.warning(message);
      return;
    case "info":
      toast.info(message);
      return;
  }
}
