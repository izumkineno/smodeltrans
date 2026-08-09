import {
  createRouter,
  createWebHashHistory,
} from "vue-router";
import type { RouteRecordRaw } from "vue-router";

export const WORKSPACE_ROUTE_NAMES = [
  "translate",
  "ocr",
  "ocr-translate",
  "settings",
  "model-monitor",
] as const;

export type WorkspaceRouteName = (typeof WORKSPACE_ROUTE_NAMES)[number];

export function isWorkspaceRouteName(value: unknown): value is WorkspaceRouteName {
  return WORKSPACE_ROUTE_NAMES.some((routeName) => routeName === value);
}

const routes: RouteRecordRaw[] = [
  {
    path: "/",
    redirect: { name: "ocr-translate" },
  },
  {
    path: "/translate",
    name: "translate",
    component: () => import("./components/TextTranslationPage.vue"),
  },
  {
    path: "/ocr",
    name: "ocr",
    component: () => import("./components/OcrPage.vue"),
  },
  {
    path: "/ocr-translate",
    name: "ocr-translate",
    component: () => import("./components/OcrTranslationPage.vue"),
  },
  {
    path: "/settings",
    name: "settings",
    component: () => import("./components/SettingsPage.vue"),
  },
  {
    path: "/model-monitor",
    name: "model-monitor",
    component: () => import("./components/ModelMonitorPage.vue"),
  },
  {
    path: "/:pathMatch(.*)*",
    redirect: { name: "ocr-translate" },
  },
];

export const router = createRouter({
  history: createWebHashHistory(),
  routes,
});
