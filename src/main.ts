import { createApp } from "vue";
import App from "./App.vue";
import { router } from "./app-router";
import { installConsoleForwarding, appLog } from "./services/app-log";

installConsoleForwarding();
appLog.info(" app boot", { href: typeof window !== "undefined" ? window.location.href : "unknown" });

createApp(App).use(router).mount("#app");
