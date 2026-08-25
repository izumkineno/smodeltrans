#!/usr/bin/env node
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = resolve(__dirname, "..");

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf-8"));
}
function writeJson(path, data) {
  writeFileSync(path, JSON.stringify(data, null, 2) + "\n", "utf-8");
}

const pkgPath = resolve(root, "package.json");
const cargoPath = resolve(root, "src-tauri/Cargo.toml");
const tauriPath = resolve(root, "src-tauri/tauri.conf.json");

const pkg = readJson(pkgPath);
const version = pkg.version?.trim();
if (!version || !/^\d+\.\d+\.\d+/.test(version)) {
  console.error(`[sync-version] package.json version 非法: ${version}`);
  process.exit(1);
}

// 1. 同步 Cargo.toml 首个 [package] 下的 version
let cargo = readFileSync(cargoPath, "utf-8");
const cargoBefore = cargo;
cargo = cargo.replace(
  /^version\s*=\s*".*?"/m,
  `version = "${version}"`
);
if (cargo !== cargoBefore) {
  writeFileSync(cargoPath, cargo, "utf-8");
  console.log(`[sync-version] Cargo.toml -> ${version}`);
} else {
  console.log(`[sync-version] Cargo.toml 已是 ${version}`);
}

// 2. 同步 tauri.conf.json 的 version
const tauri = readJson(tauriPath);
if (tauri.version !== version) {
  tauri.version = version;
  writeJson(tauriPath, tauri);
  console.log(`[sync-version] tauri.conf.json -> ${version}`);
} else {
  console.log(`[sync-version] tauri.conf.json 已是 ${version}`);
}

// 3. 提示其余位置已改为动态读取，无需同步
console.log(`[sync-version] App.vue 已改为 import { version } from '../package.json' 动态读取`);
console.log(`[sync-version] model_download.rs 已改为 env!("CARGO_PKG_VERSION") 动态读取`);
console.log(`[sync-version] 完成，统一由 package.json 管理（单一来源）`);
