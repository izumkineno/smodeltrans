#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision 图片-OCR-翻译接口冒烟脚本（对应 docs/OPENAI_COMPAT_API.md §4/§5）
用法:
  python scripts/test_vision_api.py                          # 合成 Mod Organizer 报错弹窗 502x203 并测 /health + /v1/chat/completions
  python scripts/test_vision_api.py --image D:/path/mod-error.png
  python scripts/test_vision_api.py --host 127.0.0.1 --port 11438 --target Chinese
  python scripts/test_vision_api.py --synth-only              # 仅生成合成图不发请求
"""
import argparse, base64, json, sys, urllib.request, urllib.error
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11438
SYNTH_PATH = Path("scripts/_synth_mod_error.png")
# 与 Image #1 一致的文本（截图 OCR 行）
LINES = [
    "failed to initialize plugin D:/LoreRim/plugins/phase0_runtime_probe.py:",
    "TypeError: moba.IPlugin.__init__() must be called when overriding",
    "__init__",
    "At:",
    "D:/LoreRim/plugins/phase0_runtime_probe.py(234): createPlugin",
]

def synth_mod_error(path: Path, w=502, h=203):
    """生成 502x203 近似原弹窗的 PNG（白底+标题栏+红X+文字），无 PIL 时回退 1x1 占位"""
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        # 回退：1x1 白 PNG 的 base64 占位（仍可测 decode_image/400 路径，但无真实 OCR）
        print("[warn] Pillow 未安装，生成 1x1 占位 PNG；安装后可得真实 OCR：pip install Pillow", file=sys.stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 1x1 白 PNG
        tiny = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")
        path.write_bytes(tiny)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # 标题栏
    d.rectangle([0, 0, w, 22], fill=(240, 240, 240), outline=(180,180,180))
    d.text((8, 4), "Mod Organizer", fill=(0,0,0))
    d.rectangle([w-18, 3, w-4, 17], outline=(120,120,120))
    d.text((w-14, 1), "×", fill=(0,0,0))
    # 左侧红 X 圆
    d.ellipse([12, 34, 38, 60], fill=(220, 50, 50), outline=(180,0,0))
    d.text((21, 38), "×", fill=(255,255,255))
    # 文本区（等宽近似；Pillow 默认位图字体）
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = 36
    x0 = 50
    for line in LINES:
        # 手动换行：按像素裁
        d.text((x0, y), line, fill=(0,0,0), font=font)
        y += 14
        if y > h - 20:
            break
    # 底部 OK 按钮
    d.rectangle([w-90, h-28, w-10, h-8], outline=(120,120,120), fill=(240,240,240))
    d.text((w-55, h-24), "OK", fill=(0,0,0), font=font)
    d.rectangle([0,0,w-1,h-1], outline=(160,160,160))
    img.save(path, "PNG")
    return path

def http_get(url, timeout=5):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body, None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), None
    except Exception as e:
        return None, "", str(e)

def http_post_json(url, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, body, dict(r.headers), None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace"), dict(e.headers) if e.headers else {}, None
    except Exception as e:
        return None, "", {}, str(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, default="", help="待测图片路径（缺省合成 Mod Organizer 502x203）")
    ap.add_argument("--host", type=str, default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--target", type=str, default="Chinese", help="model 后缀，对应 target_language")
    ap.add_argument("--synth-only", action="store_true")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    image_path = Path(args.image) if args.image else SYNTH_PATH

    if not image_path.exists() or args.image == "":
        p = synth_mod_error(image_path)
        print(f"[synth] 已生成 {p} ({p.stat().st_size} bytes, {502}x{203})")
        if args.synth_only:
            print(f"[synth-only] 合成完成：{p.resolve()}")
            return 0
        image_path = p
    else:
        print(f"[image] 使用 {image_path} ({image_path.stat().st_size} bytes)")

    # base64 编码
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    data_url = f"data:image/png;base64,{b64}"
    print(f"[encode] {len(raw)} bytes -> base64 {len(b64)} chars ({len(b64)/1024:.1f} KiB), dataURL {len(data_url)} chars")
    # 限流提示
    if len(b64) > 14_000_000:
        print("[warn] 超过 MAX_BASE64_CHARS=14M，将被 400", file=sys.stderr)
    if len(raw) > 10*1024*1024:
        print("[warn] 超过 MAX_ENCODED_BYTES=10MiB，将被 400", file=sys.stderr)

    # 1) health
    print(f"\n[1/2] GET {base}/health  /  {base}/v1/health")
    for path in ("/health", "/v1/health"):
        url = base + path
        status, body, err = http_get(url, timeout=5)
        if err:
            print(f"  {url} -> Failed to fetch: {err}")
        else:
            print(f"  {url} -> {status} {body[:600]}")
            try:
                j = json.loads(body)
                if j.get("status") == "ok":
                    print(f"    -> ok port={j.get('port')} model_loaded={j.get('model_loaded')} ocr_loaded(if any)")
            except Exception:
                pass
    # 若 health 不通，给排障
    status, body, err = http_get(base + "/health", timeout=3)
    if err or status != 200:
        print("\n[diag] health 未通，说明 smodeltrans 未启动 / enabled=false / 端口漂移（试 11439/11440）/ 防火墙。")
        print("      按 docs/OPENAI_COMPAT_API.md §5：bun run tauri dev 启动桌面端，设置→OpenAI 兼容 enabled=true 保存后重试。")
        print("      curl.exe -i http://127.0.0.1:11438/health  应 200；网页 bun run dev 无后端属正常。")

    # 2) chat completions Vision
    url = base + "/v1/chat/completions"
    payload = {
        "model": f"hy-mt2:{args.target}",
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}]
    }
    # 轻量日志
    print(f"\n[2/2] POST {url}")
    print(f"      model=hy-mt2:{args.target}  image_count=1  (~{len(b64)} b64)")
    status, body, headers, err = http_post_json(url, payload, timeout=90)
    if err:
        print(f"  -> Failed to fetch: {err}")
        print("     参考 §5：确认 smodeltrans 已启动且 health 200；勿用 http://127.0.0.1:11438/v1（该路径不存在）")
        return 2
    print(f"  -> HTTP {status}")
    # 打印关键头
    if headers:
        ct = headers.get("Content-Type") or headers.get("content-type") or ""
        if ct:
            print(f"     Content-Type: {ct}")
    # 解析
    try:
        j = json.loads(body)
    except Exception:
        print(body[:2000])
        return 0
    # 错误分支
    if status != 200:
        print(json.dumps(j, ensure_ascii=False, indent=2)[:3000])
        if status == 503:
            print("\n[hint] 503 service_unavailable = live 翻译占用引擎，关掉实时翻译重试。")
        if status == 400 and "too many images" in body:
            print("[hint] 单次 <=8 张。")
        if "remote url not supported" in body:
            print("[hint] image_url.url 须为 data:image/...;base64,... 或裸 base64。")
        return 0
    # 成功：兼容 Text 与 Parts 双形态
    try:
        choice = j["choices"][0]
        msg = choice["message"]
        content = msg["content"]
        if isinstance(content, str):
            print(f"\n[ok] content(string) len={len(content)}")
            print(content[:800])
            print("\n[note] 文本形态（服务端未识别为图片，检查 peel_data_url 前缀是否 data:image/png;base64,）")
        elif isinstance(content, list):
            text_part = next((p for p in content if p.get("type")=="text"), None)
            img_part = next((p for p in content if p.get("type")=="image_url"), None)
            text = (text_part or {}).get("text","") if text_part else ""
            img_url = ((img_part or {}).get("image_url") or {}).get("url","") if img_part else ""
            print(f"\n[ok] text({len(text)} chars):\n{text[:1000]}")
            if img_url.startswith("data:image/png;base64,"):
                b = img_url.split(",",1)[1]
                print(f"\n[ok] annotated image: data:image/png;base64, + {len(b)} b64 chars ({len(b)*3/4/1024:.1f} KiB decoded)")
                # 落盘便于肉眼核对
                out = Path("scripts/_vision_annotated.png")
                out.write_bytes(base64.b64decode(b))
                print(f"     已落盘 {out.resolve()}，直接打开可见红框+译文叠加")
            else:
                print(f"\n[warn] 未返回标注图或非 dataURL：{img_url[:200]}")
            usage = j.get("usage",{})
            if usage:
                print(f"\n[usage] prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}")
        else:
            print(f"[warn] 未知 content 类型：{type(content)} {str(content)[:500]}")
    except Exception as e:
        print(f"[parse error] {e}\n{json.dumps(j, ensure_ascii=False, indent=2)[:3000]}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
