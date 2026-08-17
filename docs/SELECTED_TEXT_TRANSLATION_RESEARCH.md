# 快捷选词翻译方案调研

## 1. 目标与结论

目标：为 `smodeltrans` 增加全局快捷翻译能力，在用户用鼠标选中文字后，通过快捷键取得选区文本，复用现有本地文本翻译链路，并展示翻译结果。

推荐方案：

> 全局快捷键触发后，优先使用 Windows UI Automation 读取当前文本选区；目标应用不支持 UI Automation 时，再进入用户可配置的剪贴板兼容模式。

不建议默认向任意前台程序模拟 `Ctrl+C`。这种方案会修改剪贴板，可能受 Windows UIPI 限制，还可能在终端或命令行程序中触发中断操作。

## 2. 当前工程条件

当前工程是 Tauri 2 + Vue 3 + TypeScript + Rust 桌面应用，已经具备以下相关能力：

- Tauri 2 桌面端依赖：`package.json`
- `tauri-plugin-clipboard-manager`：`package.json`、`src-tauri/Cargo.toml`
- Windows 键盘输入和窗口消息 API：`src-tauri/Cargo.toml`
- 前端文本翻译适配器：`src/services/translation-provider.ts` 中的 `TauriTextTranslationProvider`
- Rust 文本翻译命令：`src-tauri/src/backend/commands.rs` 中的 `translate_text`

当前尚未具备：

- Tauri 全局快捷键插件及对应 capability 权限
- 文本剪贴板读取权限；当前 capability 只开放了图片读取
- Windows UI Automation 选区读取模块
- 快捷翻译结果浮窗

因此后续不需要重新实现翻译引擎，主要缺少的是选区获取、全局触发和结果展示链路：

```text
全局快捷键
  → 获取其他程序选中的文本
  → 调用现有 translate_text
  → 展示快捷翻译结果
```

## 3. 主流方案比较

| 方案 | 如何判断存在选区 | 兼容性 | 用户干扰 | 实现成本 | 建议 |
| --- | --- | ---: | ---: | ---: | --- |
| Windows UI Automation | 读取 `TextPattern.GetSelection()`，排除空范围 | 中高 | 低 | 中高 | 默认方案 |
| 用户主动复制 | 监听剪贴板变化，读取用户复制后的文本 | 高 | 低 | 低 | 安全兼容模式 |
| 程序模拟 `Ctrl+C` | 发送复制键，检查剪贴板序号是否变化 | 高 | 高 | 中 | 仅作为可选回退 |
| 网页 DOM Selection API | 调用 `window.getSelection()` | 只限当前网页或 WebView | 低 | 低 | 不能解决外部程序 |
| OCR | 识别屏幕像素或重新框选区域 | 覆盖非文本界面 | 中 | 已有基础 | 最后回退 |
| 全局鼠标 Hook | 只能识别拖动和松开，不能确认选中文字 | 低 | 中 | 中 | 不能单独使用 |

## 4. 方案一：Windows UI Automation

### 4.1 工作方式

快捷键触发时执行以下流程：

1. 在显示或聚焦本工程窗口之前，取得当前拥有输入焦点的 UI Automation 元素。
2. 请求该元素的 `TextPattern`。
3. 调用 `IUIAutomationTextPattern::GetSelection()`。
4. 遍历返回的文本范围。
5. 调用 `IUIAutomationTextRange::GetText()` 取得纯文本。
6. 如果至少存在一个非空范围，则认为用户选中了文字。

Windows 对返回结果有明确约定：

- 有选中文本：返回一个或多个文本范围。
- 只有输入光标、没有选区：返回退化的空范围。
- 控件不支持文本选区：返回 `NULL`。
- 支持多个不连续选区：返回多个文本范围。

### 4.2 优点

- 不修改剪贴板。
- 不模拟键盘输入。
- 不需要先激活本工程窗口。
- 可以区分“只有输入光标”和“真正选中了文本”。
- 可以进一步读取选区边界，用于将翻译浮窗定位到选区附近。
- 不会误向终端或命令行程序发送 `Ctrl+C`。

### 4.3 局限

该方案依赖目标程序正确暴露无障碍接口：

- 标准 Win32 控件、Office、主流浏览器和常规编辑器通常支持较好。
- 自绘控件、游戏、远程桌面、部分 PDF/Canvas 和关闭无障碍树的应用可能无法读取。
- Electron、Qt 等自定义控件的效果取决于目标应用自身的无障碍实现。
- 普通权限进程访问高权限进程时可能受到系统限制。

### 4.4 工程注意事项

Microsoft 建议访问整个桌面的 UI Automation 客户端在独立的 COM MTA 工作线程中执行。直接在 Tauri 主线程或 UI 线程执行跨进程 UI Automation 调用，可能造成明显卡顿甚至无响应。

当前工程已经依赖 `windows` crate，但尚未启用 UI Automation 和 COM 所需的对应 feature。后续实现时需要补充 Windows API feature，并将选区查询封装为独立的 Rust 服务。

## 5. 方案二：用户主动复制后翻译

典型交互：

```text
选择文字
→ 按 Ctrl+C
→ 按快捷翻译键
```

另一种常见交互是连续按两次 `Ctrl+C`，由翻译应用检测短时间内的重复复制并自动翻译。

### 5.1 判断逻辑

这种方式不直接读取当前选区，而是将以下条件视为存在可翻译文本：

- 剪贴板发生变化。
- 新内容包含非空纯文本。
- 内容变化发生在快捷操作的有效时间窗口内。

Windows 可使用以下机制追踪剪贴板变化：

- `AddClipboardFormatListener`
- `WM_CLIPBOARDUPDATE`
- `GetClipboardSequenceNumber`

### 5.2 优点

- 用户主动执行复制，不需要程序注入键盘。
- 对大多数支持复制文本的应用兼容性较好。
- 实现简单。
- 不需要 Windows Accessibility 权限。

### 5.3 缺点

- 不是严格的一键选词翻译。
- 如果缺少剪贴板变化校验，可能误用之前的旧文本。
- 连续两次 `Ctrl+C` 一般需要剪贴板监听或低级键盘监听，不能只依赖普通全局快捷键。
- 无法直接获得选区坐标。

### 5.4 适用定位

适合作为安全兼容模式，例如：

- 复制后按快捷键翻译。
- 连续复制两次后翻译。
- 复制文本时自动翻译。

如果第一版优先验证功能而不是追求严格的一键操作，这是实现风险最低的方案。

## 6. 方案三：程序模拟 Ctrl+C

典型流程：

```text
按全局翻译快捷键
→ 记录剪贴板序号
→ 向前台应用发送 Ctrl+C
→ 等待剪贴板序号变化
→ 读取文本
→ 开始翻译
```

### 6.1 优点

- 对支持复制命令但不支持 UI Automation 的程序兼容性较高。
- 用户只需要按一次翻译快捷键。
- 实现成本低于完整 UI Automation。

### 6.2 风险

#### Ctrl+C 不一定表示复制

在终端、控制台、SSH 会话或 REPL 中，`Ctrl+C` 可能表示中断当前任务，而不是复制文本。Microsoft 曾专门记录过字典类程序为了读取选区而向任意前台程序发送 `Ctrl+C`，最终终止用户长时间运行任务的案例。

#### 输入注入受 UIPI 限制

Windows `SendInput` 只能向相同或更低完整性级别的应用注入输入。普通权限运行的翻译应用可能无法向管理员权限应用发送复制键。

#### 会修改用户剪贴板

原剪贴板可能包含：

- 图片
- HTML
- RTF
- 文件列表
- 应用私有格式
- 延迟渲染对象

只保存和恢复纯文本不能无损恢复这些内容。当前 Tauri 剪贴板插件也不能简单完成所有剪贴板格式的完整快照和恢复。

#### 快捷键修饰键可能干扰

如果翻译快捷键自身包含 `Ctrl`，在按键仍处于按下状态时立即注入另一个 `Ctrl+C`，可能形成错误组合。兼容实现通常需要等待快捷键释放后再执行复制。

### 6.3 建议

该方案只应作为用户主动开启的“兼容取词”模式：

- 默认关闭。
- 对终端和控制台应用禁用。
- 设置较短超时。
- 必须检查剪贴板序号变化，不能直接读取旧剪贴板。
- 明确告知用户该模式会触发目标程序的复制命令。

## 7. 方案四：网页或应用内 Selection API

在自己的 WebView 或普通网页中，可以使用：

- `window.getSelection()`
- `Selection.toString()`
- `<input>` 和 `<textarea>` 的 `selectionStart`、`selectionEnd`

该方案只能访问当前 Tauri WebView 中的 DOM，不能读取 Chrome、Word、PDF 阅读器、VS Code 或其他桌面程序中的选区。

因此它适合工程内部文本框的快捷翻译，但不能解决全局选词翻译。

## 8. 方案五：OCR 回退

对于没有真正文本节点的界面，UI Automation 和复制都可能无法取得文字，例如：

- 游戏字幕
- 图片
- 视频
- Canvas
- 扫描版 PDF
- 远程桌面画面

本工程已经具备截图和 OCR 链路，可以在文本选区获取失败后提供手动 OCR 回退：

```text
快捷翻译
→ UI Automation 未取得文字
→ 剪贴板兼容模式失败或未启用
→ 提示用户框选区域进行 OCR
```

OCR 本身不能自动知道用户高亮了哪段文字，仍然需要重新框选、读取可用的选区边界，或者执行鼠标附近 OCR。因此它属于回退方案，而不是选区检测方案。

## 9. 全局鼠标 Hook 不能单独解决问题

监听鼠标按下、拖动和松开，只能证明用户执行过拖动操作，不能证明：

- 拖动对象是文本。
- 存在非空字符选区。
- 选区仍然有效。
- 目标应用允许读取这些文字。

用户可能是在拖动窗口、选择文件、拖动滑块、框选图片或操作游戏视角。

因此鼠标 Hook 最多只能作为触发器。鼠标松开后仍然需要调用 UI Automation、检查复制结果或者执行 OCR。

## 10. 本工程推荐接入路线

### 10.1 推荐默认架构

```text
Tauri 全局快捷键
        │
        ▼
记录当前前台应用，暂不聚焦翻译窗口
        │
        ▼
Windows UI Automation 工作线程
        │
        ├── 取得非空选区 ──→ 调用现有 translate_text
        │
        └── 不支持或无选区
                 │
                 ├── 普通模式：提示“未检测到选中文字”
                 └── 兼容模式：尝试剪贴板取词
```

### 10.2 推荐原因

- 默认路径不破坏剪贴板。
- 不会给终端误发 `Ctrl+C`。
- 当前 Rust 后端适合承载 Windows UI Automation。
- 现有 `translate_text` 可以直接复用。
- 后续可以利用 UI Automation 选区边界定位翻译浮窗。
- 对不支持 UI Automation 的应用仍然保留兼容方案。

### 10.3 最小风险版本

如果第一版只需要快速验证产品交互，可以先采用：

```text
用户先按 Ctrl+C
→ 按全局翻译快捷键
→ 读取剪贴板文本
→ 调用现有 translate_text
```

这个版本实现最快、风险最低，但不是严格的一键选词翻译。

### 10.4 不推荐的默认实现

```text
全局快捷键
→ 无条件向任意前台程序发送 Ctrl+C
```

这种方案看似兼容性较高，但终端中断、剪贴板破坏和 UIPI 限制都是实际风险。

## 11. 跨平台对应接口

| 系统 | 无障碍选区接口 |
| --- | --- |
| Windows | UI Automation `TextPattern.GetSelection()` |
| macOS | Accessibility API `kAXSelectedTextAttribute`，需要用户授权 Accessibility 权限 |
| Linux | AT-SPI `Text.get_selection()`，Wayland 和应用支持情况需要单独处理 |

Tauri 的全局快捷键和剪贴板插件支持主流桌面平台，但直接读取其他应用的选区仍然需要按操作系统分别实现。

## 12. 参考资料

- [Microsoft：IUIAutomationTextPattern::GetSelection](https://learn.microsoft.com/en-us/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomationtextpattern-getselection)
- [Microsoft：IUIAutomation::GetFocusedElement](https://learn.microsoft.com/en-us/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomation-getfocusedelement)
- [Microsoft：IUIAutomationTextRange::GetText](https://learn.microsoft.com/en-us/windows/win32/api/uiautomationclient/nf-uiautomationclient-iuiautomationtextrange-gettext)
- [Microsoft：UI Automation 线程要求](https://learn.microsoft.com/en-us/windows/win32/winauto/uiauto-threading)
- [Microsoft：SendInput 与 UIPI 限制](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput)
- [Microsoft：GetClipboardSequenceNumber](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getclipboardsequencenumber)
- [Microsoft：向任意程序发送 Ctrl+C 的风险](https://devblogs.microsoft.com/oldnewthing/20110623-00/?p=10353)
- [Tauri 2：Global Shortcut 插件](https://v2.tauri.app/plugin/global-shortcut/)
- [Tauri 2：Clipboard 插件](https://v2.tauri.app/plugin/clipboard/)
- [MDN：Window.getSelection()](https://developer.mozilla.org/en-US/docs/Web/API/Window/getSelection)
- [Apple：kAXSelectedTextAttribute](https://developer.apple.com/documentation/applicationservices/kaxselectedtextattribute)
- [GNOME：AT-SPI Text.get_selection](https://gnome.pages.gitlab.gnome.org/at-spi2-core/libatspi/method.Text.get_selection.html)
