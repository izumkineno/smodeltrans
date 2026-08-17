# Agent 执行前置要求

1. 所有 agent、子 agent 和自动化任务在执行任何操作前，必须先读取并遵守仓库根目录 `AGENT.md`；未读取本文件不得继续。
2. 本文件规则适用于分析、编辑、编译、运行、测试、提交及验证全过程。

## `candle-flash-attn` 强制禁止事项

1. 严禁重新编译 `candle-flash-attn`。不得执行或组合任何会导致其重新编译的命令，包括 `cargo build`、`cargo check`、`cargo test`、`cargo run` 以及清理后构建。必须复用现有构建产物；否则会造成超长编译时间和超大的编译空间占用。
2. 严禁新建工程、临时工程、独立目录或 worktree，用于 `candle-flash-attn` 相关的编译或运行。
3. 严禁修改任何可能使 `candle-flash-attn` 失效并触发重新编译的编译条件或构建输入，包括但不限于 Cargo features、依赖版本、patch、profile、target 配置、CUDA/编译器参数、环境变量、build script 和相关配置文件。
4. 仅允许只读检查现有配置与复用现有构建产物。若任务必须重新编译或调整上述编译条件，agent 必须停止相关操作并明确报告，不得自行执行。

# 提交信息规范

## 强制要求

1. 所有 Git 提交信息必须遵循 [Conventional Commits 1.0.0](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 规范。
2. 提交信息首选简体中文；`type`、`scope`、代码标识符、文件路径和协议关键字保留英文。
3. 一个提交只包含一个完整、独立的逻辑变更，不得混入无关修改。
4. 提交前必须确认变更范围，并根据实际内容选择准确的提交类型。

## 提交格式

```text
<type>(<scope>)!: <中文说明>

[可选的中文正文]

[可选的脚注]
```

- `scope` 可选，用于标明受影响的模块或功能。
- `!` 可选，仅用于不兼容变更。
- 说明应简洁、明确，使用祈使语气，不添加结尾句号，建议不超过 72 个字符。
- 不兼容变更必须在脚注中使用 `BREAKING CHANGE: <说明>`。

## 允许的提交类型

| 类型 | 用途 |
| --- | --- |
| `feat` | 新增功能 |
| `fix` | 修复缺陷 |
| `docs` | 修改文档 |
| `style` | 调整格式，不改变程序行为 |
| `refactor` | 重构代码，不新增功能或修复缺陷 |
| `perf` | 改善性能 |
| `test` | 新增或修改测试 |
| `build` | 修改构建系统或外部依赖 |
| `ci` | 修改 CI 配置或脚本 |
| `chore` | 其他维护性修改 |
| `revert` | 撤销已有提交 |

## 示例

```text
feat(posts): 添加文章分类筛选
fix(search): 修复中文搜索索引加载失败
docs: 更新私人博客使用指南
refactor(config): 简化站点语言配置
```

不兼容变更示例：

```text
feat(content)!: 调整文章 Frontmatter 结构

BREAKING CHANGE: category 字段改为必填字段
```
