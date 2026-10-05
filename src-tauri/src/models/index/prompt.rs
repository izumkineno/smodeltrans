//! Index-Translate instTrans prompt（仿写官方 `inference/llm/translate.py:trans_prompt()`）。
//!
//! 范式：“请将以下{src}翻译成{target}，并且严格遵循所有的约束要求。【源文】…【约束要求】…”

/// 语言代码 → 中文名（官方 `LANG_NAMES` 子集，按需扩展）。
pub(crate) fn lang_name(code: &str) -> &str {
    match code.to_lowercase().as_str() {
        "en" => "英语",
        "zh" => "中文",
        "ja" => "日语",
        "ko" => "韩语",
        "fr" => "法语",
        "de" => "德语",
        "es" => "西班牙语",
        "ru" => "俄语",
        _ => "英语",
    }
}

/// 构造官方 instTrans 训练/评测同构 prompt。
///
/// `source_lang == "auto"` 时省略源语言名；`hard`/`soft` 为约束行（已带编号前缀则原样保留）；
/// `glossary` 为 `术语->译法` 对；`json_hold` 为 True 时追加 JSON 格式保持后缀。
pub(crate) fn trans_prompt(
    text: &str,
    target_lang: &str,
    source_lang: &str,
    hard: &[String],
    soft: &[String],
    glossary: &[String],
    json_hold: bool,
) -> String {
    let source = if source_lang.eq_ignore_ascii_case("auto") {
        String::new()
    } else {
        format!("{}", lang_name(source_lang))
    };
    let target = lang_name(target_lang);
    let mut constraints = Vec::new();
    for line in hard {
        let line = line.trim();
        if !line.is_empty() {
            constraints.push(format!("【硬性要求】{line}"));
        }
    }
    if !glossary.is_empty() {
        constraints.push(format!(
            "【硬性要求】专名/术语对照：{}",
            glossary.join("，")
        ));
    }
    for line in soft {
        let line = line.trim();
        if !line.is_empty() {
            constraints.push(format!("【注意】{line}"));
        }
    }
    if json_hold {
        constraints.push("【硬性要求】保留 JSON 结构与所有 key，只翻译非空 value。".to_owned());
    }
    if constraints.is_empty() {
        return format!(
            "请将以下{source}文本翻译为{target}，直接输出翻译结果，不要进行任何解释。\n\n{text}"
        );
    }
    let numbered = constraints
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{}. {line}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "请将以下{source}文本翻译成{target}，并且严格遵循所有约束要求。\n\n【源文】\n{text}\n\n【约束要求】\n{numbered}\n\n只输出译文，不要有任何额外说明。"
    )
}

/// 去 `<think>...</think>` 兜底（官方 `strip_think` 等价物）。
pub(crate) fn strip_think(text: &str) -> String {
    match text.find("</think>") {
        Some(idx) => text[idx + "</think>".len()..].trim().to_string(),
        None => text.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_source_omits_name_and_numbers_hard() {
        let prompt = trans_prompt(
            "你好",
            "en",
            "auto",
            &["保留占位符".to_owned()],
            &[],
            &[],
            false,
        );
        assert!(prompt.contains("请将以下文本翻译成英语"), "{prompt}");
        assert!(prompt.contains("1. 【硬性要求】保留占位符"), "{prompt}");
        assert!(prompt.contains("【源文】\n你好"), "{prompt}");
    }

    #[test]
    fn glossary_and_json_suffix() {
        let prompt = trans_prompt(
            "{\"title\": \"用户协议\"}",
            "en",
            "zh",
            &[],
            &[],
            &["碳纤维->carbon fiber".to_owned()],
            true,
        );
        assert!(
            prompt.contains("专名/术语对照：碳纤维->carbon fiber"),
            "{prompt}"
        );
        assert!(prompt.contains("保留 JSON 结构与所有 key"), "{prompt}");
    }
    #[test]
    fn unconstrained_prompt_matches_local_model_card() {
        let prompt = trans_prompt("Hello", "zh", "auto", &[], &[], &[], false);
        assert_eq!(
            prompt,
            "请将以下文本翻译为中文，直接输出翻译结果，不要进行任何解释。\n\nHello"
        );
    }

    #[test]
    fn strip_think_block() {
        assert_eq!(strip_think("<think>xx</think>  Hello").as_str(), "Hello");
        assert_eq!(strip_think("直接输出").as_str(), "直接输出");
    }
}
