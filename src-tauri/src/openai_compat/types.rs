use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// OpenAI extra body, we use `target_language` if present
    #[serde(default)]
    pub target_language: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    // catch-all for extra_body fields
    #[allow(dead_code)]
    #[serde(flatten)]
    pub extra: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ImageUrl {
    pub url: String,
    /// 可选透传请求头，覆盖 Referer/User-Agent/Authorization/Cookie 等，防 403/热链
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<HashMap<String, String>>,
    /// 快捷 Referer，优先级：headers.Referer > referer
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub referer: Option<String>,
    /// 快捷 User-Agent
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// OpenAI 兼容 detail 字段，忽略但保留以免反序列化失败
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ChatCompletionRequest {
    pub fn source_text(&self) -> String {
        self.messages
            .last()
            .map(|m| m.content.as_text())
            .unwrap_or_default()
            .trim()
            .to_owned()
    }

    pub fn target_language(&self, default: &str) -> String {
        if let Some(lang) = &self.target_language {
            if !lang.trim().is_empty() {
                return lang.trim().to_owned();
            }
        }
        if let Some(lang) = &self.language {
            if !lang.trim().is_empty() {
                return lang.trim().to_owned();
            }
        }
        // support model suffix "hy-mt2-1.8b:Chinese"
        if let Some((_, suffix)) = self.model.split_once(':') {
            let s = suffix.trim();
            if !s.is_empty() {
                return s.to_owned();
            }
        }
        // try parse "Translate to <lang>:" prefix in last message
        let text = self.source_text();
        if let Some(lang) = parse_translate_prefix(&text) {
            return lang;
        }
        default.to_owned()
    }

    pub fn plain_source_text(&self) -> String {
        let text = self.source_text();
        // strip "Translate to X:" prefix if present
        if let Some(idx) = text.to_lowercase().find("translate to") {
            if let Some(colon) = text[idx..].find(':') {
                let after = text[idx + colon + 1..].trim();
                if !after.is_empty() {
                    return after.to_owned();
                }
            }
        }
        // also strip "翻译成X：" etc? keep simple
        text
    }

    pub fn supplemental_prompt(&self) -> String {
        // 复用 Hy-MT2 官方附加约束：收集 system/developer 消息作为 supplemental_prompt，透传至 render_single_prompt::Additional requirements
        let parts: Vec<String> = self
            .messages
            .iter()
            .filter(|m| {
                let role = m.role.trim().to_lowercase();
                role == "system" || role == "developer"
            })
            .map(|m| m.content.as_text().trim().to_owned())
            .filter(|s| !s.is_empty())
            .collect();
        parts.join("\n\n")
    }

    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    #[allow(dead_code)]
    pub fn image_urls(&self) -> Vec<String> {
        let mut out = Vec::new();
        for msg in &self.messages {
            if let MessageContent::Parts(parts) = &msg.content {
                for p in parts {
                    if let ContentPart::ImageUrl { image_url } = p {
                        out.push(image_url.url.clone());
                    }
                }
            }
        }
        out
    }

    /// 保留 url+认证头的完整输入，供服务端拉取时透传 Referer/UA/Authorization 等
    pub fn image_inputs(&self) -> Vec<ImageUrl> {
        let mut out = Vec::new();
        for msg in &self.messages {
            if let MessageContent::Parts(parts) = &msg.content {
                for p in parts {
                    if let ContentPart::ImageUrl { image_url } = p {
                        out.push(image_url.clone());
                    }
                }
            }
        }
        out
    }
    #[allow(dead_code)]
    pub fn has_image(&self) -> bool {
        !self.image_urls().is_empty()
    }

    #[allow(dead_code)]
    pub fn image_count(&self) -> usize {
        self.image_urls().len()
    }
}

/// 是否为服务端可拉取的 http/https 远端 URL（用于跳过前端 CORS + 防 403）。
/// 仅 `http://` / `https://` 视为可 fetch，其它 `://` / `:` 仍走 `peel_data_url` 的 400。
pub fn is_http_url(url: &str) -> bool {
    let t = url.trim();
    if t.len() >= 7 && t[..7].eq_ignore_ascii_case("http://") {
        return true;
    }
    if t.len() >= 8 && t[..8].eq_ignore_ascii_case("https://") {
        return true;
    }
    false
}

pub fn peel_data_url(url: &str) -> Result<String, String> {
    let t = url.trim();
    if t.is_empty() {
        return Err("empty url".into());
    }
    if t.len() >= 5 && t[..5].eq_ignore_ascii_case("data:") {
        let comma = t.find(',').ok_or_else(|| "malformed data URL".to_string())?;
        let meta = &t[..comma];
        if !meta.to_ascii_lowercase().contains(";base64") {
            return Err("data URL missing ;base64".into());
        }
        return Ok(t[comma + 1..].to_owned());
    }
    if t.contains("://") || t.contains(':') {
        return Err("remote url not supported".into());
    }
    Ok(t.to_owned())
}

fn parse_translate_prefix(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    if let Some(start) = lower.find("translate to") {
        let rest = &text[start + "translate to".len()..];
        let rest = rest.trim_start();
        // take until ':' or '\n' or 32 chars
        let end = rest
            .find(':')
            .or_else(|| rest.find('\n'))
            .unwrap_or_else(|| rest.len().min(32));
        let lang = rest[..end].trim().trim_matches(|c| c == ' ' || c == '"' || c == '\'');
        if !lang.is_empty() && lang.len() <= 32 {
            return Some(lang.to_owned());
        }
    }
    None
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum MessageContentOut {
    Text(String),
    Parts(Vec<ContentPartOut>),
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentPartOut {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlOut },
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageUrlOut {
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: MessageContentOut,
}

impl ChatMessageOut {
    pub fn with_image(text: String, image_data_url: Option<String>) -> Self {
        let content = match image_data_url {
            Some(url) if !url.trim().is_empty() => MessageContentOut::Parts(vec![
                ContentPartOut::Text { text: text.clone() },
                ContentPartOut::ImageUrl {
                    image_url: ImageUrlOut { url },
                },
            ]),
            _ => MessageContentOut::Text(text),
        };
        Self {
            role: "assistant".to_owned(),
            content,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub port: Option<u16>,
    pub model_loaded: bool,
    pub owned_by: String,
}

pub fn new_chat_response(model: &str, content: &str, prompt_tokens: usize, completion_tokens: usize) -> ChatCompletionResponse {
    let id = format!("chatcmpl-{}", &uuid_simple());
    ChatCompletionResponse {
        id,
        object: "chat.completion".to_owned(),
        created: now_secs(),
        model: model.to_owned(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant".to_owned(),
                content: MessageContentOut::Text(content.to_owned()),
            },
            finish_reason: "stop".to_owned(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

pub fn new_chat_response_with_image(
    model: &str,
    text: String,
    image_opt: Option<String>,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> ChatCompletionResponse {
    let id = format!("chatcmpl-{}", &uuid_simple());
    ChatCompletionResponse {
        id,
        object: "chat.completion".to_owned(),
        created: now_secs(),
        model: model.to_owned(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessageOut::with_image(text, image_opt),
            finish_reason: "stop".to_owned(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn uuid_simple() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn peel_data_url_png() {
        let url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==";
        assert_eq!(peel_data_url(url).unwrap(), "iVBORw0KGgoAAAANSUhEUg==");
    }

    #[test]
    fn peel_data_url_jpeg() {
        let url = "data:image/jpeg;base64,/9j/4AAQSkZJRg==";
        assert_eq!(peel_data_url(url).unwrap(), "/9j/4AAQSkZJRg==");
    }

    #[test]
    fn peel_data_url_jpeg_uppercase() {
        let url = "DATA:IMAGE/JPEG;BASE64,/9j/4AAQSkZJRg==";
        assert_eq!(peel_data_url(url).unwrap(), "/9j/4AAQSkZJRg==");
    }

    #[test]
    fn peel_data_url_webp() {
        let url = "data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoB";
        assert_eq!(
            peel_data_url(url).unwrap(),
            "UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoB"
        );
    }

    #[test]
    fn peel_data_url_charset() {
        let url = "data:image/png;charset=utf-8;base64,iVBORw0KGgoAAAANSUhEUg==";
        assert_eq!(peel_data_url(url).unwrap(), "iVBORw0KGgoAAAANSUhEUg==");
    }

    #[test]
    fn peel_data_url_charset_webp() {
        let url = "data:image/webp;charset=utf-8;base64,UklGRxxx";
        assert_eq!(peel_data_url(url).unwrap(), "UklGRxxx");
    }

    #[test]
    fn peel_data_url_bare() {
        let url = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=";
        assert_eq!(peel_data_url(url).unwrap(), url);
    }

    #[test]
    fn peel_data_url_https_reject() {
        let url = "https://example.com/image.png";
        assert_eq!(
            peel_data_url(url).unwrap_err(),
            "remote url not supported"
        );
    }

    #[test]
    fn peel_data_url_http_reject() {
        let url = "http://example.com/x.jpg";
        assert!(peel_data_url(url).is_err());
        assert_eq!(peel_data_url(url).unwrap_err(), "remote url not supported");
    }

    #[test]
    fn peel_data_url_blob_reject() {
        let url = "blob:https://example.com/uuid";
        assert_eq!(
            peel_data_url(url).unwrap_err(),
            "remote url not supported"
        );
    }

    #[test]
    fn peel_data_url_data_missing_base64() {
        let url = "data:image/png,hello";
        assert_eq!(
            peel_data_url(url).unwrap_err(),
            "data URL missing ;base64"
        );
    }

    #[test]
    fn peel_data_url_malformed_no_comma() {
        let url = "data:image/png;base64";
        assert_eq!(peel_data_url(url).unwrap_err(), "malformed data URL");
    }

    #[test]
    fn peel_data_url_empty() {
        assert_eq!(peel_data_url("").unwrap_err(), "empty url");
        assert_eq!(peel_data_url("   ").unwrap_err(), "empty url");
    }

    #[test]
    fn image_urls_extract() {
        let req = ChatCompletionRequest {
            model: "hy-mt2:Chinese".into(),
            messages: vec![
                ChatMessage {
                    role: "user".into(),
                    content: MessageContent::Parts(vec![
                        ContentPart::Text {
                            text: "hello".into(),
                        },
                        ContentPart::ImageUrl {
                            image_url: ImageUrl {
                                url: "data:image/png;base64,AAA".into(),
                                ..Default::default()
                            },
                        },
                        ContentPart::ImageUrl {
                            image_url: ImageUrl {
                                url: "data:image/jpeg;base64,BBB".into(),
                                ..Default::default()
                            },
                        },
                    ]),
                },
                ChatMessage {
                    role: "user".into(),
                    content: MessageContent::Text("ignore".into()),
                },
            ],
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            target_language: None,
            language: None,
            extra: None,
        };
        assert_eq!(
            req.image_urls(),
            vec![
                "data:image/png;base64,AAA".to_string(),
                "data:image/jpeg;base64,BBB".to_string()
            ]
        );
        assert!(req.has_image());
        assert_eq!(req.image_count(), 2);
    }

    #[test]
    fn image_urls_empty_when_text_only() {
        let req = ChatCompletionRequest {
            model: "hy-mt2".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: MessageContent::Text("hello".into()),
            }],
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            target_language: None,
            language: None,
            extra: None,
        };
        assert!(req.image_urls().is_empty());
        assert!(!req.has_image());
        assert_eq!(req.image_count(), 0);
    }

    #[test]
    fn message_content_out_text_serializes_as_string() {
        let msg = ChatMessageOut {
            role: "assistant".into(),
            content: MessageContentOut::Text("hi".into()),
        };
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["content"], json!("hi"));
    }

    #[test]
    fn message_content_out_parts_serializes_as_array() {
        let msg = ChatMessageOut::with_image(
            "translated".into(),
            Some("data:image/png;base64,AAA".into()),
        );
        let v = serde_json::to_value(&msg).unwrap();
        assert!(v["content"].is_array());
        assert_eq!(v["content"][0]["type"], json!("text"));
        assert_eq!(v["content"][0]["text"], json!("translated"));
        assert_eq!(v["content"][1]["type"], json!("image_url"));
        assert_eq!(
            v["content"][1]["image_url"]["url"],
            json!("data:image/png;base64,AAA")
        );
    }

    #[test]
    fn new_chat_response_still_text() {
        let resp = new_chat_response("hy-mt2", "hello", 10, 20);
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], json!("hello"));
    }

    #[test]
    fn new_chat_response_with_image_parts() {
        let resp = new_chat_response_with_image(
            "hy-mt2",
            "hi".into(),
            Some("data:image/png;base64,AAA".into()),
            10,
            20,
        );
        let v = serde_json::to_value(&resp).unwrap();
        assert!(v["choices"][0]["message"]["content"].is_array());
        assert_eq!(
            v["choices"][0]["message"]["content"][1]["image_url"]["url"],
            json!("data:image/png;base64,AAA")
        );
    }

    #[test]
    fn new_chat_response_with_image_none_is_text() {
        let resp = new_chat_response_with_image("hy-mt2", "hi".into(), None, 10, 20);
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], json!("hi"));
    }

    #[test]
    fn with_image_empty_string_is_text() {
        let msg = ChatMessageOut::with_image("hi".into(), Some("   ".into()));
        let v = serde_json::to_value(&msg).unwrap();
        assert_eq!(v["content"], json!("hi"));
    }

    #[test]
    fn is_http_url_true() {
        assert!(is_http_url("http://example.com/a.png"));
        assert!(is_http_url("https://example.com/a.jpg"));
        assert!(is_http_url("HTTPS://example.com/x.webp"));
        assert!(is_http_url("  https://example.com/img.png  "));
    }

    #[test]
    fn is_http_url_false() {
        assert!(!is_http_url("data:image/png;base64,AAA"));
        assert!(!is_http_url("iVBORw0KGgoAAAANSUhEUg=="));
        assert!(!is_http_url("blob:https://example.com/uuid"));
        assert!(!is_http_url("ftp://example.com/img.png"));
        assert!(!is_http_url(""));
    }

    #[test]
    fn image_url_with_headers_deser() {
        let j = json!({"url":"https://example.com/img.jpg","headers":{"Authorization":"Bearer tok","Referer":"https://example.com/"},"referer":"https://ref.example.com/","user_agent":"MyAgent/1.0"});
        let iu: ImageUrl = serde_json::from_value(j).unwrap();
        assert_eq!(iu.url, "https://example.com/img.jpg");
        assert_eq!(iu.headers.as_ref().unwrap().get("Authorization").unwrap(), "Bearer tok");
        assert_eq!(iu.referer.as_deref().unwrap(), "https://ref.example.com/");
        assert_eq!(iu.user_agent.as_deref().unwrap(), "MyAgent/1.0");
    }

    #[test]
    fn image_inputs_with_auth() {
        let req = ChatCompletionRequest {
            model: "hy-mt2:Chinese".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/secure.png".into(),
                        headers: Some([("Authorization".into(), "Bearer abc".into())].into_iter().collect()),
                        referer: Some("https://example.com/".into()),
                        ..Default::default()
                    },
                }]),
            }],
            stream: None,
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            seed: None,
            target_language: None,
            language: None,
            extra: None,
        };
        let inputs = req.image_inputs();
        assert_eq!(inputs.len(), 1);
        assert!(is_http_url(&inputs[0].url));
        assert_eq!(inputs[0].headers.as_ref().unwrap().get("Authorization").unwrap(), "Bearer abc");
        assert_eq!(inputs[0].referer.as_deref().unwrap(), "https://example.com/");
    }
}
