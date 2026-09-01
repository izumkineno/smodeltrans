use serde::{Deserialize, Serialize};
use serde_json::Value;

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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
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

    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }
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
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
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
                content: content.to_owned(),
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
