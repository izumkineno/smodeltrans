use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OpenAiCompatConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub api_key: Option<String>,
}

fn default_host() -> String {
    tracing::trace!(target: "openai_compat::config", "default_host called");
    "127.0.0.1".to_owned()
}

fn default_port() -> u16 {
    tracing::trace!(target: "openai_compat::config", "default_port called");
    11438
}

impl Default for OpenAiCompatConfig {
    fn default() -> Self {
        tracing::debug!(target: "openai_compat::config", "OpenAiCompatConfig::default called");
        Self {
            enabled: false,
            host: default_host(),
            port: default_port(),
            api_key: None,
        }
    }
}

impl OpenAiCompatConfig {
    pub fn validate(&self) -> Result<(), String> {
        tracing::debug!(
            target: "openai_compat::config",
            enabled = self.enabled,
            host = %self.host,
            port = self.port,
            has_api_key = self.api_key.is_some(),
            api_key_len = self.api_key.as_ref().map(|k| k.len()).unwrap_or(0),
            "validate called"
        );
        if self.host.trim().is_empty() {
            tracing::warn!(target: "openai_compat::config", host = %self.host, "validate failed: host empty");
            return Err("host 不能为空".to_owned());
        }
        if self.port == 0 {
            tracing::warn!(target: "openai_compat::config", port = self.port, "validate failed: port zero");
            return Err("port 必须为 1..65535".to_owned());
        }
        if let Some(key) = &self.api_key {
            if key.trim().is_empty() {
                tracing::warn!(target: "openai_compat::config", "validate failed: apiKey empty after trim");
                return Err("apiKey 若提供则不能为空白".to_owned());
            }
            if key.len() > 256 {
                tracing::warn!(target: "openai_compat::config", api_key_len = key.len(), "validate failed: apiKey too long");
                return Err("apiKey 过长".to_owned());
            }
        }
        tracing::debug!(
            target: "openai_compat::config",
            host = %self.host,
            port = self.port,
            enabled = self.enabled,
            "validate succeeded"
        );
        Ok(())
    }

    #[allow(dead_code)]
    pub fn bound_address(&self) -> String {
        let addr = format!("{}:{}", self.host, self.port);
        tracing::trace!(target: "openai_compat::config", host = %self.host, port = self.port, addr = %addr, "bound_address");
        addr
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize)]

#[serde(rename_all = "camelCase")]
pub struct OpenAiCompatStatus {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub bound_port: Option<u16>,
    pub running: bool,
    pub has_api_key: bool,
    pub error: Option<String>,
}

impl Default for OpenAiCompatStatus {
    fn default() -> Self {
        tracing::trace!(target: "openai_compat::config", "OpenAiCompatStatus::default called");
        Self {
            enabled: false,
            host: default_host(),
            port: default_port(),
            bound_port: None,
            running: false,
            has_api_key: false,
            error: None,
        }
    }
}
