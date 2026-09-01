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
    "127.0.0.1".to_owned()
}

fn default_port() -> u16 {
    11438
}

impl Default for OpenAiCompatConfig {
    fn default() -> Self {
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
        if self.host.trim().is_empty() {
            return Err("host 不能为空".to_owned());
        }
        if self.port == 0 {
            return Err("port 必须为 1..65535".to_owned());
        }
        if let Some(key) = &self.api_key {
            if key.trim().is_empty() {
                return Err("apiKey 若提供则不能为空白".to_owned());
            }
            if key.len() > 256 {
                return Err("apiKey 过长".to_owned());
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn bound_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
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
