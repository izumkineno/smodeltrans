#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BackendFailureCode {
    Arguments,
    Cancelled,
    Device,
    Asset,
    Ocr,
    Translation,
    Output,
    Internal,
}

impl BackendFailureCode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Arguments => "arguments",
            Self::Cancelled => "cancelled",
            Self::Device => "device",
            Self::Asset => "asset",
            Self::Ocr => "ocr",
            Self::Translation => "translation",
            Self::Output => "output",
            Self::Internal => "internal",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BackendFailure {
    code: BackendFailureCode,
    message: String,
}

impl BackendFailure {
    pub(crate) fn new(code: BackendFailureCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn arguments(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Arguments, message)
    }

    pub(crate) fn cancelled(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Cancelled, message)
    }

    pub(crate) fn device(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Device, message)
    }

    pub(crate) fn asset(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Asset, message)
    }

    pub(crate) fn ocr(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Ocr, message)
    }

    pub(crate) fn translation(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Translation, message)
    }

    pub(crate) fn output(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Output, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(BackendFailureCode::Internal, message)
    }

    pub(crate) fn code(&self) -> BackendFailureCode {
        self.code
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

impl std::fmt::Display for BackendFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for BackendFailure {}
