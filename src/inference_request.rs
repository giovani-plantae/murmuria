//! Request validation for `POST /inference`.
//!
//! All validation lives in an axum extractor so the handler only ever sees an
//! already-valid `InferenceRequest`: the WAV is decoded to PCM, the language is
//! one Whisper recognizes, the temperature is in range, and an unsupported
//! response format is rejected at the boundary rather than silently downgraded.

use axum::{
    async_trait,
    extract::{multipart::Field, FromRequest, Multipart, Request},
    http::StatusCode,
};

use crate::audio::decode_wav_mono_16k;

/// The 400 response this module returns on any validation failure.
type Rejection = (StatusCode, String);

/// A validated `POST /inference` request.
pub struct InferenceRequest {
    /// 16 kHz mono PCM, ready for Whisper.
    pub samples: Vec<f32>,
    /// A Whisper language code/name, or `None` to use the server's default.
    pub language: Option<String>,
    /// Sampling temperature in `0.0..=1.0`, or `None` to keep Whisper's default.
    pub temperature: Option<f32>,
}

#[async_trait]
impl<S> FromRequest<S> for InferenceRequest
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, String);

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        let mut multipart = Multipart::from_request(request, state)
            .await
            .map_err(|error| bad_request(format!("invalid multipart body: {error}")))?;

        let mut file_bytes: Option<Vec<u8>> = None;
        let mut language: Option<String> = None;
        let mut temperature: Option<String> = None;
        let mut response_format: Option<String> = None;

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|error| bad_request(format!("could not read a multipart field: {error}")))?
        {
            // `name()` borrows the field, but reading its value consumes it, so
            // take an owned copy of the name first to release the borrow.
            let name = field.name().map(str::to_owned);
            match name.as_deref() {
                Some("file") => {
                    let bytes = field.bytes().await.map_err(|error| {
                        bad_request(format!("could not read the 'file' field: {error}"))
                    })?;
                    file_bytes = Some(bytes.to_vec());
                }
                Some("language") => language = Some(read_text(field, "language").await?),
                Some("temperature") => temperature = Some(read_text(field, "temperature").await?),
                Some("response_format") => {
                    response_format = Some(read_text(field, "response_format").await?)
                }
                _ => {}
            }
        }

        let file_bytes =
            file_bytes.ok_or_else(|| bad_request("missing 'file' field (WAV audio)"))?;
        let samples = decode_wav_mono_16k(&file_bytes)
            .map_err(|error| bad_request(format!("invalid WAV: {error}")))?;

        let language = parse_language(language.as_deref().unwrap_or_default())?;
        let temperature = parse_temperature(temperature.as_deref())?;
        validate_response_format(response_format.as_deref())?;

        Ok(Self {
            samples,
            language,
            temperature,
        })
    }
}

/// Reads a multipart field as UTF-8 text, attributing a decode failure to the
/// named field so the client gets an actionable 400.
async fn read_text(field: Field<'_>, name: &str) -> Result<String, Rejection> {
    field
        .text()
        .await
        .map_err(|error| bad_request(format!("could not read the '{name}' field: {error}")))
}

/// Validates the `language` field. An empty value means "unset" (the server
/// falls back to its configured default); otherwise it must be something
/// Whisper recognizes — a code (`pt`), a name (`portuguese`), or `auto`.
fn parse_language(raw: &str) -> Result<Option<String>, Rejection> {
    let language = raw.trim().to_lowercase();
    if language.is_empty() {
        return Ok(None);
    }
    if !is_language_supported(&language) {
        return Err(bad_request(format!("unknown language '{language}'")));
    }
    Ok(Some(language))
}

/// Whether Whisper recognizes the language. `auto` is whisper.cpp's special
/// auto-detect value (its language table does not contain it), so it is allowed
/// explicitly; everything else is checked against that table.
fn is_language_supported(language: &str) -> bool {
    if language == "auto" {
        return true;
    }
    // `get_lang_id` panics on an interior null byte; never hand it one.
    if language.contains('\0') {
        return false;
    }
    whisper_rs::get_lang_id(language).is_some()
}

/// Parses the optional `temperature` field. Absent or blank means "unset" (keep
/// Whisper's default); otherwise it must be a number in `0.0..=1.0`, which also
/// rejects NaN and infinities since neither is contained in the range.
fn parse_temperature(raw: Option<&str>) -> Result<Option<f32>, Rejection> {
    let Some(value) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    let temperature: f32 = value
        .parse()
        .map_err(|_| bad_request(format!("temperature must be a number, got '{value}'")))?;
    if !(0.0..=1.0).contains(&temperature) {
        return Err(bad_request(format!(
            "temperature must be between 0.0 and 1.0, got {temperature}"
        )));
    }
    Ok(Some(temperature))
}

/// Validates the optional `response_format` field. Absent or empty defaults to
/// JSON; any other value is rejected because the server only produces JSON.
fn validate_response_format(raw: Option<&str>) -> Result<(), Rejection> {
    match raw.map(|value| value.trim().to_lowercase()) {
        None => Ok(()),
        Some(value) if value.is_empty() || value == "json" => Ok(()),
        Some(other) => Err(bad_request(format!(
            "unsupported response_format '{other}' (only 'json' is supported)"
        ))),
    }
}

fn bad_request(message: impl Into<String>) -> Rejection {
    (StatusCode::BAD_REQUEST, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_parses_and_enforces_bounds() {
        assert_eq!(parse_temperature(Some("0")).unwrap(), Some(0.0));
        assert_eq!(parse_temperature(Some("0.5")).unwrap(), Some(0.5));
        assert_eq!(parse_temperature(Some("1")).unwrap(), Some(1.0));
        assert_eq!(parse_temperature(None).unwrap(), None);
        assert_eq!(parse_temperature(Some("   ")).unwrap(), None);
        assert!(parse_temperature(Some("abc")).is_err());
        assert!(parse_temperature(Some("1.5")).is_err());
        assert!(parse_temperature(Some("-0.1")).is_err());
        assert!(parse_temperature(Some("NaN")).is_err());
    }

    #[test]
    fn response_format_accepts_only_json() {
        assert!(validate_response_format(None).is_ok());
        assert!(validate_response_format(Some("json")).is_ok());
        assert!(validate_response_format(Some(" JSON ")).is_ok());
        assert!(validate_response_format(Some("")).is_ok());
        assert!(validate_response_format(Some("srt")).is_err());
        assert!(validate_response_format(Some("text")).is_err());
    }

    #[test]
    fn language_treats_empty_as_unset() {
        assert_eq!(parse_language("").unwrap(), None);
        assert_eq!(parse_language("   ").unwrap(), None);
    }

    #[test]
    fn language_accepts_known_values_and_rejects_unknown() {
        assert_eq!(parse_language("auto").unwrap(), Some("auto".to_string()));
        assert_eq!(
            parse_language("Portuguese").unwrap(),
            Some("portuguese".to_string())
        );
        assert_eq!(parse_language("en").unwrap(), Some("en".to_string()));
        assert!(parse_language("klingon").is_err());
    }
}
