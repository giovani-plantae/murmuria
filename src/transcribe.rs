//! Transcription engine — whisper.cpp via the `whisper-rs` binding (Vulkan
//! backend when built with the `vulkan` feature; CPU otherwise).

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Per-request transcription overrides. A `None` field keeps the server's
/// configured default (the startup `--language`, and Whisper's own temperature).
#[derive(Default)]
pub struct TranscribeOptions {
    /// A Whisper language code/name (e.g. `pt`, `portuguese`, `auto`).
    pub language: Option<String>,
    /// Sampling temperature; the request layer constrains it to `0.0..=1.0`.
    pub temperature: Option<f32>,
}

/// Loads a Whisper model once and transcribes 16 kHz mono PCM against it.
pub struct Transcriber {
    context: WhisperContext,
    language: String,
}

impl Transcriber {
    /// Loads a GGML model (e.g. `large-v3-turbo-q5_0`) and binds a forced
    /// transcription language (a whisper language code, e.g. `"pt"`). Pass
    /// `force_cpu` to run on CPU even when the GPU (Vulkan) backend is compiled
    /// in; otherwise the build's default backend is used.
    pub fn load(
        model_path: &Path,
        language: impl Into<String>,
        force_cpu: bool,
    ) -> Result<Self, whisper_rs::WhisperError> {
        let mut context_params = WhisperContextParameters::default();
        if force_cpu {
            context_params.use_gpu = false;
        }
        let context = WhisperContext::new_with_params(model_path, context_params)?;
        Ok(Self {
            context,
            language: language.into(),
        })
    }

    /// Transcribes mono 16 kHz PCM samples into a single text string. `options`
    /// override the forced language and sampling temperature per request; unset
    /// fields fall back to the server's configured defaults.
    pub fn transcribe(
        &self,
        samples: &[f32],
        options: &TranscribeOptions,
    ) -> Result<String, whisper_rs::WhisperError> {
        let mut state = self.context.create_state()?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let language = options
            .language
            .as_deref()
            .unwrap_or(self.language.as_str());
        params.set_language(Some(language));
        if let Some(temperature) = options.temperature {
            params.set_temperature(temperature);
        }
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, samples)?;

        let mut text = String::new();
        for segment in state.as_iter() {
            text.push_str(&segment.to_str_lossy()?);
        }
        Ok(text.trim().to_string())
    }
}
