//! Transcription engine — whisper.cpp via the `whisper-rs` binding (Vulkan
//! backend when built with the `vulkan` feature; CPU otherwise).

use std::path::Path;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

    /// Transcribes mono 16 kHz PCM samples into a single text string.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String, whisper_rs::WhisperError> {
        let mut state = self.context.create_state()?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(self.language.as_str()));
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
