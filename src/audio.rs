//! WAV decoding for the transcription endpoints.

use std::io::Cursor;

const TARGET_RATE: u32 = 16_000;

/// Decodes a WAV byte buffer into mono `f32` samples at 16 kHz (what whisper.cpp
/// expects). Multi-channel audio is down-mixed to mono and any sample rate is
/// linearly resampled to 16 kHz.
pub fn decode_wav_mono_16k(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let reader = hound::WavReader::new(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?
        }
    };

    let mono: Vec<f32> = if channels <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok(resample_linear(&mono, spec.sample_rate, TARGET_RATE))
}

/// Linear-interpolation resampler. Good enough for speech ASR; for higher
/// fidelity a windowed/polyphase resampler (e.g. `rubato`) could replace this.
fn resample_linear(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.len() < 2 {
        return input.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = input.len() - 1;

    for i in 0..out_len {
        let source = i as f64 / ratio;
        let index = source.floor() as usize;
        let frac = (source - index as f64) as f32;
        let a = input[index.min(last)];
        let b = input[(index + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}
