use std::sync::Arc;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tokio::sync::Semaphore;

use crate::audio::decode_wav_mono_16k;
use crate::transcribe::Transcriber;

/// ~2 s of new 16 kHz audio between streaming partial results.
const STREAM_STEP_SAMPLES: usize = 32_000;
/// Cap each streaming pass to the most recent ~30 s of audio.
const STREAM_WINDOW_SAMPLES: usize = 480_000;

/// Shared application state. The GPU semaphore serializes inference so the
/// single iGPU isn't thrashed by concurrent transcriptions.
#[derive(Clone)]
pub struct AppState {
    pub transcriber: Arc<Transcriber>,
    pub gpu: Arc<Semaphore>,
}

/// Builds the HTTP router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        // Axum caps request bodies at 2 MB by default — only ~1 min of 16 kHz
        // mono WAV — so longer voice messages fail while reading the 'file'
        // field. Raise it to 50 MB (~27 min) for uploads only.
        .route(
            "/inference",
            post(inference).layer(DefaultBodyLimit::max(50 * 1024 * 1024)),
        )
        .route("/stream", get(stream))
        .with_state(state)
}

/// Liveness probe that also identifies the service, so a client probing an
/// unknown address can tell a murmuria server apart from anything else listening
/// on the same port.
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({
        "service": "murmuria",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Transcribes owned samples on a blocking worker, holding the GPU permit for
/// the duration so only one inference runs at a time.
async fn run_transcription(state: &AppState, samples: Vec<f32>) -> Result<String, String> {
    let _permit = state.gpu.acquire().await.map_err(|e| e.to_string())?;
    let transcriber = state.transcriber.clone();
    tokio::task::spawn_blocking(move || transcriber.transcribe(&samples))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct InferenceResponse {
    text: String,
}

/// File upload → transcript. Reads the multipart `file` field (WAV), decodes it
/// to 16 kHz mono PCM, and transcribes it.
async fn inference(State(state): State<AppState>, mut multipart: Multipart) -> impl IntoResponse {
    let mut audio: Option<Vec<u8>> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            match field.bytes().await {
                Ok(bytes) => audio = Some(bytes.to_vec()),
                Err(_) => {
                    return (StatusCode::BAD_REQUEST, "could not read the 'file' field")
                        .into_response()
                }
            }
        }
    }

    let Some(bytes) = audio else {
        return (StatusCode::BAD_REQUEST, "missing 'file' field (WAV audio)").into_response();
    };

    let samples = match decode_wav_mono_16k(&bytes) {
        Ok(samples) => samples,
        Err(error) => {
            return (StatusCode::BAD_REQUEST, format!("invalid WAV: {error}")).into_response()
        }
    };

    match run_transcription(&state, samples).await {
        Ok(text) => Json(InferenceResponse { text }).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("transcription failed: {error}"),
        )
            .into_response(),
    }
}

/// Realtime streaming over WebSocket.
///
/// Protocol: the client sends binary frames of 16 kHz mono PCM as little-endian
/// `f32` samples. The server emits JSON text messages `{"type":"partial","text"}`
/// every ~2 s of new audio (over a sliding 30 s window) and, on any text message
/// (a flush) or on close, `{"type":"final","text"}` — then resets the buffer.
async fn stream(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_stream(socket, state))
}

async fn handle_stream(mut socket: WebSocket, state: AppState) {
    let mut buffer: Vec<f32> = Vec::new();
    let mut transcribed_at: usize = 0;

    while let Some(Ok(message)) = socket.recv().await {
        match message {
            Message::Binary(bytes) => {
                for frame in bytes.chunks_exact(4) {
                    buffer.push(f32::from_le_bytes([frame[0], frame[1], frame[2], frame[3]]));
                }
                if buffer.len().saturating_sub(transcribed_at) >= STREAM_STEP_SAMPLES {
                    transcribed_at = buffer.len();
                    if !emit(&mut socket, &state, &buffer, "partial").await {
                        break;
                    }
                }
            }
            Message::Text(_) => {
                emit(&mut socket, &state, &buffer, "final").await;
                buffer.clear();
                transcribed_at = 0;
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
}

/// Transcribes the tail window of `buffer` and sends one `{type, text}` message.
/// Returns `false` if the socket send failed (caller should stop).
async fn emit(socket: &mut WebSocket, state: &AppState, buffer: &[f32], kind: &str) -> bool {
    if buffer.is_empty() {
        return true;
    }
    let window = buffer[buffer.len().saturating_sub(STREAM_WINDOW_SAMPLES)..].to_vec();
    let Ok(text) = run_transcription(state, window).await else {
        return true;
    };
    let payload = serde_json::json!({ "type": kind, "text": text }).to_string();
    socket.send(Message::Text(payload)).await.is_ok()
}
