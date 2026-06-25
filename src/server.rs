use std::sync::Arc;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use tokio::sync::Semaphore;

use crate::inference_request::InferenceRequest;
use crate::transcribe::{TranscribeOptions, Transcriber};

/// ~2 s of new 16 kHz audio between streaming partial results.
const STREAM_STEP_SAMPLES: usize = 32_000;
/// Cap each streaming pass to the most recent ~30 s of audio.
const STREAM_WINDOW_SAMPLES: usize = 480_000;
/// Upper bound on an `/inference` upload. Whisper WAV (16 kHz mono) is ~32 KB/s,
/// so 50 MB is ~27 min of audio; axum's 2 MB default would reject ~1 min. The
/// limit wraps only the upload route, leaving /stream's long-lived socket alone.
const MAX_UPLOAD_BYTES: usize = 50 * 1024 * 1024;

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
        .route(
            "/inference",
            post(inference).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
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
/// the duration so only one inference runs at a time. `options` are moved into
/// the worker so the validated language outlives the borrow Whisper takes.
async fn run_transcription(
    state: &AppState,
    samples: Vec<f32>,
    options: TranscribeOptions,
) -> Result<String, String> {
    let _permit = state.gpu.acquire().await.map_err(|e| e.to_string())?;
    let transcriber = state.transcriber.clone();
    tokio::task::spawn_blocking(move || transcriber.transcribe(&samples, &options))
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct InferenceResponse {
    text: String,
}

/// File upload → transcript. The `InferenceRequest` extractor has already
/// validated the multipart body, so this only runs transcription with the
/// per-request options.
async fn inference(State(state): State<AppState>, request: InferenceRequest) -> impl IntoResponse {
    let options = TranscribeOptions {
        language: request.language,
        temperature: request.temperature,
    };

    match run_transcription(&state, request.samples, options).await {
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
    // The streaming path carries no language field, so it always uses the
    // server's startup-default language and Whisper's default temperature.
    let Ok(text) = run_transcription(state, window, TranscribeOptions::default()).await else {
        return true;
    };
    let payload = serde_json::json!({ "type": kind, "text": text }).to_string();
    socket.send(Message::Text(payload)).await.is_ok()
}
