//! murmuria — self-hosted Whisper transcription server.
//!
//! HTTP application layer (axum). Inference runs in whisper.cpp through the
//! `whisper-rs` binding (Vulkan backend when built with the `vulkan` feature).

mod audio;
mod discovery;
mod inference_request;
mod server;
mod transcribe;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use server::AppState;
use transcribe::Transcriber;

#[tokio::main]
async fn main() {
    let model = std::env::var("MURMURIA_MODEL")
        .unwrap_or_else(|_| "/models/ggml-large-v3-turbo-q5_0.bin".to_string());
    let language = std::env::var("MURMURIA_LANGUAGE").unwrap_or_else(|_| "en".to_string());
    // A set --port flag or MURMURIA_PORT pins the port; unset, the server scans
    // upward from BASE_PORT for the first free one (like `artisan serve`).
    let requested_port: Option<u16> = cli_flag("--port")
        .or_else(|| std::env::var("MURMURIA_PORT").ok())
        .map(|raw| {
            raw.trim().parse().unwrap_or_else(|_| {
                panic!("invalid port {raw:?} (--port / MURMURIA_PORT must be a number)")
            })
        });
    // Force CPU even on a GPU (Vulkan) build: --cpu flag or MURMURIA_CPU truthy.
    let force_cpu = cli_present("--cpu")
        || matches!(
            std::env::var("MURMURIA_CPU").as_deref(),
            Ok("1") | Ok("true")
        );

    if std::env::args().any(|arg| arg == "doctor") {
        std::process::exit(doctor(&model, &language, force_cpu));
    }

    let backend = if cfg!(feature = "vulkan") && !force_cpu {
        "GPU (Vulkan)"
    } else {
        "CPU"
    };
    println!("murmuria: loading model {model} (language: {language}, backend: {backend})…");
    let transcriber = Transcriber::load(&PathBuf::from(&model), language, force_cpu)
        .expect("failed to load the Whisper model");
    let state = AppState {
        transcriber: Arc::new(transcriber),
        // One permit: serialize inference so the single iGPU isn't thrashed.
        gpu: Arc::new(tokio::sync::Semaphore::new(1)),
    };

    let (listener, port) = bind_listener(requested_port).await;
    println!("murmuria listening on http://0.0.0.0:{port}");

    // Held for the process lifetime so the mDNS responder keeps answering; best
    // effort, so a missing mDNS stack never stops the server from serving.
    let _mdns = discovery::advertise(port);

    axum::serve(listener, server::router(state))
        .await
        .expect("server error");
}

/// First port tried when none is requested (like Laravel's `artisan serve`).
const BASE_PORT: u16 = 8000;
/// How many ports to scan upward from [`BASE_PORT`] before giving up.
const PORT_SCAN_SPAN: u16 = 64;

/// Binds the server socket. A requested port is pinned — binding it is fatal if
/// it's taken; otherwise the first free port from [`BASE_PORT`] upward is used,
/// like Laravel's `artisan serve`. Returns the listener and the bound port.
async fn bind_listener(requested: Option<u16>) -> (tokio::net::TcpListener, u16) {
    if let Some(port) = requested {
        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
            .await
            .unwrap_or_else(|error| panic!("failed to bind port {port}: {error}"));
        return (listener, port);
    }

    let last_port = BASE_PORT.saturating_add(PORT_SCAN_SPAN);
    for port in BASE_PORT..last_port {
        if let Ok(listener) = tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
            if port != BASE_PORT {
                println!("murmuria: port {BASE_PORT} busy, using {port}");
            }
            return (listener, port);
        }
    }
    panic!("no free port in {BASE_PORT}..{last_port}");
}

/// Reads a `--flag value` / `--flag=value` CLI option, returning its value if
/// present. Lets flags override the matching environment variable.
fn cli_flag(name: &str) -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix(&format!("{name}=")) {
            return Some(value.to_string());
        }
        if arg == name {
            return args.next();
        }
    }
    None
}

/// Returns whether a boolean CLI flag (e.g. `--cpu`) is present.
fn cli_present(name: &str) -> bool {
    std::env::args().skip(1).any(|arg| arg == name)
}

/// Self-diagnostic (`murmuria doctor`): checks the model and that the backend
/// loads, so a fresh clone can be verified in a single command.
fn doctor(model: &str, language: &str, force_cpu: bool) -> i32 {
    println!("murmuria doctor");
    println!("  MURMURIA_MODEL    = {model}");
    println!("  MURMURIA_LANGUAGE = {language}");

    let path = Path::new(model);
    match std::fs::metadata(path) {
        Ok(meta) => println!(
            "  ✓ model found ({:.0} MB)",
            meta.len() as f64 / 1_048_576.0
        ),
        Err(_) => {
            println!("  ✗ model NOT found at {model}");
            println!("    → run `just model` or set MURMURIA_MODEL");
            return 1;
        }
    }

    let backend = if cfg!(feature = "vulkan") && !force_cpu {
        "GPU (Vulkan)"
    } else {
        "CPU"
    };
    print!("  loading the model (initializes the {backend} backend)… ");
    let _ = std::io::stdout().flush();
    match Transcriber::load(path, language, force_cpu) {
        Ok(_) => {
            println!("✓ OK");
            println!("all set — `just run` should work.");
            0
        }
        Err(error) => {
            println!("✗ {error}");
            1
        }
    }
}
