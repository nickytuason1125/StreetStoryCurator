//! framegrade-rs — FrameGrade native orchestrator, slice 1.
//!
//! Implements the state/telemetry surface of the Python server's API with
//! byte-compatible response shapes, so the React frontend can point at this
//! process without knowing the difference. Every JSON write is ATOMIC
//! (temp file + rename): the flaw class found in the Python server during
//! the 2026-08 audit cannot recur here.
//!
//! Not yet in Rust (still on Python until parity): grading pipeline,
//! embeddings, LLM/VLM inference, SSE streams.

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::PathBuf;
use sysinfo::System;

#[derive(Clone)]
struct App {
    root: PathBuf,
    sys: std::sync::Arc<std::sync::Mutex<System>>,
}

impl App {
    fn new() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        Self {
            root: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("..")
                .join(".."),
            sys: std::sync::Arc::new(std::sync::Mutex::new(sys)),
        }
    }
    fn cache_path(&self, name: &str) -> PathBuf {
        self.root.join("cache").join(name)
    }
    /// Atomic write: temp file + rename. The whole point of this binary.
    fn atomic_write(&self, path: &PathBuf, text: &str) -> Result<(), String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension(format!("json.{}.tmp", std::process::id()));
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    }
    fn read_json(&self, path: &PathBuf, fallback: Value) -> Value {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(fallback)
    }
}

// ── handlers: telemetry ─────────────────────────────────────────────────────

async fn system_ram(State(app): State<App>) -> Json<Value> {
    let s = app.sys.lock().unwrap();
    let total = s.total_memory() as f64 / 2f64.powi(30); // KiB → GiB
    let free = s.available_memory() as f64 / 2f64.powi(30);
    drop(s);
    let percent = if total > 0.0 { (total - free) / total * 100.0 } else { 0.0 };
    Json(json!({
        "ram_free_gb":  (free * 10.0).round() / 10.0,
        "ram_total_gb": (total * 10.0).round() / 10.0,
        "ram_percent":  (percent * 10.0).round() / 10.0,
        "ram_min_gb":   1.8,
    }))
}

async fn config() -> Json<Value> {
    Json(json!({ "force_frontier": false }))
}

// ── handlers: state stores ──────────────────────────────────────────────────

async fn flags_load(State(app): State<App>) -> Json<Value> {
    let v = app.read_json(
        &app.cache_path("photo_flags.json"),
        json!({"locked": [], "used": []}),
    );
    Json(v)
}

#[derive(Deserialize)]
struct FlagToggleReq {
    path: String,
}

async fn flag_toggle(
    State(app): State<App>,
    Json(body): Json<FlagToggleReq>,
    key: &'static str,
) -> Result<Json<Value>, StatusCode> {
    if body.path.trim().is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let path = app.cache_path("photo_flags.json");
    let mut data = app.read_json(&path, json!({"locked": [], "used": []}));
    let items = data[key].as_array().cloned().unwrap_or_default();
    let needle = Value::String(body.path.clone());
    let present = items.contains(&needle);
    let mut out: Vec<Value> = items.into_iter().filter(|v| *v != needle).collect();
    if !present {
        out.push(needle);
    }
    data[key] = Value::Array(out);
    let text =
        serde_json::to_string_pretty(&data).map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    app.atomic_write(&path, &text)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(json!({ "success": true, key: !present })))
}

async fn flags_lock(
    State(app): State<App>,
    Json(b): Json<FlagToggleReq>,
) -> Result<Json<Value>, StatusCode> {
    flag_toggle(State(app), Json(b), "locked").await
}

async fn flags_used(
    State(app): State<App>,
    Json(b): Json<FlagToggleReq>,
) -> Result<Json<Value>, StatusCode> {
    flag_toggle(State(app), Json(b), "used").await
}

async fn catalog_get(State(app): State<App>) -> Json<Value> {
    let path = app.cache_path("catalog.json");
    if !path.exists() {
        return Json(json!({ "exists": false }));
    }
    let mut v = app.read_json(&path, Value::Null);
    if v.is_null() {
        return Json(json!({ "exists": false }));
    }
    v["exists"] = json!(true);
    Json(v)
}

#[derive(Deserialize)]
struct CatalogSave {
    #[serde(default)]
    photos: Value,
    #[serde(default)]
    folders: Value,
}

async fn catalog_save(
    State(app): State<App>,
    Json(body): Json<CatalogSave>,
) -> Json<Value> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let doc = json!({
        "photos": body.photos,
        "folders": body.folders,
        "saved_at": format!("epoch:{secs}"),
    });
    let text = serde_json::to_string_pretty(&doc).unwrap_or_default();
    match app.atomic_write(&app.cache_path("catalog.json"), &text) {
        Ok(_) => Json(json!({ "ok": true })),
        Err(e) => Json(json!({ "ok": false, "error": e })),
    }
}

// ── main ────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let port: u16 = std::env::var("FRAMEGRADE_RS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8001);

    let app_state = App::new();

    let toggle_routes = Router::new()
        .route("/lock", post(flags_lock))
        .route("/used", post(flags_used))
        .with_state(app_state.clone());

    let app = Router::new()
        .route("/api/system/ram", get(system_ram))
        .route("/api/config", get(config))
        .route("/api/flags/load", get(flags_load))
        .route("/api/catalog", get(catalog_get))
        .route("/api/catalog/save", post(catalog_save))
        .nest("/api/flags", toggle_routes)
        .with_state(app_state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    println!("[framegrade-rs] listening on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
    axum::serve(listener, app).await.unwrap();
}


