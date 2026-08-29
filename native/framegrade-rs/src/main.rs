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

// ── cull RAM requirement ────────────────────────────────────────────────────
// MIRRORS src/run_profile.py::required_ram_gb. Two implementations of one
// number is exactly the drift that put a stale `1.8` in this file's
// system_ram handler while the Python gate had moved to 3.8 — the shim told
// the photographer a cull would fit and the server then refused it.
//
// This crate cannot import Python, so the table is copied verbatim and locked
// by the tests at the bottom of this file. If run_profile's measurements are
// redone, change both or the tests fail.
//
//   photos   draft ON   draft OFF
//    <=300     3.8 GB     6.6 GB
//     >300     4.2 GB     7.0 GB

/// Pure arithmetic — no env, no I/O — so it is testable without races.
fn ram_need_gb(draft: bool, n_photos: u32, override_gb: Option<f64>) -> f64 {
    // Python: `return override if override > 0 else need`. A zero or negative
    // override means "unset", never "no floor".
    if let Some(v) = override_gb {
        if v > 0.0 {
            return v;
        }
    }
    match (draft, n_photos <= 300) {
        (true, true) => 3.8,
        (true, false) => 4.2,
        (false, true) => 6.6,
        (false, false) => 7.0,
    }
}

/// Env-reading wrapper. Same two variables the Python side honours:
/// FRAMEGRADE_MIN_RAM_GB (absolute override) and FRAMEGRADE_DRAFT_DECODE
/// ("0" disables scaled decode, which roughly doubles the requirement).
fn required_ram_gb(n_photos: u32) -> f64 {
    let override_gb = std::env::var("FRAMEGRADE_MIN_RAM_GB")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok());
    let draft = std::env::var("FRAMEGRADE_DRAFT_DECODE")
        .map(|s| s.trim() != "0")
        .unwrap_or(true);
    ram_need_gb(draft, n_photos, override_gb)
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
        // n_photos = 0: this endpoint is polled with no job in front of it, so
        // it reports the floor for a small cull. The real per-cull gate lives
        // in the Python grading router, which knows the folder size.
        "ram_min_gb":   required_ram_gb(0),
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

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::ram_need_gb;

    // This crate claims byte-compatible responses with the Python server, so a
    // number it invents is a bug even when it is plausible. These lock the
    // table to src/run_profile.py::_RAM_NEED_GB. If that table is re-measured,
    // both sides move together or this fails — which is the point.
    //
    // Pure function, no env reads: env is process-global and `cargo test` runs
    // threads in parallel, so testing the wrapper directly would be flaky.

    #[test]
    fn draft_on_small_job_matches_python() {
        assert_eq!(ram_need_gb(true, 0, None), 3.8);
        assert_eq!(ram_need_gb(true, 58, None), 3.8);
        assert_eq!(ram_need_gb(true, 300, None), 3.8);
    }

    #[test]
    fn draft_on_large_job_uses_the_extrapolated_figure() {
        assert_eq!(ram_need_gb(true, 301, None), 4.2);
        assert_eq!(ram_need_gb(true, 5000, None), 4.2);
    }

    #[test]
    fn draft_off_roughly_doubles_the_requirement() {
        // Full-resolution decode was measured at 5.85-6.35 GB against
        // 2.89-3.60 GB drafting. The gate is deliberately above the measured
        // peak, not at it.
        assert_eq!(ram_need_gb(false, 0, None), 6.6);
        assert_eq!(ram_need_gb(false, 301, None), 7.0);
    }

    #[test]
    fn an_explicit_override_wins_over_every_branch() {
        assert_eq!(ram_need_gb(true, 0, Some(2.5)), 2.5);
        assert_eq!(ram_need_gb(false, 9000, Some(2.5)), 2.5);
    }

    #[test]
    fn a_zero_or_negative_override_is_ignored() {
        // Python treats "" and 0 as absent (`override if override > 0`). A
        // gate of 0 GB would admit every cull, so this must not be a way to
        // switch the gate off by accident.
        assert_eq!(ram_need_gb(true, 0, Some(0.0)), 3.8);
        assert_eq!(ram_need_gb(true, 0, Some(-1.0)), 3.8);
    }

    #[test]
    fn the_dead_floor_is_gone() {
        // 1.8 was Balanced's ENCODER floor masquerading as a whole-cull
        // budget. No branch may return it.
        for &draft in &[true, false] {
            for &n in &[0u32, 58, 300, 301, 5000] {
                assert_ne!(ram_need_gb(draft, n, None), 1.8);
            }
        }
    }
}

