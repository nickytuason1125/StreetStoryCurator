// Prevents an extra console window on Windows in release, while keeping it in debug.
use std::env;
use std::sync::{Arc, Mutex};
use std::process::{Child, Command};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use std::time::Duration;
use std::thread;
use std::path::PathBuf;
use tauri::{Manager, RunEvent};

// ─── Platform / arch triple helpers ─────────────────────────────────────────

/// Return the Tauri sidecar platform triple for the current target,
/// e.g. "x86_64-pc-windows-msvc" or "aarch64-apple-darwin".
fn sidecar_triple() -> &'static str {
    cfg_if_triple()
}

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
fn cfg_if_triple() -> &'static str { "x86_64-pc-windows-msvc" }

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
fn cfg_if_triple() -> &'static str { "aarch64-pc-windows-msvc" }

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn cfg_if_triple() -> &'static str { "x86_64-apple-darwin" }

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn cfg_if_triple() -> &'static str { "aarch64-apple-darwin" }

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn cfg_if_triple() -> &'static str { "x86_64-unknown-linux-gnu" }

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
fn cfg_if_triple() -> &'static str { "aarch64-unknown-linux-gnu" }

/// Extension for the sidecar binary (empty on Unix, ".exe" on Windows).
fn exe_ext() -> &'static str {
    if cfg!(target_os = "windows") { ".exe" } else { "" }
}

// ─── Dev-mode helpers (unchanged from original) ──────────────────────────────

/// Walk up from the running executable until we find server.py.
/// Works for both dev builds (exe deep inside src-tauri/target/) and
/// release builds run from the source tree.
fn find_project_root() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| exe.clone());
        for _ in 0..8 {
            if dir.join("server.py").exists() {
                return dir;
            }
            match dir.parent() {
                Some(p) => dir = p.to_path_buf(),
                None => break,
            }
        }
    }
    // Last resort: working directory (works when launched via build_tauri.bat)
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Spawn the Python FastAPI server in dev/fallback mode and return the child handle.
///
/// Uses `src/local_launcher.py` — the proven daily-driver path (it redirects
/// stdout/stderr to crash.log, because under pythonw sys.stdout is None and a
/// bare `server.py` silently dies before uvicorn binds). `server.py` is kept
/// only as a last-resort fallback.
fn start_python_server(project_root: &std::path::Path) -> Option<Child> {
    let venv_python = project_root.join("venv/Scripts/pythonw.exe");
    let launcher_py  = project_root.join("src").join("local_launcher.py");
    let server_py    = project_root.join("server.py");

    let script = if launcher_py.exists() { launcher_py } else { server_py };
    if !script.exists() {
        eprintln!("[tauri] ERROR: no launcher script found at {:?}", script);
        return None;
    }

    let python_exe = if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("pythonw")
    };

    eprintln!("[tauri] Starting: {:?} {:?}", python_exe, script);

    let mut cmd = Command::new(&python_exe);
    cmd.arg(&script).current_dir(project_root);
    // Suppress the console window that would otherwise flash on Windows.
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    cmd.spawn()
        .map_err(|e| eprintln!("[tauri] Failed to spawn server: {}", e))
        .ok()
}

// ─── Release-mode sidecar launcher ──────────────────────────────────────────

/// Extract the engine zip (shipped as a single resource) into
/// <resource_dir>/binaries/curator-api-<triple>/. Runs once on first launch
/// — the NSIS payload stores the engine zipped because tauri-build's
/// resources glob stack-overflows on the 8k-file tree and NSIS is far
/// faster with one entry. A version marker re-triggers extraction when the
/// app version changes, so engine updates ship cleanly.
fn extract_engine_zip(zip_path: &std::path::Path, dest_root: &std::path::Path) -> Result<(), String> {
    eprintln!("[tauri] extracting engine part {:?} (one-time, a few minutes)…", zip_path);
    let file = std::fs::File::open(zip_path).map_err(|e| format!("open engine zip: {}", e))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .map_err(|e| format!("read engine zip: {}", e))?;

    let total = archive.len();
    for i in 0..total {
        let mut entry = archive.by_index(i)
            .map_err(|e| format!("zip entry {}: {}", i, e))?;
        let Some(rel) = entry.enclosed_name().map(|p| p.to_path_buf()) else {
            continue; // skip unsafe paths (.., absolute, etc.)
        };
        let out_path = dest_root.join(rel);
        if entry.is_dir() {
            std::fs::create_dir_all(&out_path)
                .map_err(|e| format!("mkdir {}: {}", out_path.display(), e))?;
            continue;
        }
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
        }
        let mut out = std::fs::File::create(&out_path)
            .map_err(|e| format!("create {}: {}", out_path.display(), e))?;
        std::io::copy(&mut entry, &mut out)
            .map_err(|e| format!("write {}: {}", out_path.display(), e))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Some(mode) = entry.unix_mode() {
                let _ = std::fs::set_permissions(&out_path, std::fs::Permissions::from_mode(mode));
            }
        }
        if i % 1000 == 0 {
            eprintln!("[tauri]   extracted {}/{}", i, total);
        }
    }

    std::fs::write(dest_root.join(".engine-part-ok"), "1")
        .map_err(|e| format!("write marker: {}", e))?;
    eprintln!("[tauri] engine part extracted.");
    Ok(())
}

/// Find engine zip parts (curator-engine-NN.zip) under the given base dirs.
fn collect_engine_parts(bases: &[PathBuf]) -> Vec<PathBuf> {
    let mut parts: Vec<PathBuf> = Vec::new();
    for base in bases {
        if let Ok(rd) = std::fs::read_dir(base) {
            for e in rd.flatten() {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with("curator-engine-") && name.ends_with(".zip") {
                    parts.push(e.path());
                }
            }
        }
    }
    parts.sort();
    parts.dedup();
    parts
}

/// Download one engine part with HTTP range resume. Ok(false) = 404 (no more parts).
fn download_with_resume(url: &str, dest: &std::path::Path) -> Result<bool, String> {
    let have = dest.metadata().map(|m| m.len()).unwrap_or(0);
    let mut resp = match ureq::get(url).set("Range", &format!("bytes={}-", have)).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) if code == 404 => return Ok(false),
        Err(e) => return Err(format!("GET {}: {}", url, e)),
    };
    eprintln!("[tauri]   downloading {} (resuming from {} bytes)", url, have);
    use std::io::Write;
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dest)
        .map_err(|e| format!("open {}: {}", dest.display(), e))?;
    let mut reader = resp.into_reader();
    let mut copied: u64 = 0;
    loop {
        let mut buf = [0u8; 256 * 1024];
        let n = reader.read(&mut buf).map_err(|e| format!("read {}: {}", url, e))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| format!("write {}: {}", dest.display(), e))?;
        copied += n as u64;
        if copied / (128 * 1024 * 1024) != (copied - n as u64) / (128 * 1024 * 1024) {
            eprintln!("[tauri]   … {} MB of this part", copied / (1024 * 1024));
        }
    }
    Ok(true)
}

/// Download every engine part from `base` (e.g. https://host/engine/1.0.0/),
/// extracting each as it lands. Returns the bundle dir on success.
fn download_engine(base: &str, engine_root: &std::path::Path, triple: &str) -> Option<PathBuf> {
    std::fs::create_dir_all(engine_root).ok()?;
    let mut n: u32 = 1;
    loop {
        let name = format!("curator-engine-{:02}.zip", n);
        let part_path = engine_root.join(format!("{}.part", name));
        match download_with_resume(&format!("{}/{}", base, name), &part_path) {
            Ok(true) => {
                extract_engine_zip(&part_path, engine_root).ok()?;
                let _ = std::fs::remove_file(&part_path);
                n += 1;
                if n > 64 {
                    break;
                }
            }
            Ok(false) => break, // 404 — no more parts
            Err(e) => {
                eprintln!("[tauri] ERROR downloading engine: {} (progress kept; relaunch to resume)", e);
                return None;
            }
        }
    }
    if n == 1 {
        eprintln!("[tauri] ERROR: engine download URL has no parts (404 on part 01): {}", base);
        return None;
    }
    let dir = engine_root.join(format!("curator-api-{}", triple));
    if dir.join(format!("curator-api{}", exe_ext())).exists() {
        Some(dir)
    } else {
        eprintln!("[tauri] ERROR: engine downloaded but curator-api exe missing");
        None
    }
}

/// Resolve the engine bundle dir: 1) previously downloaded/extracted engine in
/// the writable data dir, 2) loose sidecar in resources (local builds),
/// 3) zip parts shipped in resources (offline distribution),
/// 4) first-run download from FIRSTCUT_ENGINE_URL (public distribution).
fn resolve_engine(app: &tauri::AppHandle, triple: &str, ext: &str) -> Option<PathBuf> {
    let resource_dir = app.path().resource_dir()
        .map_err(|e| eprintln!("[tauri] resource_dir error: {}", e))
        .ok()?;
    let engine_root = app.path().data_dir().unwrap_or_else(|_| resource_dir.clone()).join("engine");
    let name = format!("curator-api-{}", triple);
    let exe_rel = format!("curator-api{}", ext);

    // 1) Already installed in the writable data dir.
    let installed = engine_root.join(&name);
    if installed.join(&exe_rel).exists() {
        return Some(installed);
    }

    // 2) Loose sidecar next to resources (local/dev builds).
    for base in [resource_dir.join("binaries"), resource_dir.join("resources").join("binaries")] {
        let d = base.join(&name);
        if d.join(&exe_rel).exists() {
            return Some(d);
        }
    }

    // 3) Zip parts shipped inside the installer (offline distribution).
    let parts = collect_engine_parts(&[resource_dir.clone(), resource_dir.join("resources")]);
    if !parts.is_empty() {
        std::fs::create_dir_all(engine_root.join(&name)).ok()?;
        let marker = engine_root.join(&name).join(".engine-version");
        let want = format!("{}\t{}", env!("CARGO_PKG_VERSION"), parts.len());
        if std::fs::read_to_string(&marker).map(|v| v.trim() != want).unwrap_or(true) {
            for part in &parts {
                if let Err(e) = extract_engine_zip(part, &engine_root) {
                    eprintln!("[tauri] ERROR extracting {:?}: {}", part, e);
                    return None;
                }
            }
            let _ = std::fs::write(&marker, &want);
        }
        return Some(engine_root.join(&name));
    }

    // 4) First-run download.
    if let Ok(base) = env::var("FIRSTCUT_ENGINE_URL") {
        if !base.trim().is_empty() {
            eprintln!("[tauri] No engine installed — downloading from {}", base.trim());
            return download_engine(base.trim(), &engine_root, triple);
        }
    }
    eprintln!("[tauri] ERROR: no engine found and FIRSTCUT_ENGINE_URL is not set — cannot start backend.");
    None
}

/// The bundle dir lives at:
///   <resource_dir>/binaries/curator-api-<triple>/
/// and the executable is:
///   curator-api-<triple>/curator-api[.exe]
///
/// We set:
///   CWD          → the bundle dir  (so models/ and frontend/dist/ resolve)
///   CURATOR_DATA_DIR → user's AppData/FirstCut (writable cache)
fn start_sidecar(app: &tauri::AppHandle) -> Option<Child> {
    let triple      = sidecar_triple();
    let ext         = exe_ext();
    let bundle_dir  = resolve_engine(app, &triple, &ext)?;
    let exe_path    = bundle_dir.join(format!("curator-api{}", ext));

    eprintln!("[tauri] Sidecar bundle dir: {:?}", bundle_dir);
    eprintln!("[tauri] Sidecar exe:        {:?}", exe_path);

    if !exe_path.exists() {
        eprintln!("[tauri] ERROR: sidecar exe not found — did you run build-backend.bat?");
        return None;
    }

    // Resolve the writable data directory (AppData\Roaming\FirstCut on Windows)
    let data_dir = app.path().data_dir()
        .map(|d| d.join("FirstCut"))
        .unwrap_or_else(|_| bundle_dir.clone());

    eprintln!("[tauri] CURATOR_DATA_DIR: {:?}", data_dir);

    let mut cmd = Command::new(&exe_path);
    cmd.current_dir(&bundle_dir);
    cmd.env("CURATOR_DATA_DIR", data_dir.to_str().unwrap_or(""));

    // Suppress console window on Windows
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

    cmd.spawn()
        .map_err(|e| eprintln!("[tauri] Failed to spawn sidecar: {}", e))
        .ok()
}

// ─── Poll until the server responds ─────────────────────────────────────────

/// Poll until the server responds or timeout expires.
fn wait_for_server(url: &str, max_wait_secs: u64) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(max_wait_secs);
    while std::time::Instant::now() < deadline {
        if ureq::get(url).call().is_ok() {
            return true;
        }
        thread::sleep(Duration::from_secs(2));
    }
    false
}

// ─── Tauri commands ──────────────────────────────────────────────────────────

/// Call the Python MOGCO sequencer and return its JSON response as a string.
/// The frontend can JSON.parse() the result to read `paths`, `slots`, etc.
#[tauri::command]
async fn generate_sequence(prompt: Option<String>) -> Result<String, String> {
    let client = reqwest::Client::new();
    let res = client
        .post("http://127.0.0.1:8000/api/sequence")
        .json(&serde_json::json!({ "vibe_prompt": prompt }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    res.text().await.map_err(|e| e.to_string())
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // ── WebView2 isolation (Windows) ─────────────────────────────────────────
    // Give the app its own WebView2 user-data folder instead of sharing Edge's
    // profile state. Without this, shared Edge state leaks in: first-run
    // experiences, "restore pages?" recovery tabs after a crash, and the
    // occasional Microsoft tab opening at launch. MUST be set before the
    // first webview is created — WebView2 reads it at environment init.
    #[cfg(target_os = "windows")]
    {
        let base = env::var("LOCALAPPDATA")
            .unwrap_or_else(|_| std::env::temp_dir().to_string_lossy().into_owned());
        let wv_data = PathBuf::from(base).join("FirstCut").join("WebView2");
        let _ = std::fs::create_dir_all(&wv_data);
        env::set_var("WEBVIEW2_USER_DATA_FOLDER", &wv_data);
    }

    let server_handle: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let handle_for_setup = server_handle.clone();
    let handle_for_exit  = server_handle.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![generate_sequence])
        .setup(move |app| {
            // Show a static loading page immediately while Python starts up.
            // Once the server is ready the background thread navigates to the real app.
            let webview_url = tauri::WebviewUrl::App("loading.html".into());

            // ── Create and show the window immediately (no waiting) ──────────
            // The loading screen in App.tsx polls the backend and shows a
            // spinner until the server is ready.
            let window = tauri::WebviewWindowBuilder::new(
                app,
                "main",
                webview_url,
            )
            .title("FirstCut")
            // Calm WebView2: no first-run experience, no default-browser-check
            // prompts, no Microsoft UI surfaces inside the app window. NOTE:
            // setting these args REPLACES Tauri's defaults, so Tauri's own
            // disable-features list is included verbatim.
            .additional_browser_args(
                "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --no-first-run --no-default-browser-check",
            )
            .inner_size(1400.0, 900.0)
            .min_inner_size(960.0, 640.0)
            .resizable(true)
            .decorations(true)
            .build()?;

            window.show()?;
            window.set_focus()?;

            // ── Spawn the backend in a background thread ─────────────────────
            let handle_clone   = handle_for_setup.clone();
            let app_handle     = app.handle().clone();

            thread::spawn(move || {
                let child: Option<Child> = if cfg!(debug_assertions) {
                    // ── Debug: venv python + server.py ───────────────────────
                    let project_root = find_project_root();
                    eprintln!("[tauri] [debug] Project root: {:?}", project_root);
                    start_python_server(&project_root)
                } else {
                    // ── Release: prefer PyInstaller sidecar, fall back to venv ─
                    eprintln!("[tauri] [release] Starting sidecar...");
                    let sidecar = start_sidecar(&app_handle);
                    if sidecar.is_some() {
                        sidecar
                    } else {
                        eprintln!("[tauri] Sidecar not found — falling back to venv Python.");
                        let project_root = find_project_root();
                        start_python_server(&project_root)
                    }
                };

                *handle_clone.lock().unwrap() = child;

                // Wait up to 90 min (first-run engine download can be large), then navigate.
                let ready = wait_for_server("http://127.0.0.1:8000/", 90 * 60);
                if ready {
                    eprintln!("[tauri] Server ready — navigating to app.");
                    if let Some(win) = app_handle.get_webview_window("main") {
                        let _ = win.navigate("http://127.0.0.1:8000".parse().unwrap());
                    }
                } else {
                    eprintln!("[tauri] WARNING: Server did not respond in 90 min.");
                }
            });

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Tauri application")
        .run(move |_app_handle, event| {
            if let RunEvent::Exit = event {
                // Kill the backend process when the window closes
                if let Ok(mut guard) = handle_for_exit.lock() {
                    if let Some(ref mut child) = *guard {
                        let _ = child.kill();
                    }
                }
            }
        });
}
