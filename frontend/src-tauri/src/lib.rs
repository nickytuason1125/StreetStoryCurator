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

/// Spawn the Python FastAPI server in dev mode and return the child handle.
fn start_python_server(project_root: &std::path::Path) -> Option<Child> {
    let venv_python = project_root.join("venv/Scripts/pythonw.exe");
    let server_py   = project_root.join("server.py");

    if !server_py.exists() {
        eprintln!("[tauri] ERROR: server.py not found at {:?}", server_py);
        return None;
    }

    let python_exe = if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("pythonw")
    };

    eprintln!("[tauri] Starting: {:?} {:?}", python_exe, server_py);

    let mut cmd = Command::new(&python_exe);
    cmd.arg(&server_py).current_dir(project_root);
    // Suppress the console window that would otherwise flash on Windows.
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    cmd.spawn()
        .map_err(|e| eprintln!("[tauri] Failed to spawn server: {}", e))
        .ok()
}

// ─── Release-mode sidecar launcher ──────────────────────────────────────────

/// Spawn the PyInstaller sidecar bundle in release mode.
/// The bundle dir lives at:
///   <resource_dir>/binaries/curator-api-<triple>/
/// and the executable is:
///   curator-api-<triple>/curator-api[.exe]
///
/// We set:
///   CWD          → the bundle dir  (so models/ and frontend/dist/ resolve)
///   CURATOR_DATA_DIR → user's AppData/StreetStoryCurator (writable cache)
fn start_sidecar(app: &tauri::AppHandle) -> Option<Child> {
    let resource_dir = app.path().resource_dir()
        .map_err(|e| eprintln!("[tauri] resource_dir error: {}", e))
        .ok()?;

    let triple      = sidecar_triple();
    let ext         = exe_ext();
    let bundle_dir  = resource_dir.join("binaries").join(format!("curator-api-{}", triple));
    let exe_path    = bundle_dir.join(format!("curator-api{}", ext));

    eprintln!("[tauri] Sidecar bundle dir: {:?}", bundle_dir);
    eprintln!("[tauri] Sidecar exe:        {:?}", exe_path);

    if !exe_path.exists() {
        eprintln!("[tauri] ERROR: sidecar exe not found — did you run build-backend.bat?");
        return None;
    }

    // Resolve the writable data directory (AppData\Roaming\StreetStoryCurator on Windows)
    let data_dir = app.path().data_dir()
        .map(|d| d.join("StreetStoryCurator"))
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
fn wait_for_server(url: &str, retries: u32) -> bool {
    for _ in 0..retries {
        if ureq::get(url).call().is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(400));
    }
    false
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_handle: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    let handle_for_setup = server_handle.clone();
    let handle_for_exit  = server_handle.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
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
            .title("Street Story Curator")
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

                // Wait up to 60 s, then navigate the window to the real React app.
                let ready = wait_for_server("http://127.0.0.1:8000/", 150);
                if ready {
                    eprintln!("[tauri] Server ready — navigating to app.");
                    if let Some(win) = app_handle.get_webview_window("main") {
                        let _ = win.navigate("http://127.0.0.1:8000".parse().unwrap());
                    }
                } else {
                    eprintln!("[tauri] WARNING: Server did not respond in 60 s.");
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
