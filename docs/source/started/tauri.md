# Tauri Desktop + Python Sidecar

This repository now includes a base Tauri desktop scaffold integrated with a Python sidecar.

## Layout

- `src-tauri/`: Rust/Tauri host app.
- `binaries/`: sidecar executable expected by `externalBin`.
- `hy3dgen/sidecar/main.py`: JSON sidecar entrypoint.
- `hy3dgen/agentic/`: Planner/Executor/Validator pipeline over `plan.json`.

## Build Sidecar

```bash
python scripts/build_sidecar.py
```

This produces target-triple sidecars:

- Linux: `binaries/python-backend-x86_64-unknown-linux-gnu`
- Windows: `binaries/python-backend-x86_64-pc-windows-msvc.exe`

## Sidecar Protocol

The sidecar accepts JSON requests:

```json
{"id":"1","method":"health","params":{}}
```

Methods:

- `health`
- `detect_runtime`
- `generate_plan`
- `execute_plan`

## Frontend Call

Use `@tauri-apps/plugin-shell` and `Command.sidecar(\"python-backend\", ...)`.

Reference implementation:

- `src/sidecarClient.ts`
- `src/sidecarClient.js`
