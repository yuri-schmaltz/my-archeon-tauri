# Sidecar Binaries

This folder stores Tauri `externalBin` artifacts built per OS.

Expected names used by Tauri:
- Linux: `python-backend-x86_64-unknown-linux-gnu`
- Windows: `python-backend-x86_64-pc-windows-msvc.exe`

Build from source:

```bash
python scripts/build_sidecar.py
```

The build script also copies the generated binary to the Tauri target-triple filename.
