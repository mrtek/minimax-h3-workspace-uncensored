# Changelog

## Sprint 0 (P0 Hotfixes)

- **S0-01**: Add missing `import tempfile` in `app/server.py` for montage endpoint and sanitize montage audio filename.
- **S0-02**: Remove dangerous global process killing (`taskkill` and PowerShell `Stop-Process`) from `cleanup_stale_environment()` in `app/main.py`.
- **S0-03**: Resilient error handling and quarantine for corrupted `config.json` and `library.json`, atomic config writes, and startup failure explanation dialog in `app/main.py`.
- **S0-04**: Fix HTTP 416 download resume in `app/civitai_client.py` to prevent file truncation to 0 bytes and erroneous deletion of complete files.
- **S0-05**: Implement `same_origin_only` middleware, WebSocket Origin verification, LoRA path sanitization, and API token leakage guard in `app/server.py` and `app/civitai_client.py`.
- **S0-06**: Add minimal automated test suite under `tests/` covering config recovery, download 416 resume, and HTTP guards.
