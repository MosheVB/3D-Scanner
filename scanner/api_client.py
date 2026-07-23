"""HTTP client for the scanner web API (CLI / agents — no direct camera access)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

DEFAULT_API_URL = "http://127.0.0.1:8765"

SERVER_DOWN_MSG = (
    "Scanner web server is not running.\n"
    "Start it in another terminal:\n"
    "  python -m scanner web"
)


class ApiError(RuntimeError):
    pass


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _request(
    method: str,
    path: str,
    *,
    base_url: str = DEFAULT_API_URL,
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        _url(base_url, path),
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(detail)
            msg = payload.get("error", detail)
        except json.JSONDecodeError:
            msg = detail or str(exc)
        raise ApiError(f"API {exc.code}: {msg}") from exc
    except urllib.error.URLError as exc:
        raise ApiError(SERVER_DOWN_MSG) from exc


def ping(*, base_url: str = DEFAULT_API_URL, timeout: float = 3.0) -> bool:
    try:
        _request("GET", "/api/status", base_url=base_url, timeout=timeout)
        return True
    except ApiError:
        return False


def get_status(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("GET", "/api/status", base_url=base_url)


def get_mask_status(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("GET", "/api/mask/status", base_url=base_url)


def get_camera_options(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("GET", "/api/camera/options", base_url=base_url)


def patch_camera_options(
    updates: dict[str, Any],
    *,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    return _request("PATCH", "/api/camera/options", body=updates, base_url=base_url)


def connect_camera(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("POST", "/api/camera/connect", base_url=base_url)


def disconnect_camera(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("POST", "/api/camera/disconnect", base_url=base_url)


def get_telemetry(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("GET", "/api/telemetry", base_url=base_url)


def fetch_jpeg(path: str, *, base_url: str = DEFAULT_API_URL, timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(_url(base_url, path), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise ApiError(SERVER_DOWN_MSG) from exc


def turntable_connect(
    *,
    ble_address: str | None = None,
    scan_timeout: float = 10.0,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    body: dict[str, Any] = {"scan_timeout": scan_timeout}
    if ble_address:
        body["ble_address"] = ble_address
    return _request("POST", "/api/turntable/connect", body=body, base_url=base_url)


def turntable_disconnect(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("POST", "/api/turntable/disconnect", base_url=base_url)


def turntable_rotate(
    degrees: float,
    *,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/turntable/rotate",
        body={"degrees": degrees},
        base_url=base_url,
    )


def turntable_tilt(
    degrees: float,
    *,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/api/turntable/tilt",
        body={"degrees": degrees},
        base_url=base_url,
    )


def turntable_home(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("POST", "/api/turntable/home", base_url=base_url)


def turntable_stop(*, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("POST", "/api/turntable/stop", base_url=base_url)


def mask_click(
    u: int,
    v: int,
    *,
    mode: str = "include",
    use_haiku: bool | None = None,
    base_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    body: dict[str, Any] = {"u": u, "v": v, "mode": mode}
    if use_haiku is not None:
        body["use_haiku"] = use_haiku
    return _request("POST", "/api/mask/click", body=body, base_url=base_url)


def start_turntable_scan(
    *,
    name: str = "scan",
    output_dir: Path | str | None = None,
    auto_process: bool = False,
    use_calibration: bool = True,
    calibration_dir: Path | str | None = None,
    tilt_levels: list[float] | None = None,
    rotate_step_deg: float | None = None,
    rotate_wait_s: float | None = None,
    tilt_wait_s: float | None = None,
    ble_address: str | None = None,
    scan_timeout: float = 10.0,
    base_url: str = DEFAULT_API_URL,
) -> str:
    body: dict[str, Any] = {
        "name": name,
        "auto_process": auto_process,
        "use_calibration": use_calibration,
        "scan_timeout": scan_timeout,
    }
    if output_dir is not None:
        body["output_dir"] = str(output_dir)
    if calibration_dir is not None:
        body["calibration_dir"] = str(calibration_dir)
    if tilt_levels is not None:
        body["tilt_levels"] = tilt_levels
    if rotate_step_deg is not None:
        body["rotate_step_deg"] = rotate_step_deg
    if rotate_wait_s is not None:
        body["rotate_wait_s"] = rotate_wait_s
    if tilt_wait_s is not None:
        body["tilt_wait_s"] = tilt_wait_s
    if ble_address is not None:
        body["ble_address"] = ble_address
    payload = _request("POST", "/api/scan/turntable", body=body, base_url=base_url)
    job_id = payload.get("job_id")
    if not job_id:
        raise ApiError("API did not return job_id")
    return str(job_id)


def get_job(job_id: str, *, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("GET", f"/api/jobs/{job_id}", base_url=base_url)


def list_jobs(*, base_url: str = DEFAULT_API_URL) -> list[dict[str, Any]]:
    payload = _request("GET", "/api/jobs", base_url=base_url)
    return list(payload.get("jobs") or [])


def cancel_job(job_id: str, *, base_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    return _request("POST", f"/api/jobs/{job_id}/cancel", base_url=base_url)


def poll_job(
    job_id: str,
    *,
    base_url: str = DEFAULT_API_URL,
    interval_s: float = 1.0,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Block until job reaches a terminal state."""
    while True:
        job = get_job(job_id, base_url=base_url)
        if on_update is not None:
            on_update(job)
        status = job.get("status")
        if status in ("completed", "failed", "cancelled"):
            return job
        time.sleep(interval_s)


def run_turntable_scan_via_api(
    *,
    name: str = "scan",
    output_dir: Path | None = None,
    auto_process: bool = False,
    use_calibration: bool = True,
    calibration_dir: Path | None = None,
    tilt_levels: list[float] | None = None,
    rotate_step_deg: float | None = None,
    rotate_wait_s: float | None = None,
    tilt_wait_s: float | None = None,
    ble_address: str | None = None,
    scan_timeout: float = 10.0,
    base_url: str = DEFAULT_API_URL,
) -> int:
    """CLI entry: submit turntable scan to web server and wait for completion."""
    try:
        job_id = start_turntable_scan(
            name=name,
            output_dir=output_dir,
            auto_process=auto_process,
            use_calibration=use_calibration,
            calibration_dir=calibration_dir,
            tilt_levels=tilt_levels,
            rotate_step_deg=rotate_step_deg,
            rotate_wait_s=rotate_wait_s,
            tilt_wait_s=tilt_wait_s,
            ble_address=ble_address,
            scan_timeout=scan_timeout,
            base_url=base_url,
        )
    except ApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Turntable scan job started: {job_id}")

    def _print_progress(job: dict[str, Any]) -> None:
        prog = job.get("progress") or {}
        cur, total = prog.get("current", 0), prog.get("total", 0)
        msg = job.get("message") or job.get("status", "")
        if total:
            print(f"  [{cur}/{total}] {msg}", flush=True)
        else:
            print(f"  {msg}", flush=True)

    try:
        final = poll_job(job_id, base_url=base_url, on_update=_print_progress)
    except ApiError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    status = final.get("status")
    if status == "completed":
        path = final.get("result_path")
        print(f"Scan complete: {path}")
        if auto_process:
            print("(auto-process requested — check server logs / session folder)")
        elif path:
            print(f"Run: python -m scanner process {path}")
        return 0
    if status == "cancelled":
        print("Scan cancelled.", file=sys.stderr)
        return 1
    err = final.get("error") or "unknown error"
    print(f"Scan failed: {err}", file=sys.stderr)
    return 1
