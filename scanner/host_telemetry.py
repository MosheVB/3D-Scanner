"""Host CPU/GPU telemetry without extra dependencies."""

from __future__ import annotations

import platform
import shutil
import subprocess
import threading
import time
from typing import Any

_CACHE_TTL_S = 12.0
_WARMING: dict[str, Any] = {
    "cpu_c": None,
    "gpu_c": None,
    "fan_rpm": None,
    "status": "warming_up",
    "notes": [],
}
_cache: dict[str, Any] = dict(_WARMING)
_cache_lock = threading.Lock()
_refresh_lock = threading.Lock()
_refresh_started = False


def _fmt_temp(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 1)


def _windows_ohm_sensors() -> tuple[float | None, float | None, int | None]:
    """OpenHardwareMonitor / LibreHardwareMonitor WMI (fast no-op if not installed)."""
    ps = (
        "Get-CimInstance -Namespace root/OpenHardwareMonitor -ClassName Sensor "
        "-ErrorAction SilentlyContinue | "
        "Where-Object { $_.SensorType -eq 'Temperature' -or $_.SensorType -eq 'Fan' } | "
        "Select-Object Name, SensorType, Value"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            return None, None, None
        cpu_temps: list[float] = []
        gpu_temps: list[float] = []
        fan_rpms: list[int] = []
        for line in (proc.stdout or "").splitlines():
            parts = [p.strip() for p in line.split() if p.strip()]
            if len(parts) < 3:
                continue
            try:
                val = float(parts[-1])
            except ValueError:
                continue
            name = " ".join(parts[:-2]).lower()
            kind = parts[-2].lower() if len(parts) >= 3 else ""
            if "fan" in kind or "fan" in name:
                if val > 0:
                    fan_rpms.append(int(val))
            elif "temperature" in kind or "temp" in name or "°c" in name:
                if not (5 <= val <= 120):
                    continue
                if "gpu" in name or "graphics" in name:
                    gpu_temps.append(val)
                elif "cpu" in name or "core" in name or "package" in name:
                    cpu_temps.append(val)
        cpu = max(cpu_temps) if cpu_temps else None
        gpu = max(gpu_temps) if gpu_temps else None
        fan = max(fan_rpms) if fan_rpms else None
        return cpu, gpu, fan
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None, None, None


def _windows_cpu_temp_c() -> float | None:
    """ACPI thermal zone via PowerShell CIM (decikelvin → °C)."""
    ps = (
        "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
        "-ErrorAction SilentlyContinue | "
        "Select-Object -First 1 -ExpandProperty CurrentTemperature"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            check=False,
        )
        raw = (proc.stdout or "").strip()
        if not raw or not raw.lstrip("-").isdigit():
            return None
        val = int(raw)
        if val < 100:
            return None
        temp = (val / 10.0) - 273.15
        if temp < 5 or temp > 120:
            return None
        return temp
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _linux_cpu_temp_c() -> float | None:
    import glob

    for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        try:
            raw = open(path, encoding="utf-8").read().strip()
            val = int(raw)
            if val > 1000:
                val /= 1000.0
            if 10 <= val <= 110:
                return float(val)
        except OSError:
            continue
    return None


def _nvidia_gpu_temp_c() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return None
        return float(line[0].strip())
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _collect_host_telemetry_sync() -> dict[str, Any]:
    """Blocking collection — run off the HTTP thread."""
    cpu: float | None = None
    gpu: float | None = None
    fan_rpm: int | None = None
    notes: list[str] = []

    system = platform.system()
    if system == "Windows":
        ohm_cpu, ohm_gpu, ohm_fan = _windows_ohm_sensors()
        cpu = ohm_cpu
        gpu = ohm_gpu
        fan_rpm = ohm_fan
        if cpu is None:
            cpu = _windows_cpu_temp_c()
        if cpu is None and gpu is None and fan_rpm is None:
            notes.append(
                "CPU/GPU/fan n/a — run LibreHardwareMonitor (WMI) for readings on Windows."
            )
    elif system == "Linux":
        cpu = _linux_cpu_temp_c()
    elif system == "Darwin":
        try:
            proc = subprocess.run(
                ["sysctl", "-n", "machdep.xcpm.cpu_thermal_level"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=2,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip().isdigit():
                cpu = float(proc.stdout.strip())
        except OSError:
            pass

    if gpu is None:
        gpu = _nvidia_gpu_temp_c()

    return {
        "cpu_c": _fmt_temp(cpu),
        "gpu_c": _fmt_temp(gpu),
        "fan_rpm": fan_rpm,
        "status": "ok",
        "notes": notes,
    }


def _refresh_loop() -> None:
    while True:
        try:
            data = _collect_host_telemetry_sync()
            with _cache_lock:
                _cache.clear()
                _cache.update(data)
        except Exception:
            with _cache_lock:
                _cache["status"] = "error"
        time.sleep(_CACHE_TTL_S)


def _ensure_refresh_started() -> None:
    global _refresh_started
    with _refresh_lock:
        if _refresh_started:
            return
        _refresh_started = True
        t = threading.Thread(target=_refresh_loop, name="host-telemetry", daemon=True)
        t.start()


def collect_host_telemetry() -> dict[str, Any]:
    """Return cached host telemetry immediately; refresh runs in background."""
    _ensure_refresh_started()
    with _cache_lock:
        return dict(_cache)
