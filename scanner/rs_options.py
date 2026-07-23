"""RealSense D405 option helpers (presets, ranges, live patch)."""

from __future__ import annotations

from typing import Any

# D405 stereo module — no IR projector; RGB+depth share exposure (fallback when offline).
D405_FALLBACK_PRESETS: tuple[str, ...] = (
    "high_accuracy",
    "short_range",
    "default",
    "high_density",
    "medium_density",
    "mid_density",
    "low_density",
)
D405_FALLBACK_SUPPORTED: dict[str, bool] = {
    "visual_preset": True,
    "auto_exposure": True,
    "exposure": True,
    "gain": True,
    "laser": False,
    "emitter": False,
}


def _preset_table(rs: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for enum_name in ("l500_visual_preset", "rs400_visual_preset"):
        enum = getattr(rs, enum_name, None)
        if enum is None:
            continue
        for name in dir(enum):
            if name.startswith("_"):
                continue
            try:
                val = int(getattr(enum, name))
            except (TypeError, ValueError):
                continue
            key = name.lower()
            out[key] = val
    return out


def preset_name_for_value(rs: Any, value: float) -> str | None:
    iv = int(round(value))
    for name, val in _preset_table(rs).items():
        if val == iv:
            return name
    return str(iv)


def preset_value_for_name(rs: Any, name: str) -> int | None:
    table = _preset_table(rs)
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key in table:
        return table[key]
    for k, v in table.items():
        if k.replace("_", "") == key.replace("_", ""):
            return v
    return None


def list_preset_names(rs: Any, sensor: Any) -> list[str]:
    table = _preset_table(rs)
    if sensor is not None and sensor.supports(rs.option.visual_preset):
        try:
            rng = sensor.get_option_range(rs.option.visual_preset)
            lo, hi = int(rng.min), int(rng.max)
            names = sorted(name for name, val in table.items() if lo <= val <= hi)
            if names:
                return names
        except RuntimeError:
            pass
    return list(D405_FALLBACK_PRESETS)


def depth_option_support(rs: Any, depth_sensor: Any | None) -> dict[str, bool]:
    if depth_sensor is None:
        return dict(D405_FALLBACK_SUPPORTED)
    return {
        "visual_preset": depth_sensor.supports(rs.option.visual_preset),
        "auto_exposure": depth_sensor.supports(rs.option.enable_auto_exposure),
        "exposure": depth_sensor.supports(rs.option.exposure),
        "gain": depth_sensor.supports(rs.option.gain),
        "laser": depth_sensor.supports(rs.option.laser_power),
        "emitter": depth_sensor.supports(rs.option.emitter_enabled),
    }


def _option_range(sensor: Any, opt: Any) -> dict[str, float] | None:
    if not sensor.supports(opt):
        return None
    try:
        rng = sensor.get_option_range(opt)
        return {
            "min": float(rng.min),
            "max": float(rng.max),
            "step": float(rng.step),
            "default": float(rng.default),
        }
    except RuntimeError:
        return None


def _read_option(sensor: Any, opt: Any) -> float | None:
    if not sensor.supports(opt):
        return None
    try:
        return float(sensor.get_option(opt))
    except RuntimeError:
        return None


def _write_option(sensor: Any, opt: Any, value: float) -> None:
    if not sensor.supports(opt):
        raise RuntimeError(f"Option {opt} not supported")
    sensor.set_option(opt, float(value))


def read_depth_options(rs: Any, depth_sensor: Any) -> dict[str, Any]:
    support = depth_option_support(rs, depth_sensor)
    preset_val = _read_option(depth_sensor, rs.option.visual_preset)
    preset_name = preset_name_for_value(rs, preset_val) if preset_val is not None else None
    ae = _read_option(depth_sensor, rs.option.enable_auto_exposure)
    exposure = _read_option(depth_sensor, rs.option.exposure)
    gain = _read_option(depth_sensor, rs.option.gain)
    laser = _read_option(depth_sensor, rs.option.laser_power) if support["laser"] else None
    emitter = (
        _read_option(depth_sensor, rs.option.emitter_enabled) if support["emitter"] else None
    )

    return {
        "visual_preset": {
            "value": preset_name,
            "choices": list_preset_names(rs, depth_sensor),
            "supported": support["visual_preset"],
        },
        "auto_exposure": bool(ae) if ae is not None else None,
        "exposure_us": {
            "value": exposure,
            "range": _option_range(depth_sensor, rs.option.exposure),
            "editable": support["exposure"] and ae is not None and not bool(ae),
            "supported": support["exposure"],
        },
        "gain": {
            "value": gain,
            "range": _option_range(depth_sensor, rs.option.gain),
            "supported": support["gain"],
        },
        "laser_power": {
            "value": laser,
            "range": _option_range(depth_sensor, rs.option.laser_power)
            if support["laser"]
            else None,
            "supported": support["laser"],
        },
        "emitter_enabled": {
            "value": bool(emitter) if emitter is not None else None,
            "supported": support["emitter"],
        },
    }


def apply_depth_options(rs: Any, depth_sensor: Any, updates: dict[str, Any]) -> None:
    if "visual_preset" in updates:
        val = preset_value_for_name(rs, str(updates["visual_preset"]))
        if val is None:
            raise ValueError(f"Unknown visual preset: {updates['visual_preset']!r}")
        _write_option(depth_sensor, rs.option.visual_preset, val)

    if "auto_exposure" in updates:
        _write_option(
            depth_sensor,
            rs.option.enable_auto_exposure,
            1.0 if updates["auto_exposure"] else 0.0,
        )

    if "exposure_us" in updates:
        ae = _read_option(depth_sensor, rs.option.enable_auto_exposure)
        if ae is not None and bool(ae):
            raise RuntimeError("Disable auto exposure before setting exposure")
        _write_option(depth_sensor, rs.option.exposure, float(updates["exposure_us"]))

    if "gain" in updates:
        _write_option(depth_sensor, rs.option.gain, float(updates["gain"]))

    if "laser_power" in updates and depth_sensor.supports(rs.option.laser_power):
        _write_option(depth_sensor, rs.option.laser_power, float(updates["laser_power"]))

    if "emitter_enabled" in updates and depth_sensor.supports(rs.option.emitter_enabled):
        _write_option(
            depth_sensor,
            rs.option.emitter_enabled,
            1.0 if updates["emitter_enabled"] else 0.0,
        )


def read_camera_telemetry(rs: Any, depth_sensor: Any) -> dict[str, Any]:
    temps: dict[str, float | None] = {}
    for opt_name, key in (
        ("asic_temperature", "asic_c"),
        ("mcu_temperature", "mcu_c"),
        ("projector_temperature", "projector_c"),
        ("temperature", "module_c"),
    ):
        opt = getattr(rs.option, opt_name, None)
        if opt is None:
            temps[key] = None
            continue
        val = _read_option(depth_sensor, opt)
        temps[key] = round(val, 1) if val is not None else None

    fan_rpm: float | None = None
    for name in ("fan_speed",):
        opt = getattr(rs.option, name, None)
        if opt is not None and depth_sensor.supports(opt):
            fan_rpm = _read_option(depth_sensor, opt)
            break

    return {
        "temperatures_c": temps,
        "fan_rpm": int(fan_rpm) if fan_rpm is not None else None,
    }
