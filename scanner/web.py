"""Scanner web server — owns RealSense D405, live view + REST API."""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from scanner.camera_service import CameraService, _BOUNDARY
from scanner.platform_io import configure_stdio_utf8
from scanner.config import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_SCANS_DIR,
    TURNTABLE_ROTATE_STEP_DEG,
    TURNTABLE_ROTATE_WAIT_S,
    TURNTABLE_TILT_LEVELS,
    TURNTABLE_TILT_WAIT_S,
)
from scanner.turntable_capture import TurntableScanConfig

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765

_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>3D Scanner — live</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem;
           background: #111; color: #eee; max-width: 1400px; margin-inline: auto; }
    h1 { font-size: 1.25rem; font-weight: 600; margin: 0 0 0.75rem; }
    .status-bar { background: #1a1a1f; border: 1px solid #333; border-radius: 8px;
                  padding: 0.6rem 0.85rem; margin-bottom: 1rem; font-size: 0.88rem; }
    .status-bar span { margin-right: 1rem; }
    .ok { color: #6f6; } .warn { color: #fc6; } .err { color: #f66; }
    .panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
              gap: 0.75rem; margin-bottom: 1rem; }
    .panel { background: #1a1a1f; border: 1px solid #333; border-radius: 8px; padding: 0.5rem; }
    .panel h2 { font-size: 0.8rem; margin: 0 0 0.35rem; color: #aaa; font-weight: 500; }
    .panel img { width: 100%; background: #000; border-radius: 4px; display: block; aspect-ratio: 4/3; object-fit: contain; }
    .click-panel { position: relative; }
    .click-panel img { pointer-events: none; display: block; width: 100%; }
    .click-overlay { position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index: 2; cursor: crosshair; }
    .click-panel.mode-remove { cursor: crosshair; }
    .mask-help { font-size: 0.78rem; color: #888; margin: 0.35rem 0 0; line-height: 1.35; }
    .mode-toggle button.active { background: #2a6a3a; }
    .mode-toggle button.active.remove { background: #7a2a2a; }
    section { background: #1a1a1f; border: 1px solid #333; border-radius: 8px;
              padding: 0.75rem; margin-bottom: 0.75rem; }
    section h3 { font-size: 0.85rem; margin: 0 0 0.5rem; color: #ccc; font-weight: 600; }
    .row { display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; margin-bottom: 0.4rem; }
    button, .btn { background: #2a4a7a; color: #fff; border: none; border-radius: 6px;
                   padding: 0.45rem 0.75rem; font-size: 0.85rem; cursor: pointer; }
    button:hover { background: #356099; }
    button:disabled { opacity: 0.45; cursor: not-allowed; }
    button.danger { background: #7a2a2a; } button.danger:hover { background: #993535; }
    button.secondary { background: #333; } button.secondary:hover { background: #444; }
    input, select { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px;
                    padding: 0.35rem 0.5rem; font-size: 0.85rem; }
    label { font-size: 0.82rem; color: #aaa; }
    #job-detail { font-size: 0.85rem; color: #9cf; min-height: 1.2em; }
    #toast { position: fixed; bottom: 1rem; right: 1rem; background: #333; color: #fff;
             padding: 0.5rem 0.85rem; border-radius: 6px; font-size: 0.85rem;
             opacity: 0; transition: opacity 0.2s; pointer-events: none; z-index: 99; }
    #toast.show { opacity: 1; }
    a { color: #9cf; }
    .telemetry-card { background: #1a1a1f; border: 1px solid #333; border-radius: 8px;
                      padding: 0.5rem 0.85rem; margin-bottom: 0.75rem; font-size: 0.82rem;
                      display: flex; flex-wrap: wrap; gap: 0.75rem 1.25rem; color: #bbb; }
    .telemetry-card span { white-space: nowrap; }
    .telemetry-card .val { color: #9cf; font-weight: 500; }
    details > summary { cursor: pointer; user-select: none; margin-bottom: 0.5rem; }
    .settings-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                     gap: 0.5rem 0.75rem; margin-top: 0.5rem; }
    .settings-grid label { display: flex; flex-direction: column; gap: 0.2rem; }
    .hint { font-size: 0.75rem; color: #777; margin: 0.35rem 0 0; }
    .haiku-status { font-size: 0.82rem; color: #b8d4ff; margin: 0.25rem 0 0.5rem; padding: 0.35rem 0.5rem; border-radius: 4px; background: #1e2430; }
    .haiku-status.flash { background: #2a3d28; color: #cfe9b8; transition: background 0.3s; }
  </style>
</head>
<body>
  <h1>3D Scanner — live viewer</h1>
  <div class="status-bar" id="status-bar">Loading…</div>
  <div class="telemetry-card" id="telemetry">Telemetry loading…</div>
  <div id="job-detail"></div>

  <div class="panels">
    <div class="panel">
      <h2>RGB — live voxel + mask clicks</h2>
      <div class="click-panel" id="rgb-click-panel">
        <img id="img-rgb" src="/stream/rgb" alt="RGB">
        <canvas id="rgb-click-overlay" class="click-overlay"></canvas>
      </div>
      <p class="mask-help" id="mask-help">
        <strong>Add (include)</strong> — green numbered markers seed the object; a yellow
        <strong>adaptive contour</strong> (click hull + depth/edges) outlines shape until the 3D cube locks in.
        Optional <strong>AI shape (Haiku)</strong> uses vision when an API key is configured.
        <strong>Remove (exclude)</strong> — red X — subtracts table/background and
        <strong>tightens</strong> the cube on that side. Cyan wireframe tracks during scan.
        <strong>Clear</strong> resets all clicks.
      </p>
      <div class="row mode-toggle">
        <button type="button" id="btn-mode-include" class="active" onclick="setMaskMode('include')">Add (include)</button>
        <button type="button" id="btn-mode-exclude" class="remove" onclick="setMaskMode('exclude')">Remove (exclude)</button>
      </div>
      <div class="row" id="haiku-toggle-row" style="display:none">
        <label><input id="opt-use-haiku" type="checkbox"> AI shape (Haiku)</label>
        <button type="button" class="secondary" id="btn-haiku-rerun" onclick="rerunHaiku()">Re-run AI shape</button>
      </div>
      <div class="row">
        <button class="secondary" onclick="clearMask()">Clear clicks</button>
        <span id="mask-status" style="font-size:0.82rem;color:#9cf"></span>
        <div id="haiku-status-line" class="haiku-status">Haiku: checking…</div>
      </div>
    </div>
    <div class="panel"><h2>Depth</h2><img id="img-depth" src="/stream/depth" alt="Depth"></div>
    <div class="panel"><h2>Voxel 3D (side)</h2><img id="img-voxel" src="/stream/voxel" alt="Voxel"></div>
  </div>

  <section>
    <h3>Live voxel on RGB (auto, no clicks)</h3>
    <p class="hint">Auto-detects object at frame center (no clicks). RGB-D odometry tracks
      rotation or handheld motion. Cyan = voxel, red = depth hole, orange = live depth.
      Rotate slowly or move the camera to fill gaps.</p>
    <div class="row">
      <label><input id="opt-live-voxel" type="checkbox" checked> Overlay on RGB stream</label>
      <button type="button" class="secondary" id="btn-voxel-reset" onclick="resetVoxel()">Reset voxels</button>
    </div>
  </section>

  <section>
    <h3>Camera</h3>
    <div class="row">
      <button id="btn-cam-connect" onclick="apiPost('/api/camera/connect').then(loadCamOpts)">Connect</button>
      <button id="btn-cam-disconnect" class="secondary" onclick="apiPost('/api/camera/disconnect')">Disconnect</button>
    </div>
    <details id="cam-settings-details">
      <summary>Camera settings (D405 stereo module)</summary>
      <p class="hint">D405 stereo module: RGB and depth share exposure/gain. No IR projector on D405.
        Connect camera to read live ranges. Changes apply live; session-only memory.</p>
      <div class="settings-grid" id="cam-settings-grid">
        <label>Visual preset
          <select id="opt-preset"></select>
        </label>
        <label>Auto exposure
          <select id="opt-ae"><option value="true">On</option><option value="false">Off</option></select>
        </label>
        <label>Exposure µs
          <input id="opt-exposure" type="number" min="1" step="100">
        </label>
        <label>Gain
          <input id="opt-gain" type="number" min="16" step="1">
        </label>
        <label id="label-opt-laser" style="display:none">Laser power
          <input id="opt-laser" type="number" min="0" step="1">
        </label>
        <label id="label-opt-emitter" style="display:none">Emitter
          <select id="opt-emitter"><option value="true">On</option><option value="false">Off</option></select>
        </label>
      </div>
      <p class="hint" style="margin-top:0.6rem">Depth post-processing filters</p>
      <div class="settings-grid">
        <label><input id="opt-spatial" type="checkbox" checked> Spatial filter</label>
        <label>Spatial magnitude <input id="opt-spatial-mag" type="number" min="1" max="5" value="2"></label>
        <label><input id="opt-temporal" type="checkbox" checked> Temporal filter</label>
        <label><input id="opt-hole" type="checkbox" checked> Hole filling</label>
        <label><input id="opt-decimate" type="checkbox"> Decimation</label>
        <label>Decimation mag <input id="opt-decimate-mag" type="number" min="1" max="8" value="1"></label>
      </div>
      <div class="row" style="margin-top:0.5rem">
        <button id="btn-cam-apply" onclick="applyCamOpts()">Apply settings</button>
        <button id="btn-cam-refresh" class="secondary" onclick="loadCamOpts()">Refresh</button>
      </div>
    </details>
  </section>

  <section>
    <h3>Turntable</h3>
    <div class="row">
      <button id="btn-tt-connect" onclick="apiPost('/api/turntable/connect')">Connect BLE</button>
      <button id="btn-tt-disconnect" class="secondary" onclick="apiPost('/api/turntable/disconnect')">Disconnect</button>
      <button id="btn-tt-home" onclick="apiPost('/api/turntable/home')">Home</button>
      <button id="btn-tt-stop" class="danger" onclick="apiPost('/api/turntable/stop')">E-stop</button>
    </div>
    <div class="row">
      <button id="btn-tt-rotate-minus" onclick="apiPost('/api/turntable/rotate', {degrees: -15})">Rotate −15°</button>
      <button id="btn-tt-rotate-plus" onclick="apiPost('/api/turntable/rotate', {degrees: 15})">Rotate +15°</button>
      <button id="btn-tt-rotate-m2" onclick="apiPost('/api/turntable/rotate', {degrees: -2})">−2°</button>
      <button id="btn-tt-rotate-m5" onclick="apiPost('/api/turntable/rotate', {degrees: -5})">−5°</button>
      <button id="btn-tt-rotate-p2" onclick="apiPost('/api/turntable/rotate', {degrees: 2})">+2°</button>
      <button id="btn-tt-rotate-p5" onclick="apiPost('/api/turntable/rotate', {degrees: 5})">+5°</button>
      <button id="btn-tt-tilt-minus" onclick="apiPost('/api/turntable/tilt', {delta: -15})">Tilt −15°</button>
      <button id="btn-tt-tilt-plus" onclick="apiPost('/api/turntable/tilt', {delta: 15})">Tilt +15°</button>
    </div>
  </section>

  <section>
    <h3>Turntable scan</h3>
    <div class="row">
      <label>Name <input id="scan-name" value="scan" size="12"></label>
      <label>Rotate step° <input id="scan-rotate-step" type="number" value="15" min="5" max="90" style="width:4rem"></label>
      <label>Tilt levels <input id="scan-tilt-levels" value="-30,-15,0,15,30" size="18"></label>
      <label><input id="scan-auto-process" type="checkbox"> Auto-process</label>
    </div>
    <div class="row">
      <button id="btn-start-scan" onclick="startScan()">Start scan</button>
      <button id="btn-cancel-scan" class="danger" disabled onclick="cancelScan()">Cancel job</button>
    </div>
  </section>

  <div id="toast"></div>

  <script>
    let activeJobId = null;
    const MAX_MASK_CLICKS = 16;
    let maskClicks = [];
    let maskExcludeClicks = [];
    let maskClickMode = 'include';
    let maskClickBusy = false;
    let camResolution = [640, 480];
    let useHaikuShape = false;
    let haikuToggleBusy = false;
    let liveVoxelToggleBusy = false;

    async function resetVoxel() {
      await apiPost('/api/voxel/reset', {});
      toast('Voxel model reset');
    }

    document.getElementById('opt-live-voxel').addEventListener('change', async (ev) => {
      const enabled = ev.target.checked;
      liveVoxelToggleBusy = true;
      try {
        await apiPost('/api/voxel/live', { enabled });
        toast(enabled ? 'Live voxel overlay on' : 'Live voxel overlay off');
      } catch (e) {
        ev.target.checked = !enabled;
        toast(e.message || 'Live voxel toggle failed', true);
      } finally {
        liveVoxelToggleBusy = false;
      }
    });

    function setMaskMode(mode) {
      maskClickMode = mode;
      const panel = document.getElementById('rgb-click-panel');
      const btnInc = document.getElementById('btn-mode-include');
      const btnExc = document.getElementById('btn-mode-exclude');
      btnInc.classList.toggle('active', mode === 'include');
      btnExc.classList.toggle('active', mode === 'exclude');
      panel.classList.toggle('mode-remove', mode === 'exclude');
    }

    async function postMaskClicks() {
      const data = await apiPost('/api/mask/clicks', {
        include: maskClicks,
        exclude: maskExcludeClicks,
        use_haiku: useHaikuShape,
      }, { quiet: true });
      const m = data && data.mask;
      if (m && m.needs_camera_depth) {
        toast('Camera not streaming — click Connect, then click again on RGB', true);
      } else if (m && m.click_count) {
        toast(`Mask: ${m.click_count} include` + (m.exclude_count ? `, ${m.exclude_count} exclude` : ''));
      }
      return data;
    }

    async function rerunHaiku() {
      await apiPost('/api/mask/assist-haiku', {});
    }

    document.getElementById('opt-use-haiku').addEventListener('change', async (ev) => {
      const enabled = ev.target.checked;
      useHaikuShape = enabled;
      haikuToggleBusy = true;
      try {
        await apiPost('/api/mask/use-haiku', { enabled }, { quiet: true });
        toast(enabled ? 'AI shape (Haiku) enabled' : 'AI shape (Haiku) off');
        if (enabled) await rerunHaiku();
      } catch (e) {
        ev.target.checked = !enabled;
        useHaikuShape = !enabled;
        toast(e.message || 'Haiku toggle failed', true);
      } finally {
        haikuToggleBusy = false;
      }
    });

    function toast(msg, isErr) {
      const el = document.getElementById('toast');
      el.textContent = msg;
      el.style.background = isErr ? '#633' : '#333';
      el.classList.add('show');
      setTimeout(() => el.classList.remove('show'), 3500);
    }

    async function apiPatch(path, body) {
      try {
        const r = await fetch(path, {
          method: 'PATCH',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || r.statusText);
        toast('Settings applied');
        return data;
      } catch (e) {
        toast(e.message, true);
        throw e;
      }
    }

    function fmtTemp(v) { return (v === null || v === undefined) ? 'n/a' : v + '°C'; }

    async function refreshTelemetry() {
      try {
        const t = await fetch('/api/telemetry').then(r => r.json());
        const h = t.host || {};
        const c = t.camera || {};
        const temps = c.temperatures_c || {};
        const fan = (h.fan_rpm != null) ? (h.fan_rpm + ' RPM') : 'n/a';
        let html =
          `<span>CPU <span class="val">${fmtTemp(h.cpu_c)}</span></span>` +
          `<span>GPU <span class="val">${fmtTemp(h.gpu_c)}</span></span>` +
          `<span>Host fan <span class="val">${fan}</span></span>`;
        if (c.connected) {
          html +=
            `<span>Cam ASIC <span class="val">${fmtTemp(temps.asic_c)}</span></span>` +
            `<span>Cam MCU <span class="val">${fmtTemp(temps.mcu_c)}</span></span>`;
        }
        const notes = (h.notes || []).join(' ');
        if (notes) html += `<span style="color:#888">${notes}</span>`;
        document.getElementById('telemetry').innerHTML = html;
      } catch (_) {
        document.getElementById('telemetry').textContent = 'Telemetry unavailable';
      }
    }

    const CAM_SETTING_IDS = [
      'opt-preset', 'opt-ae', 'opt-exposure', 'opt-gain', 'opt-laser', 'opt-emitter',
      'opt-spatial', 'opt-spatial-mag', 'opt-temporal', 'opt-hole', 'opt-decimate',
      'opt-decimate-mag', 'btn-cam-apply', 'btn-cam-refresh',
    ];

    function setCamSettingsDisabled(disabled) {
      for (const id of CAM_SETTING_IDS) setDisabled(id, disabled);
    }

    async function loadCamOpts() {
      try {
        const o = await fetch('/api/camera/options').then(r => r.json());
        const d = o.depth || {};
        const f = o.filters || {};
        const sup = o.supported || {};
        const preset = document.getElementById('opt-preset');
        const choices = (d.visual_preset && d.visual_preset.choices) || [];
        const cur = (d.visual_preset && d.visual_preset.value) || 'short_range';
        preset.innerHTML = choices.length
          ? choices.map(c => `<option value="${c}"${c === cur ? ' selected' : ''}>${c}</option>`).join('')
          : `<option value="${cur}">${cur}</option>`;
        document.getElementById('opt-ae').value = d.auto_exposure ? 'true' : 'false';
        const exp = d.exposure_us || {};
        const expIn = document.getElementById('opt-exposure');
        expIn.value = exp.value != null ? Math.round(exp.value) : '';
        if (exp.range) {
          expIn.min = exp.range.min;
          expIn.max = exp.range.max;
          expIn.step = exp.range.step || 100;
        }
        expIn.disabled = !!d.auto_exposure || !sup.exposure;
        const gain = d.gain || {};
        const gainIn = document.getElementById('opt-gain');
        gainIn.value = gain.value != null ? Math.round(gain.value) : 16;
        if (gain.range) {
          gainIn.min = gain.range.min;
          gainIn.max = gain.range.max;
          gainIn.step = gain.range.step || 1;
        }
        const laser = d.laser_power || {};
        const laserLbl = document.getElementById('label-opt-laser');
        const showLaser = !!(sup.laser || laser.supported);
        laserLbl.style.display = showLaser ? '' : 'none';
        const laserIn = document.getElementById('opt-laser');
        laserIn.value = laser.value != null ? Math.round(laser.value) : '';
        laserIn.disabled = !showLaser;
        const emit = d.emitter_enabled || {};
        const emitLbl = document.getElementById('label-opt-emitter');
        const showEmitter = !!(sup.emitter || emit.supported);
        emitLbl.style.display = showEmitter ? '' : 'none';
        const emitSel = document.getElementById('opt-emitter');
        emitSel.disabled = !showEmitter;
        if (emit.value != null) emitSel.value = emit.value ? 'true' : 'false';
        document.getElementById('opt-spatial').checked = !!f.spatial_enabled;
        document.getElementById('opt-spatial-mag').value = f.spatial_magnitude || 2;
        document.getElementById('opt-temporal').checked = !!f.temporal_enabled;
        document.getElementById('opt-hole').checked = !!f.hole_fill_enabled;
        document.getElementById('opt-decimate').checked = !!f.decimation_enabled;
        document.getElementById('opt-decimate-mag').value = f.decimation_magnitude || 1;
      } catch (e) { /* camera may be disconnected */ }
    }

    document.getElementById('opt-ae').addEventListener('change', (ev) => {
      document.getElementById('opt-exposure').disabled = ev.target.value === 'true';
    });

    async function applyCamOpts() {
      const body = {
        depth: {
          visual_preset: document.getElementById('opt-preset').value,
          auto_exposure: document.getElementById('opt-ae').value === 'true',
          gain: parseFloat(document.getElementById('opt-gain').value),
        },
        filters: {
          spatial_enabled: document.getElementById('opt-spatial').checked,
          spatial_magnitude: parseInt(document.getElementById('opt-spatial-mag').value, 10) || 2,
          temporal_enabled: document.getElementById('opt-temporal').checked,
          hole_fill_enabled: document.getElementById('opt-hole').checked,
          decimation_enabled: document.getElementById('opt-decimate').checked,
          decimation_magnitude: parseInt(document.getElementById('opt-decimate-mag').value, 10) || 1,
        },
      };
      const exp = document.getElementById('opt-exposure').value;
      if (!body.depth.auto_exposure && exp) body.depth.exposure_us = parseFloat(exp);
      const laser = document.getElementById('opt-laser').value;
      if (laser && !document.getElementById('opt-laser').disabled)
        body.depth.laser_power = parseFloat(laser);
      if (!document.getElementById('opt-emitter').disabled)
        body.depth.emitter_enabled = document.getElementById('opt-emitter').value === 'true';
      await apiPatch('/api/camera/options', body);
      await loadCamOpts();
    }

    async function apiPost(path, body, opts) {
      const quiet = opts && opts.quiet;
      try {
        const r = await fetch(path, {
          method: 'POST',
          headers: body ? {'Content-Type': 'application/json'} : {},
          body: body ? JSON.stringify(body) : undefined,
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || r.statusText);
        if (!quiet) toast(data.message || 'OK');
        refresh();
        return data;
      } catch (e) {
        toast(e.message, true);
        throw e;
      }
    }

    async function apiDelete(path) {
      try {
        const r = await fetch(path, { method: 'DELETE' });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.error || r.statusText);
        toast(data.message || 'Cleared');
        refresh();
        return data;
      } catch (e) {
        toast(e.message, true);
        throw e;
      }
    }

    function syncMaskClicksFromStatus(mask) {
      if (maskClickBusy) return;
      if (Array.isArray(mask.include_clicks)) {
        maskClicks = mask.include_clicks.map(c => [c.u, c.v]);
      } else if (mask.clicks) {
        maskClicks = mask.clicks.filter(c => c.mode !== 'exclude').map(c => [c.u, c.v]);
      }
      if (Array.isArray(mask.exclude_clicks)) {
        maskExcludeClicks = mask.exclude_clicks.map(c => [c.u, c.v]);
      } else if (mask.clicks) {
        maskExcludeClicks = mask.clicks.filter(c => c.mode === 'exclude').map(c => [c.u, c.v]);
      }
      drawClickOverlay();
    }

    function drawClickOverlay() {
      const canvas = document.getElementById('rgb-click-overlay');
      const panel = document.getElementById('rgb-click-panel');
      if (!canvas || !panel) return;
      const rect = panel.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = rect.width + 'px';
      canvas.style.height = rect.height + 'px';
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const nw = camResolution[0] || 640;
      const nh = camResolution[1] || 480;
      const scale = Math.min(rect.width / nw, rect.height / nh);
      const dispW = nw * scale;
      const dispH = nh * scale;
      const ox = (rect.width - dispW) * 0.5;
      const oy = (rect.height - dispH) * 0.5;
      const toDisp = (uv) => [ox + uv[0] * scale, oy + uv[1] * scale];
      const incColors = ['#00ff78', '#00e68c', '#28ffa0', '#50ffb4', '#78ffc8', '#a0ffdc'];
      maskClicks.forEach((uv, i) => {
        const [x, y] = toDisp(uv);
        ctx.strokeStyle = incColors[i % incColors.length];
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = incColors[i % incColors.length];
        ctx.font = '12px system-ui';
        ctx.fillText(String(i + 1), x + 14, y - 6);
      });
      maskExcludeClicks.forEach((uv) => {
        const [x, y] = toDisp(uv);
        ctx.strokeStyle = '#ff5050';
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#ff5050';
        ctx.font = '11px system-ui';
        ctx.fillText('X', x + 12, y + 4);
      });
    }

    function mapClickToImage(ev) {
      const panel = document.getElementById('rgb-click-panel');
      const rect = panel.getBoundingClientRect();
      const nw = camResolution[0] || 640;
      const nh = camResolution[1] || 480;
      const scale = Math.min(rect.width / nw, rect.height / nh);
      const dispW = nw * scale;
      const dispH = nh * scale;
      const offsetX = rect.left + (rect.width - dispW) * 0.5;
      const offsetY = rect.top + (rect.height - dispH) * 0.5;
      const x = ev.clientX - offsetX;
      const y = ev.clientY - offsetY;
      if (x < 0 || y < 0 || x > dispW || y > dispH) return null;
      const u = Math.round(x / scale);
      const v = Math.round(y / scale);
      return [Math.max(0, Math.min(nw - 1, u)), Math.max(0, Math.min(nh - 1, v))];
    }

    async function addMaskClick(uv, mode) {
      if (mode === 'include' && maskClicks.length >= MAX_MASK_CLICKS) {
        toast(`Max ${MAX_MASK_CLICKS} include clicks — Clear first`, true);
        return;
      }
      if (mode === 'exclude') {
        if (!maskClicks.length) {
          toast('Add an include click on the object first', true);
          return;
        }
        if (maskExcludeClicks.length >= MAX_MASK_CLICKS) {
          toast(`Max ${MAX_MASK_CLICKS} exclude clicks — Clear first`, true);
          return;
        }
      }
      maskClickBusy = true;
      try {
        const data = await apiPost('/api/mask/click', {
          u: uv[0], v: uv[1], mode, use_haiku: useHaikuShape,
        }, { quiet: true });
        const m = data && data.mask;
        if (m) syncMaskClicksFromStatus(m);
        if (m && m.needs_camera_depth) {
          toast('Camera not streaming — click Connect, then try again', true);
        } else if (m && m.click_count) {
          toast(`Mask: ${m.click_count} include` + (m.exclude_count ? `, ${m.exclude_count} exclude` : ''));
        }
      } catch (e) {
        toast(e.message || 'Click failed', true);
      } finally {
        maskClickBusy = false;
      }
    }

    const clickOverlay = document.getElementById('rgb-click-overlay');
    clickOverlay.addEventListener('click', async (ev) => {
      if (ev.button !== 0) return;
      ev.preventDefault();
      const uv = mapClickToImage(ev);
      if (!uv) {
        toast('Click on the image area', true);
        return;
      }
      await addMaskClick(uv, maskClickMode);
    });
    clickOverlay.addEventListener('contextmenu', async (ev) => {
      ev.preventDefault();
      const uv = mapClickToImage(ev);
      if (!uv) return;
      await addMaskClick(uv, 'exclude');
    });
    window.addEventListener('resize', () => drawClickOverlay());


    async function clearMask() {
      maskClicks = [];
      maskExcludeClicks = [];
      await apiDelete('/api/mask/clicks');
      drawClickOverlay();
    }

    async function startScan() {
      const tiltRaw = document.getElementById('scan-tilt-levels').value;
      const tilt_levels = tiltRaw.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
      const body = {
        name: document.getElementById('scan-name').value || 'scan',
        rotate_step_deg: parseFloat(document.getElementById('scan-rotate-step').value) || 15,
        tilt_levels,
        auto_process: document.getElementById('scan-auto-process').checked,
      };
      const data = await apiPost('/api/scan/turntable', body);
      activeJobId = data.job_id;
      document.getElementById('btn-cancel-scan').disabled = false;
    }

    async function cancelScan() {
      if (!activeJobId) return;
      await apiPost('/api/jobs/' + activeJobId + '/cancel');
    }

    function setDisabled(id, disabled) {
      const el = document.getElementById(id);
      if (el) el.disabled = disabled;
    }

    function updateButtonStates(s) {
      const cam = s.camera || {};
      const camOn = !!(cam.connected || s.camera_connected);
      const tt = s.turntable || {};
      const job = s.active_job;
      const scanActive = job && (job.status === 'pending' || job.status === 'running');

      setDisabled('btn-cam-connect', camOn || scanActive);
      setDisabled('btn-cam-disconnect', !camOn || scanActive);
      setCamSettingsDisabled(!camOn || scanActive);

      setDisabled('btn-tt-connect', tt.connected || scanActive);
      setDisabled('btn-tt-disconnect', !tt.connected || scanActive);
      setDisabled('btn-tt-home', !tt.connected || scanActive);
      setDisabled('btn-tt-stop', !tt.connected);
      setDisabled('btn-tt-rotate-minus', !tt.connected || scanActive);
      setDisabled('btn-tt-rotate-plus', !tt.connected || scanActive);
      setDisabled('btn-tt-rotate-m2', !tt.connected || scanActive);
      setDisabled('btn-tt-rotate-m5', !tt.connected || scanActive);
      setDisabled('btn-tt-rotate-p2', !tt.connected || scanActive);
      setDisabled('btn-tt-rotate-p5', !tt.connected || scanActive);
      setDisabled('btn-tt-tilt-minus', !tt.connected || scanActive);
      setDisabled('btn-tt-tilt-plus', !tt.connected || scanActive);
      setDisabled('btn-voxel-reset', !camOn || scanActive);

      setDisabled('btn-start-scan', scanActive || !camOn);
      setDisabled('btn-cancel-scan', !scanActive);
    }

    async function pollStatus() {
      const s = await fetch('/api/status').then(r => r.json());
      const cam = s.camera || {};
      const camOn = !!(cam.connected || s.camera_connected);
      const tt = s.turntable || {};
      const vx = s.voxel || {};
      const mask = s.mask || {};
      const job = s.active_job;

      if (cam.resolution) camResolution = cam.resolution;
      syncMaskClicksFromStatus(mask);

      document.getElementById('status-bar').innerHTML =
        `<span class="${camOn ? 'ok' : 'err'}">Camera: ${camOn ? (cam.device || 'connected') : 'disconnected'}</span>` +
        `<span>${camOn ? cam.resolution?.[0] + '×' + cam.resolution?.[1] + ' @ ' + cam.fps_actual + ' fps' : ''}</span>` +
        `<span class="${tt.connected ? 'ok' : 'warn'}">Turntable: ${tt.connected ? 'connected' : 'disconnected'}</span>` +
        `<span>Mask: ${mask.click_count || 0} incl` +
        `${mask.exclude_count ? ', ' + mask.exclude_count + ' excl' : ''}` +
        `${mask.mode && mask.mode !== 'none' ? ' (' + mask.mode + ')' : ''}</span>` +
        `<span>Voxel: ${vx.points || 0} pts / ${vx.frames || 0} frames` +
        `${vx.live_on_rgb ? ' (on RGB)' : ''}</span>`;

      const liveVxCb = document.getElementById('opt-live-voxel');
      if (liveVxCb && !liveVoxelToggleBusy && vx.live_on_rgb !== undefined) {
        liveVxCb.checked = !!vx.live_on_rgb;
      }

      let maskLabel = mask.ready
        ? `${mask.click_count} incl${mask.exclude_count ? ', ' + mask.exclude_count + ' excl' : ''} — ${mask.mode}${mask.focus_mm ? ', ' + Math.round(mask.focus_mm) + ' mm' : ''}`
        : 'No mask (full frame at scan)';
      if (mask.assist_succeeded) {
        maskLabel += ` · assist OK (${mask.detected_pixels || 0} px)`;
      } else if (mask.click_count >= 1 && mask.mode === 'assist_detect') {
        maskLabel += ' · assist fallback cube';
      }
      if (mask.last_action === 'expanded_by_mm' && mask.expanded_by_mm) {
        const d = mask.expanded_by_mm;
        maskLabel += ` · expanded +${Math.round(d.x)}/${Math.round(d.y)}/${Math.round(d.z)} mm`;
      } else if (mask.last_click_mode && mask.click_count > 1) {
        maskLabel += ` · last: ${mask.last_click_mode}`;
      }
      if (mask.cube && mask.cube.dimensions_mm) {
        const dim = mask.cube.dimensions_mm;
        maskLabel += ` · cube ${Math.round(dim.w)}×${Math.round(dim.h)}×${Math.round(dim.d)} mm`;
      } else if (mask.cube && mask.cube.half_extents_mm) {
        const h = mask.cube.half_extents_mm;
        maskLabel += ` · cube ±${Math.round(h.x)}×${Math.round(h.y)}×${Math.round(h.z)} mm`;
      }
      if (mask.tracking && mask.cube) {
        const c = mask.cube.center_mm;
        maskLabel += ` · tracking (${mask.points_in_bounds || 0} pts)`;
        if (c) maskLabel += ` @ ${Math.round(c.z)} mm`;
      } else if (mask.cube && mask.cube.initialized) {
        maskLabel += ' · cube ready';
      }
      if (mask.shape_source) {
        maskLabel += ` · shape: ${mask.shape_source}`;
      }
      const haikuLine = document.getElementById('haiku-status-line');
      if (haikuLine) {
        let haikuText = 'Haiku: ';
        if (!mask.haiku_available) {
          haikuText += 'off';
          if (mask.haiku_unavailable_reason) {
            haikuText += ' — ' + mask.haiku_unavailable_reason;
          }
        } else {
          const hs = mask.haiku_status || 'idle';
          if (hs === 'pending' || mask.haiku_pending) {
            haikuText += 'calling…';
          } else if (hs === 'success' && mask.haiku_last_used_at) {
            const t = mask.haiku_last_used_at.slice(11, 19);
            const n = (mask.contour_points && mask.contour_points.length) || 0;
            haikuText += `used at ${t} (${n} vertices)`;
          } else if (hs === 'error') {
            haikuText += mask.haiku_message || 'error';
          } else {
            haikuText += hs;
          }
        }
        haikuLine.textContent = haikuText;
        if (mask.haiku_available) {
          const usedAt = mask.haiku_last_used_at || '';
          if (usedAt && usedAt !== window._lastHaikuUsedAt) {
            window._lastHaikuUsedAt = usedAt;
            haikuLine.classList.add('flash');
            setTimeout(() => haikuLine.classList.remove('flash'), 2000);
          }
        }
      }
      const haikuRow = document.getElementById('haiku-toggle-row');
      const haikuCb = document.getElementById('opt-use-haiku');
      if (mask.haiku_available) {
        haikuRow.style.display = '';
        if (!haikuToggleBusy && mask.use_haiku !== undefined) {
          useHaikuShape = !!mask.use_haiku;
          haikuCb.checked = useHaikuShape;
        }
      } else {
        haikuRow.style.display = 'none';
        useHaikuShape = false;
      }
      document.getElementById('mask-status').textContent = maskLabel;

      let jobHtml = 'No active scan job';
      if (job) {
        activeJobId = job.id;
        const p = job.progress || {};
        jobHtml = `Job <code>${job.id}</code>: <strong>${job.status}</strong> — ${job.message}`;
        if (p.total) jobHtml += ` (${p.current}/${p.total})`;
        if (job.result_path) jobHtml += ` · <a href="#">${job.result_path}</a>`;
        if (job.error) jobHtml += ` · <span class="err">${job.error}</span>`;
      } else {
        activeJobId = null;
      }
      document.getElementById('job-detail').innerHTML = jobHtml;

      updateButtonStates(s);
      if (camOn) loadCamOpts();
      else setCamSettingsDisabled(true);
      return s;
    }

    async function refresh() {
      try {
        await pollStatus();
      } catch (e) {
        document.getElementById('status-bar').textContent = 'Status unavailable';
      }
    }

    setInterval(refresh, 1000);
    setInterval(refreshTelemetry, 3000);
    refresh();
    refreshTelemetry();
  </script>
</body>
</html>"""


def _parse_turntable_body(body: dict[str, Any]) -> tuple[TurntableScanConfig, dict[str, Any]]:
    tilt_levels = body.get("tilt_levels", TURNTABLE_TILT_LEVELS)
    if isinstance(tilt_levels, str):
        tilt_levels = [float(v) for v in tilt_levels.split(",")]
    cfg = TurntableScanConfig(
        tilt_levels=[float(v) for v in tilt_levels],
        rotate_step_deg=float(body.get("rotate_step_deg", TURNTABLE_ROTATE_STEP_DEG)),
        rotate_wait_s=float(body.get("rotate_wait_s", TURNTABLE_ROTATE_WAIT_S)),
        tilt_wait_s=float(body.get("tilt_wait_s", TURNTABLE_TILT_WAIT_S)),
        ble_address=body.get("ble_address"),
        scan_timeout=float(body.get("scan_timeout", 10.0)),
    )
    opts = {
        "name": str(body.get("name", "scan")),
        "output_dir": Path(body.get("output_dir", str(DEFAULT_SCANS_DIR))),
        "auto_process": bool(body.get("auto_process", False)),
        "use_calibration": bool(body.get("use_calibration", True)),
        "calibration_dir": Path(body["calibration_dir"])
        if body.get("calibration_dir")
        else DEFAULT_CALIBRATION_DIR,
    }
    return cfg, opts


def _make_handler(service: CameraService) -> type[BaseHTTPRequestHandler]:
    _job_re = re.compile(r"^/api/jobs/([a-f0-9]+)$")
    _cancel_re = re.compile(r"^/api/jobs/([a-f0-9]+)/cancel$")

    class Handler(BaseHTTPRequestHandler):
        server_version = "ScannerWeb/2.3"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error_json(self, message: str, status: HTTPStatus) -> None:
            self._send_json({"error": message}, status=status)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = _INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/status":
                self._send_json(service.status_dict())
            elif path == "/api/mask/status":
                self._send_json(service.mask_status())
            elif path == "/api/camera/options":
                self._send_json(service.get_camera_options())
            elif path == "/api/telemetry":
                self._send_json(service.get_telemetry())
            elif path == "/api/jobs":
                jobs = [j.to_dict() for j in service.jobs.list_jobs()]
                self._send_json({"jobs": jobs})
            elif m := _job_re.match(path):
                job = service.jobs.get(m.group(1))
                if job is None:
                    self._send_error_json("job not found", HTTPStatus.NOT_FOUND)
                else:
                    self._send_json(job.to_dict())
            elif path == "/stream/rgb":
                self._serve_mjpeg(service.rgb_jpeg)
            elif path == "/stream/depth":
                self._serve_mjpeg(service.depth_jpeg)
            elif path == "/stream/voxel":
                self._serve_mjpeg(service.voxel_jpeg)
            elif path == "/api/frame/rgb":
                self._serve_jpeg(service.rgb_jpeg())
            elif path == "/api/frame/depth":
                self._serve_jpeg(service.depth_jpeg())
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json_body() if self.headers.get("Content-Length") else {}

                if path == "/api/mask/clicks":
                    points = body.get("points")
                    include = body.get("include")
                    exclude = body.get("exclude")
                    mode = body.get("mode")
                    use_haiku = body.get("use_haiku")
                    if include is None and exclude is None and points is None:
                        raise ValueError("Provide include/exclude clicks or legacy points")
                    status = service.set_mask_clicks(
                        points,
                        include=include,
                        exclude=exclude,
                        mode=mode,
                        use_haiku=use_haiku if use_haiku is not None else None,
                    )
                    self._send_json({"ok": True, "mask": status})
                    return

                if path == "/api/mask/click":
                    u = body.get("u")
                    v = body.get("v")
                    if u is None or v is None:
                        raise ValueError("Provide u and v")
                    mode = body.get("mode", "include")
                    use_haiku = body.get("use_haiku")
                    status = service.append_mask_click(
                        int(u),
                        int(v),
                        mode=str(mode),
                        use_haiku=use_haiku if use_haiku is not None else None,
                    )
                    self._send_json({"ok": True, "mask": status})
                    return

                if path == "/api/mask/use-haiku":
                    enabled = body.get("enabled")
                    if enabled is None:
                        raise ValueError("Provide enabled (true/false)")
                    status = service.set_mask_use_haiku(bool(enabled))
                    self._send_json({"ok": True, "mask": status})
                    return

                if path == "/api/mask/assist-haiku":
                    status = service.assist_mask_haiku()
                    self._send_json({"ok": True, "mask": status})
                    return

                if path == "/api/camera/connect":
                    service.connect_camera()
                    self._send_json({"ok": True, "message": "Camera connected"})
                    return
                if path == "/api/camera/disconnect":
                    service.disconnect_camera()
                    self._send_json({"ok": True, "message": "Camera disconnected"})
                    return

                if path == "/api/voxel/reset":
                    service.reset_voxel()
                    self._send_json({"ok": True, "message": "Voxel model reset"})
                    return
                if path == "/api/voxel/live":
                    enabled = body.get("enabled")
                    if enabled is None:
                        raise ValueError("Provide enabled (true/false)")
                    out = service.set_live_voxel_on_rgb(bool(enabled))
                    self._send_json({"ok": True, **out})
                    return

                if path == "/api/turntable/connect":
                    service.turntable.connect(
                        body.get("ble_address"),
                        scan_timeout=float(body.get("scan_timeout", 10.0)),
                    )
                    self._send_json({"ok": True, "message": "Turntable connected"})
                    return
                if path == "/api/turntable/disconnect":
                    service.turntable.disconnect()
                    self._send_json({"ok": True, "message": "Turntable disconnected"})
                    return
                if path == "/api/turntable/rotate":
                    degrees = body.get("degrees")
                    if degrees is None:
                        raise ValueError("Missing degrees")
                    service.turntable.rotate(float(degrees))
                    self._send_json({"ok": True, "message": f"Rotated {degrees}°"})
                    return
                if path == "/api/turntable/tilt":
                    if "angle" in body:
                        service.turntable.tilt(angle=float(body["angle"]))
                    elif "delta" in body or "degrees" in body:
                        delta = body.get("delta", body.get("degrees"))
                        service.turntable.tilt(delta=float(delta))
                    else:
                        raise ValueError("Provide angle or delta")
                    self._send_json({"ok": True, "message": "Tilt command sent"})
                    return
                if path == "/api/turntable/home":
                    service.turntable.home()
                    self._send_json({"ok": True, "message": "Homing turntable"})
                    return
                if path == "/api/turntable/stop":
                    service.turntable.emergency_stop()
                    self._send_json({"ok": True, "message": "Emergency stop sent"})
                    return

                if path == "/api/scan/turntable":
                    cfg, opts = _parse_turntable_body(body)
                    job = service.submit_turntable_scan(cfg=cfg, **opts)
                    self._send_json(
                        {"job_id": job.id, "job": job.to_dict()},
                        HTTPStatus.ACCEPTED,
                    )
                    return
                if m := _cancel_re.match(path):
                    try:
                        job = service.cancel_job(m.group(1))
                    except KeyError:
                        self._send_error_json("job not found", HTTPStatus.NOT_FOUND)
                        return
                    self._send_json(job.to_dict())
                    return
            except json.JSONDecodeError:
                self._send_error_json("invalid JSON body", HTTPStatus.BAD_REQUEST)
                return
            except RuntimeError as exc:
                self._send_error_json(str(exc), HTTPStatus.CONFLICT)
                return
            except ValueError as exc:
                self._send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_PATCH(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json_body()
                if path == "/api/camera/options":
                    opts = service.patch_camera_options(body)
                    self._send_json({"ok": True, "options": opts})
                    return
                if path == "/api/mask/settings":
                    settings = service.patch_mask_settings(body)
                    self._send_json({"ok": True, "settings": settings})
                    return
            except json.JSONDecodeError:
                self._send_error_json("invalid JSON body", HTTPStatus.BAD_REQUEST)
                return
            except RuntimeError as exc:
                self._send_error_json(str(exc), HTTPStatus.CONFLICT)
                return
            except ValueError as exc:
                self._send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_DELETE(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/api/mask/clicks":
                    service.clear_mask_clicks()
                    self._send_json({"ok": True, "message": "Mask cleared"})
                    return
            except RuntimeError as exc:
                self._send_error_json(str(exc), HTTPStatus.CONFLICT)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _serve_jpeg(self, jpeg: bytes) -> None:
            if not jpeg:
                self._send_error_json("no frame available", HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)

        def _serve_mjpeg(self, frame_fn: Any) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}",
            )
            self.end_headers()
            try:
                while True:
                    jpeg = frame_fn()
                    if not jpeg:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b"--" + _BOUNDARY + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.001)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    return Handler


def _lan_urls(host: str, port: int) -> list[str]:
    urls = [f"http://127.0.0.1:{port}/"]
    if host in ("0.0.0.0", ""):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                lan_ip = s.getsockname()[0]
            urls.append(f"http://{lan_ip}:{port}/")
        except OSError:
            pass
    elif host not in ("127.0.0.1", "localhost"):
        urls.append(f"http://{host}:{port}/")
    return urls


def run_web_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> int:
    """Start web server; owns camera until Ctrl+C."""
    configure_stdio_utf8()
    from scanner.env_secrets import ensure_anthropic_api_key

    if ensure_anthropic_api_key():
        print("Haiku: ANTHROPIC_API_KEY loaded")
    else:
        print("Haiku: off (ANTHROPIC_API_KEY not found)")
    service = CameraService()
    try:
        service.start()
    except RuntimeError as exc:
        print(f"RealSense not available at startup:\n{exc}", file=sys.stderr)
        print("Server will run — use Connect in the web UI when the camera is ready.")

    cam = service.camera
    handler = _make_handler(service)
    httpd = ThreadingHTTPServer((host, port), handler)

    if service.camera_connected:
        print(f"Camera: {cam.label}  {cam.resolution[0]}×{cam.resolution[1]} @ {cam.fps} fps")
    else:
        print("Camera: not connected")
    print("Scanner web server running (camera owned by this process).")
    for url in _lan_urls(host, port):
        print(f"  Viewer: {url}")
    print(f"  API:    http://127.0.0.1:{port}/api/status")
    print("Start scans from the web UI or: python -m scanner turntable-scan --name myobject")
    print("Stop with Ctrl+C.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        httpd.shutdown()
        httpd.server_close()
        service.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "3D scanner web server — live view + REST API. "
            f"Default LAN port {DEFAULT_PORT}."
        )
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    args = parser.parse_args(argv)
    return run_web_server(host=args.host, port=args.port)


if __name__ == "__main__":
    raise SystemExit(main())
