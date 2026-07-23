"""Run scanner subprocesses so Open3D crashes do not kill the capture process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_process_subprocess(
    session_path: Path,
    *,
    use_icp: bool = True,
    output_mesh: Optional[Path] = None,
) -> int:
    """Run `python -m scanner process` in a child process; return exit code."""
    session_path = session_path.resolve()
    cmd = [sys.executable, "-m", "scanner", "process", str(session_path)]
    if not use_icp:
        cmd.append("--no-icp")
    if output_mesh is not None:
        cmd.extend(["--mesh", str(output_mesh.resolve())])
    project_root = Path(__file__).resolve().parent.parent
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(project_root))
    return int(result.returncode)


def process_session_safe(
    session_path: Path,
    *,
    output_mesh: Optional[Path] = None,
    prefer_icp: bool = True,
) -> int:
    """
    Process a session without risking a segfault in the caller (e.g. live capture).

    Tries ICP when prefer_icp is True; on failure (non-zero exit, often 139 on macOS)
    retries once with --no-icp.
    """
    if prefer_icp:
        code = run_process_subprocess(
            session_path, use_icp=True, output_mesh=output_mesh
        )
        if code == 0:
            return 0
        print(
            f"Process exited with code {code}"
            + (" (possible Open3D ICP crash on macOS)" if code in (-11, 139) else "")
            + "; retrying without ICP...",
            file=sys.stderr,
        )
    code = run_process_subprocess(
        session_path, use_icp=False, output_mesh=output_mesh
    )
    if code != 0:
        print(f"Processing failed (exit {code}).", file=sys.stderr)
    else:
        print(
            "Processing finished (no ICP). For moving-camera scans, fix Open3D/ICP "
            "or run process again after upgrading the conda env.",
        )
    return code
