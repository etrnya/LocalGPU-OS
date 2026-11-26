"""CLI entrypoint for LocalGPU-OS (`lgpu`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from localgpu_os.application.doctor_service import DoctorService
from localgpu_os.application.status_service import StatusReport, StatusService
from localgpu_os.domain.hardware import DoctorReport
from localgpu_os.domain.worker import WorkerRunResult
from localgpu_os.sdk import LocalGpuClient


def _configure_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def format_doctor_human(report: DoctorReport) -> str:
    d = report.profile
    s = report.snapshot
    p = report.decision

    lines = [
        f"LocalGPU-OS Doctor (schema v{report.schema_version})",
        "=" * 52,
        f"Device          : {d.device_name} ({d.vendor}, {d.device_id})",
        f"Architecture    : {d.architecture or 'UNKNOWN'} (CC {d.compute_capability or 'N/A'})",
        f"Driver Version  : {d.driver_version or 'N/A'}",
        f"Total VRAM      : {f'{d.total_vram_mib} MiB' if d.total_vram_mib is not None else 'N/A'}",
        f"Free VRAM       : {f'{s.available_vram_mib} MiB' if s.available_vram_mib is not None else 'UNAVAILABLE'}",
        f"Admission VRAM  : {p.vram_admission_mib} MiB (safety margin: {p.vram_safety_mib} MiB)",
        f"Temperature     : {f'{s.temperature_c} C' if s.temperature_c is not None else 'N/A'} ({p.thermal_state.value})",
        f"NVENC Support   : {d.nvenc_supported}",
        "-" * 52,
        f"Policy Tier     : {p.tier.value}",
        f"Backend         : {p.recommended_backend.value}",
        f"GPU Admission   : {'ALLOWED' if p.gpu_admission_allowed else 'DENIED'}",
    ]
    if p.error_codes:
        lines.append(f"Error Codes     : {', '.join(e.value for e in p.error_codes)}")
    if p.warnings:
        lines.append("Warnings:")
        for w in p.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def format_status_human(report: StatusReport) -> str:
    d = report.profile
    s = report.snapshot
    lines = [
        f"LocalGPU-OS Status (schema v{report.schema_version})",
        "=" * 52,
        f"Device          : {d.device_name} ({d.vendor}, {d.device_id})",
        f"Free VRAM       : {f'{s.available_vram_mib} MiB' if s.available_vram_mib is not None else 'UNAVAILABLE'}",
        f"Pending VRAM    : {report.vram_pending_mib} MiB",
        f"Safety Margin   : {report.vram_safety_mib} MiB",
        f"Admission VRAM  : {report.vram_admission_mib} MiB",
        f"Temperature     : {f'{s.temperature_c} C' if s.temperature_c is not None else 'N/A'} ({report.thermal_state.value})",
        "-" * 52,
        f"Registry Gen    : {report.registry_generation}",
        f"Active Leases   : {len(report.active_leases)}",
    ]
    for lease in report.active_leases:
        lines.append(
            f"  - {lease['lease_id']} (gen={lease['lease_generation']}, "
            f"owner={lease['owner_id']}, state={lease['reservation_state']}, "
            f"vram={lease['reserved_vram_mib']} MiB, confirmed={lease['vram_observed_confirmed']})"
        )
    if report.error_codes:
        lines.append(f"Error Codes     : {', '.join(e.value for e in report.error_codes)}")
    if report.warnings:
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def format_run_human(result: WorkerRunResult) -> str:
    lines = [
        f"LocalGPU-OS Worker Run (schema v{result.schema_version})",
        "=" * 52,
        f"Run ID          : {result.run_id}",
        f"Worker ID       : {result.worker_id}",
        f"Status          : {result.status}",
        f"Exit Code       : {result.exit_code if result.exit_code is not None else 'N/A'}",
        f"Lease           : {result.lease_id or 'NONE'} (gen={result.lease_generation if result.lease_generation is not None else 'N/A'})",
        f"Network Policy  : {result.network_policy.value}",
        f"CPU Fallback    : {result.used_cpu_fallback}",
        f"Forced Kill     : {result.forced_kill} (VRAM reprobed: {result.post_kill_vram_reprobed})",
    ]
    if result.error_codes:
        lines.append(f"Error Codes     : {', '.join(e.value for e in result.error_codes)}")
    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")
    if result.output_data is not None:
        lines.append(f"Output Data     : {json.dumps(result.output_data, ensure_ascii=False)}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lgpu",
        description="LocalGPU-OS: Local Compute Resource Arbitration Layer",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doc = sub.add_parser("doctor", help="Inspect local GPU hardware and dynamic policy")
    doc.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON snapshot")
    doc.add_argument("--device", type=int, default=0, help="Target NVIDIA GPU index (default: 0)")

    st = sub.add_parser("status", help="Inspect live VRAM-Gate arbitration state and leases")
    st.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON status report")
    st.add_argument("--device", type=int, default=0, help="Target NVIDIA GPU index (default: 0)")
    st.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Override state directory containing registry.lock and leases.json",
    )

    run_p = sub.add_parser("run", help="Run an isolated worker under VRAM-Gate arbitration")
    run_p.add_argument("worker_id", help="Registered worker ID under workers/<worker_id>")
    run_p.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON run result")
    run_p.add_argument("--device", type=int, default=0, help="Target NVIDIA GPU index (default: 0)")
    run_p.add_argument(
        "--workers-dir",
        type=Path,
        default=None,
        help="Override workers root directory (default: ./workers)",
    )
    run_p.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Override state directory containing registry.lock and leases.json",
    )
    run_p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Override per-run working directory",
    )
    run_p.add_argument(
        "--input-manifest",
        type=Path,
        default=None,
        help="Path to json_file_manifest input file",
    )
    run_p.add_argument(
        "--allowed-root",
        type=Path,
        default=None,
        help="Allowed root directory for json_file_manifest path traversal protection",
    )
    run_p.add_argument(
        "--allow-eviction",
        action="store_true",
        help="Allow external service eviction if VRAM is insufficient",
    )
    run_p.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Allow automatic fallback to CPU_COMPUTE when GPU admission is denied",
    )
    run_p.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override manifest execution timeout in seconds",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    forwarded_worker_args: list[str] = []
    if "--" in raw_argv:
        sep_idx = raw_argv.index("--")
        forwarded_worker_args = raw_argv[sep_idx + 1 :]
        raw_argv = raw_argv[:sep_idx]

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.command == "doctor":
        report = DoctorService().run_doctor(device_index=args.device)
        if args.as_json:
            print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_doctor_human(report))
        return 0

    if args.command == "status":
        status_report = StatusService(state_dir=args.state_dir).run_status(
            device_index=args.device
        )
        if args.as_json:
            print(json.dumps(status_report.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_status_human(status_report))
        return 0

    if args.command == "run":
        client = LocalGpuClient(
            workers_dir=args.workers_dir,
            state_dir=args.state_dir,
        )
        result = client.run_worker(
            worker_id=args.worker_id,
            extra_args=forwarded_worker_args,
            input_manifest=args.input_manifest,
            allowed_root=args.allowed_root,
            work_dir=args.work_dir,
            device_index=args.device,
            allow_eviction=args.allow_eviction,
            allow_cpu_fallback=args.allow_cpu_fallback,
            timeout_seconds=args.timeout,
        )
        if args.as_json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(format_run_human(result))
        return 0 if result.succeeded else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
