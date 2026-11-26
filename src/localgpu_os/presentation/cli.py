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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
