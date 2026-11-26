# ⚡ LocalGPU-OS

> **A Model- and Application-Agnostic Local Compute Resource Arbitration Layer**  
> Manages GPU resources, workloads, isolated worker environments, and hardware safety so multiple AI applications can reliably share a single local GPU — **zero cloud API fees, no mandatory subscriptions, local-first by default**.

🌐 **Languages**: [繁體中文](README.md) | **English**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![PRD](https://img.shields.io/badge/PRD-v1.1.2_Frozen-brightgreen.svg)](PRD.md)

---

## 🎯 Core Positioning

`LocalGPU-OS` is **not** a monolithic AI mega-suite. It is a lightweight **local compute resource arbitration layer** built for personal workstations and discrete laptop GPUs (from 6GB/8GB RTX 4060 Laptops to 24GB+ cards, with safe CPU fallback).

It focuses strictly on four responsibilities:
1. **Know what hardware exists** (`HardwareProfile` probing).
2. **Know how much resource is actually free** (tracking both managed workers and external unmanaged GPU processes).
3. **Schedule workloads safely** (3-layer `VRAM-Gate`, dynamic execution tiers, thermal pacing, and Fail-Safe admission).
4. **Launch, monitor, and reclaim isolated workers reliably** (per-worker `.venv`, standardized manifest contract, graceful cancellation, and crash recovery).

---

## ✨ Key Architectural Highlights

- **🔒 Three-Layer `VRAM-Gate`**: Combines a *Registry-Locked Atomic Lease* (TTL, heartbeat, PID-reuse guard, fencing token) + *External GPU Monitoring* + *Dynamic Admission Control* (`VRAM_admission = VRAM_free,observed - VRAM_safety - VRAM_pending`, avoiding double-subtraction) with pluggable `EvictionProvider` verification.
- **🎛️ Hardware Profile, Live Snapshot & Runtime Policy Decoupled**: `HardwareProfile` captures static capabilities, `HardwareSnapshot` enforces freshness TTLs, and `AdmissionControl` gates execution dynamically.
- **🌡️ Four-State Thermal Pacing (`ThermalGuard`)**: `NORMAL` / `COOLING` / `CRITICAL` / `SENSOR_DEGRADED` state machine with hysteresis to pace cooperative workers (complements—never replaces—native NVIDIA driver and BIOS protections).
- **🛡️ Fail-Safe Resource Management**: Fails closed on GPU tasks whenever hardware probing, VRAM reads, lease state, or eviction verification fails, while allowing pure `CPU_COMPUTE` workloads to degrade safely.
- **🧩 Zero-Heavy-Dependency Core + Isolated Workers**: Core OS never installs PyTorch, PaddlePaddle, or TensorFlow. Workers run in isolated `.venv` environments via `Worker Manifest Contract v1.0` with decoupled resource classes (including dedicated `GPU_ENCODE` for NVENC).

---

## 🏗️ Architecture & Phased Roadmap

```mermaid
flowchart TD
    Caller["CLI (`lgpu`) / Python SDK / Downstream Apps (COA-parser, ZeroRAG, CodeCut-Studio)"] --> Core
    subgraph Core["LocalGPU-OS Core (Lightweight Resource Arbitration Layer)"]
        H["HardwareDetector & ThermalGuard"]
        G["3-Layer VRAM-Gate (Lock · External Monitor · Admission)"]
        R["Worker Runtime (Manifest v1.0 · Lifecycle · Cancel)"]
    end
    Core --> W1["M1-D: Layered Offline OCR / PDF Worker<br/>(Native PyMuPDF → RapidOCR / Optional Docling)"]
    Core --> W2["M2: Local LLM & Embedding Worker<br/>(Ollama + bge-m3 + Verified Eviction)"]
    Core --> W3["M3: NVENC Media & Offline Speech Worker<br/>(Decoupled GPU_ENCODE Resource Class)"]
```

- **M0 (Frozen — `m0-prd-v1.1.2-frozen`)**: Specification & governance baseline ([PRD.md](PRD.md) `v1.1.2`, [GLOSSARY.md](GLOSSARY.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **M1-A (✅ Completed & Verified — `m1-a-hardware-foundation-v1.0.0`)**: Hardware Foundation (`HardwareProfile` / `HardwareSnapshot`, `PolicyEngine`, `lgpu doctor`, 12 tests).
- **M1-B (✅ Completed & Verified — `m1-b-resource-arbitration-v1.0.0`)**: Resource Arbitration (`RegistryLock`, atomic `FileLeaseStore`, observation-confirmed `VRAM_pending`, fencing tokens, `LEASE_RECLAIM_BLOCKED`, all-or-nothing `VramGate`, 4-state `ThermalGuard`, `OllamaEvictionProvider`, `lgpu status`, 13 tests).
- **M1-C (✅ Completed & Verified — `m1-c-worker-runtime-v1.0.0`)**: Worker Runtime (`WorkerRegistry`, `Worker Manifest Contract v1.0`, strict isolated `.venv` resolution, `json_file_manifest` Base64/path-traversal/symlink guards, `DISK_SPACE_INSUFFICIENT`, 7-step cross-platform cancellation with Step 7 Fail-Closed VRAM re-probe, `lgpu run` + Python SDK, 15 tests).
- **M1-D (✅ Completed & Verified — `m1-d-ocr-pdf-v1.0.0`)**: First production isolated worker — [workers/ocr_pdf](workers/ocr_pdf/README.md) (D1 native-text PyMuPDF → D2 offline RapidOCR CUDA/CPU with pre-load SHA-256 verification → D3 VRAM/OOM, thermal, multi-page cancel, forced-kill output preservation & mixed-batch stress tests → D4 end-to-end integration & 5-round stability, 16 tests; **56 / 56** total tests passing on Windows 11 + RTX 4060 Laptop).
- **M2 – M4 (Planned)**: Local LLM & Embedding Worker + secured localhost HTTP API (M2), NVENC Media & Speech Worker (M3), MCP Server & multi-language docs (M4).

---

## 🚀 Quick Start (M1 Complete)

> **Verified Platform**: Windows 11 + NVIDIA CUDA (single GPU, verified on RTX 4060 Laptop 8GB); Linux as secondary target; gracefully degrades to `CPU_INT8` when GPU is unavailable or VRAM is insufficient. Zero heavy runtime dependencies in Core OS (Python 3.10+ stdlib only).

```powershell
# 1. Clone the repository
git clone https://github.com/etrnya/LocalGPU-OS.git
cd LocalGPU-OS

# 2. Run hardware doctor and dynamic policy check (M1-A)
$env:PYTHONPATH="src"; python -m localgpu_os.presentation.cli doctor --json

# 3. Inspect live VRAM-Gate arbitration state and active leases (M1-B)
$env:PYTHONPATH="src"; python -m localgpu_os.presentation.cli status --json

# 4. Run the isolated offline OCR / PDF worker (M1-C + M1-D; see workers/ocr_pdf/README.md for setup)
$env:PYTHONPATH="src"; python -m localgpu_os.presentation.cli run ocr_pdf `
  --input-manifest .\inputs\manifest.json `
  --allowed-root .\inputs `
  --allow-cpu-fallback `
  --json `
  -- --output-dir .\output --job-id job_001

# 5. Run the full test suite (M1-A through M1-D, 56 tests)
$env:PYTHONPATH="src"; python -m unittest discover -s tests -v
```

---

## 📄 Documentation

- [Product Requirements Document (PRD.md — v1.1.2 Frozen)](PRD.md) *(Traditional Chinese)*
- [M1-A Engineering Design & Test Spec (docs/M1-A_DESIGN_AND_TEST_SPEC.md)](docs/M1-A_DESIGN_AND_TEST_SPEC.md)
- [Domain Glossary (GLOSSARY.md)](GLOSSARY.md)
- [Third-Party Notices (THIRD_PARTY_NOTICES.md)](THIRD_PARTY_NOTICES.md)

---

## ⚖️ License

Source code is licensed under the [Apache License 2.0](LICENSE). Third-party packages, binaries, and model weights remain subject to their respective licenses (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).  
Copyright © 2026 **etrnya**.
