# 🛠️ M1-A Engineering Design & Test Specification (Hardware Foundation)

> **階段目標**：建立 `LocalGPU-OS` 穩定、可驗證、零重型依賴的硬體事實來源 (`HardwareProfile` + `HardwareSnapshot` + `PolicyEngine` + `lgpu doctor`)。  
> **嚴格邊界**：本階段**只負責硬體可觀測性與策略判定**；絕不提前實作 `VramLease`、`RegistryLock`、任務排隊、Ollama 卸載或 Worker 啟動。

---

## 1. 模組邊界與 DDD 分層設計

```text
src/localgpu_os/
├── __init__.py
├── domain/
│   ├── enums.py            # ExecutionPolicyTier, ComputeBackend, SensorStatus, ErrorCode
│   ├── hardware.py         # HardwareProfile (靜態), HardwareSnapshot (動態), DoctorReport
│   └── policy.py           # PolicyEngine (純函式動態策略評估，零 I/O)
├── infrastructure/
│   ├── clock.py            # ClockProtocol (UTC + monotonic clock，便於精準 Mock 測試)
│   └── nvidia_detector.py  # NvidiaSmiDetector (Windows/Linux 單一指定 GPU 探測，零 shell=True)
├── application/
│   └── doctor_service.py   # 組合 HardwareDetector + PolicyEngine 產出 schema_version "1.0" 快照
└── presentation/
    └── cli.py              # `lgpu doctor [--json] [--device 0]`
```

### 核心介面契約
1. **`HardwareProfile`（靜態能力，不可變 dataclass）**：
   - `device_id: str`（預設 `"cuda:0"` 或 `"cpu"`）
   - `vendor: str`（`"NVIDIA"` | `"CPU"` | `"UNKNOWN"`）
   - `device_name: str`
   - `architecture: str | None`（無法取得為 `None`，絕不填 `0`）
   - `compute_capability: str | None`
   - `total_vram_mib: int | None`
   - `memory_bandwidth_gbps: float | None`
   - `power_limit_w: float | None`
   - `nvenc_supported: bool | None`
   - `temperature_support: bool`
   - `backend_support: tuple[ComputeBackend, ...]`
2. **`HardwareSnapshot`（動態觀測，不可變 dataclass）**：
   - `device_id: str`
   - `observed_at: str`（UTC ISO-8601）
   - `observed_monotonic: float`（內部單調時鐘）
   - `freshness_ttl_seconds: float`（預設 `10.0`）
   - `available_vram_mib: int | None`（裝置層級即時 Free Memory；探測失敗為 `None`）
   - `external_vram_mib: int | None`（輔助歸因，絕不重複扣除）
   - `temperature_c: int | None`
   - `utilization_percent: int | None`
   - `sensor_status: SensorStatus`（`OK` | `DEGRADED` | `UNAVAILABLE`）
   - 方法：`is_fresh(now_monotonic: float) -> bool`（若 `now_monotonic < observed_monotonic` 或年齡超過 TTL 均回傳 `False`）
3. **`PolicyEngine.evaluate(profile, snapshot, now_monotonic, safety_config)`**：
   - 計算 `vram_admission_mib = max(0, available_vram_mib - safety_mib)`（M1-A 無 pending，預設 `safety_mib = system_reserve_mib(768) + runtime_overhead_mib(512) + uncertainty_margin_mib(256) = 1536 MiB`）。
   - 回傳 `ExecutionPolicyDecision`：
     - `tier: ExecutionPolicyTier`
     - `recommended_backend: ComputeBackend`
     - `gpu_admission_allowed: bool`
     - `thermal_state: ThermalState`（`NORMAL` / `COOLING` / `CRITICAL` / `SENSOR_DEGRADED`）
     - `vram_admission_mib: int`
     - `error_codes: tuple[ErrorCode, ...]`
     - `warnings: tuple[str, ...]`
4. **標準 JSON 輸出契約 (`schema_version: "1.0"`)**：
   - 頂層鍵固定為：`schema_version`, `device`, `snapshot`, `capabilities`, `execution_policy`。

---

## 2. M1-A 測試規格 (Test Specification)

### A. 必要邊界情境驗收（Scenario Coverage = 100%）
| 測試編號 | 測試情境 | 預期判定結果 |
| :---: | :--- | :--- |
| **`S1`** | 8GB NVIDIA GPU，外部佔用 `0 MiB`（Free `8188 MiB`，溫度 `55°C`） | `tier == TIER_STANDARD`, `gpu_admission_allowed == True`, `recommended_backend == CUDA`, `thermal_state == NORMAL` |
| **`S2`** | 8GB NVIDIA GPU，外部已佔 `4096 MiB`（Free `4092 MiB`） | `vram_admission_mib == 2556`, 動態降級 `tier == TIER_LITE`, `gpu_admission_allowed == True` |
| **`S3`** | 8GB NVIDIA GPU，外部已佔 `7168 MiB`（Free `1020 MiB` < Safety `1536 MiB`） | `vram_admission_mib == 0`, `gpu_admission_allowed == False`, 含 `VRAM_INSUFFICIENT` |
| **`S4`** | 16GB NVIDIA GPU，外部已佔 `12288 MiB`（Free `4096 MiB`） | 動態降級 `tier == TIER_LITE`（證明不寫死顯卡型號）, `gpu_admission_allowed == True` |
| **`S5`** | 24GB NVIDIA GPU，但 VRAM 探測失敗（`available_vram_mib is None`） | Fail-Closed：`gpu_admission_allowed == False`, 含 `HARDWARE_UNVERIFIED`, `recommended_backend == CPU_INT8` |
| **`S6`** | GPU 正常但無溫度感測器（`temperature_support=False` 或 `temperature_c is None`） | `thermal_state == SENSOR_DEGRADED`, 含 `THERMAL_SENSOR_DEGRADED` 告警 |

### B. 核心契約與防禦測試（Unit & Branch Tests）
| 測試編號 | 驗證項目 |
| :---: | :--- |
| **`U1`** | `HardwareSnapshot` 新鮮度測試：正常新鮮、超過 `10s` TTL 判定 `STALE`（`STALE_HARDWARE_SNAPSHOT`）、單調時鐘倒退判定 `STALE`。 |
| **`U2`** | 未知欄位嚴格輸出 `None`，絕不偽填 `0`；`VRAM_admission` 下限鎖定 `0` 不得為負。 |
| **`U3`** | 無 NVIDIA 顯卡（`nvidia-smi` 不存在或回傳非零）優雅降級為 `TIER_CPU_ONLY` + `CPU_INT8`，不拋未處理例外。 |
| **`U4`** | `nvidia-smi` CSV 解析器抗污測試（`[N/A]`、`[Not Supported]`、浮點功耗、空白行、多卡僅選指定 `device_index`）。 |
| **`U5`** | `lgpu doctor --json` 輸出完全符合 `schema_version == "1.0"` 契約且可被 `json.loads()` 無損解析。 |
| **`U6`** | AST 靜態安全掃描：全倉 `src/localgpu_os/` 確認 `0` 處 `shell=True`。 |

### C. 實機驗收（Windows 11 + RTX 4060 Laptop）
- 執行 `lgpu doctor` 與 `lgpu doctor --json`，真實讀取 `NVIDIA GeForce RTX 4060 Laptop GPU`、`8188 MiB`、驅動版本、即時溫度與動態策略，Exit Code 為 `0`。
