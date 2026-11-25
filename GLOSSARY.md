# 📖 GLOSSARY — LocalGPU-OS 專案術語字典 (v1.1.2 — M0 Frozen)

> 本文件遵循「字典先行 (Dictionary First)」原則，定義 `LocalGPU-OS` 全專案之標準中英文術語、程式碼變數命名、列舉值與架構契約。後續所有程式碼、測試與文件必須嚴格對齊本字典。

---

## 1. 核心系統與仲裁術語 (Core Arbitration)

| 中文術語 | 英文術語 | 標準程式碼識別符 | 定義與職責 |
| :--- | :--- | :--- | :--- |
| 本機運算資源仲裁層 | Local Compute Resource Arbitration Layer | `localgpu_os` | 與 AI 模型及應用程式無關的本機資源管理與排程層。 |
| 核心底座 | Core OS | `localgpu_os.core` | 零重型 AI 依賴（嚴禁安裝 PyTorch / Paddle / TensorFlow）的輕量仲裁核心。 |
| 三層顯存閘門 | Three-Layer VRAM-Gate | `VramGate` | 管理受控 Worker 的資源使用，並對外部非受控程序進行資源感知與風險評估（Cooperative Lock → External Monitoring → Admission Control）。 |
| 租約登錄鎖 | Lease Registry Lock | `RegistryLock` | 保護 Lease Registry 完整 **Read-Modify-Write** 交易原子性之單一寫入者鎖。 |
| 合作式資源租約 | Cooperative Resource Lease | `VramLease` | 受控 Worker 執行 GPU 任務前必須取得之有時效租約，含 `lease_id`、`lease_generation`、`reservation_state`、`ttl_seconds`、`heartbeat_at`、`pid`、`pid_create_time`。 |
| 預留狀態 | Reservation State | `ReservationState` | 控制是否計入 `VRAM_pending`：`RESERVED`（計入）、`STARTING`（計入）、`ACTIVE`（須經實際 VRAM 觀測確認後方移出）、`RELEASING`（依實際觀測與回收政策）、`RELEASED`（不計入）。 |
| 隔離世代權杖 | Fencing Token / Lease Generation | `lease_generation` | 每次核發租約單調遞增之世代編號，防止過期 Worker 回寫污染新租約狀態（僅保護受控狀態，無法使已提交至 GPU 的 CUDA kernel 失效）。 |
| 卸載提供者介面 | Eviction Provider Interface | `EvictionProvider` | 外部常駐 GPU 服務之抽象卸載介面（`OllamaEvictionProvider` 為第一個實作），含能力探測、狀態紀錄、逾時與事後 VRAM 複驗。 |
| 失效安全資源管理 | Fail-Safe Resource Management | `FailSafePolicy` | 資訊不足或異常時對 GPU 任務採 **Fail-Closed**；純 `CPU_COMPUTE` 任務可依資源類型獨立安全降級。 |

---

## 2. 硬體事實、動態觀測與執行策略分離

| 模型名稱 | 標準識別符 | 性質與核心欄位 |
| :--- | :--- | :--- |
| **靜態硬體能力** | `HardwareProfile` | 物理能力（M1 限定單一指定 NVIDIA GPU）；不支援或無法取得欄位一律為 `null` / `UNKNOWN`，**絕不填 `0`**。欄位：`device_id`, `vendor`, `device_name`, `architecture`, `compute_capability`, `total_vram_mib`, `memory_bandwidth_gbps`, `power_limit_w`, `nvenc_supported`, `temperature_support`, `backend_support`。 |
| **動態硬體快照** | `HardwareSnapshot` | 具時效性之單點觀測：`device_id`, `observed_at`（UTC ISO-8601 供紀錄）, `observed_monotonic`（程序內部單調時鐘計算年齡）, `freshness_ttl_seconds`, `available_vram_mib`（裝置層級即時 `VRAM_free,observed`）, `external_vram_mib`（僅供輔助歸因）, `temperature_c`, `utilization_percent`, `sensor_status`。休眠喚醒、裝置重置、採樣失敗或無法確認新鮮度時一律視為 `STALE`。 |
| **動態執行策略** | `ExecutionPolicyTier` | 由 `PolicyEngine` 產出的描述性預設配置（`TIER_CPU_ONLY`, `TIER_LITE`, `TIER_STANDARD`, `TIER_PRO`, `TIER_ULTRA`）；**真正決定是否放行的是 `AdmissionControl`**。 |

### 動態入場顯存公式（統一以 `MiB` 計算、下限歸零、杜絕重複扣除）
`VRAM_admission = max(0, VRAM_free,observed - VRAM_safety - VRAM_pending)`
- 內部統一以 **`MiB`**（$1\text{ MiB} = 1024^2\text{ Bytes}$，對齊 `nvidia-smi`）運算與儲存；JSON 欄位名稱明確標註 `_mib`。
- `VRAM_free,observed`：裝置層級即時可用顯存（已自然反映外部程序佔用，絕不再重複減去 `external_vram_mib`）。
- `VRAM_safety`：`system_reserve_mib + runtime_overhead_mib + uncertainty_margin_mib`。
- `VRAM_pending`：處於 `RESERVED` / `STARTING` 或尚未經裝置層級觀測確認之預留顯存總和。

---

## 3. 資源分類與配置契約 (Resource Classes & Allocation)

| 列舉值 (`ResourceClass`) | 單位 (`unit`) | M1-B 實作狀態 | 管理對象 |
| :--- | :--- | :---: | :--- |
| `GPU_COMPUTE` | `slot`（M1 保守互斥） | ✅ 實作 | PyTorch、OCR、LLM、Whisper CUDA 運算 |
| `GPU_MEMORY` | `MiB` | ✅ 實作 | VRAM 容量配額與 `VRAM_admission` 審查 |
| `CPU_COMPUTE` | `threads` / `concurrency` | ✅ 實作 | 原生 PDF 抽取、CPU int8 推論、前處理 |
| `GPU_ENCODE` | `session` / `capacity` | 🔖 介面預留 (M3 啟用) | NVENC（AV1 / H.265 / H.264）硬體編碼 |
| `SYSTEM_MEMORY` | `MiB` | 🔖 基礎檢查 | RAM 預留量與 CPU 降級記憶體上限 |

- **全有或全無原子申請**：同一筆 `ResourceRequest` 清單中的所有資源類別視為單一交易；任一資源不合規即整筆拒絕，不留部分有效租約。
- **Core OS 政策最高原則**：Worker Manifest 不得放寬 Core OS 資源相容性與互斥規則。

---

## 4. 溫控四態狀態機 (`ThermalState`)

| 狀態 | 觸發條件 | 行為與恢復條件 |
| :--- | :--- | :--- |
| `NORMAL` | 感測器正常且溫度低於警戒值 | 正常核發與執行任務。 |
| `COOLING` | 溫度超過警戒門檻（參考 `82°C`） | 節流／拉長間隔／暫停新任務；須低於恢復門檻（參考 `72°C`）且滿足遲滯時間才回 `NORMAL`。 |
| `CRITICAL` | 溫度超過緊急門檻（參考 `88°C`） | 停止新任務並通知安全中止；須降至恢復門檻以下且連續穩定超過冷卻遲滯時間（預設 `60s`）方可重新核發。 |
| `SENSOR_DEGRADED` | 感測器不支援、讀取失敗或快照過期 | 保守模式，禁止高負載新 GPU 任務；須連續成功採樣到有效且新鮮之溫度讀值方可恢復。 |

---

## 5. 網路政策與標準錯誤碼

### 5.1 網路政策 (`NetworkPolicy`)
- `INSTALL_PHASE`：安裝與模型初始化階段，允許下載套件與模型並鎖定版本／SHA-256。
- `RUNTIME_OFFLINE`（M1 預設）：核心與官方 Worker 不主動呼叫雲端 API、不發送遙測（軟體政策層級）。
- `AIRGAPPED`（後續階段）：要求系統環境具備 OS 防火牆或實體網路隔離之外部強制機制。

### 5.2 標準錯誤碼 (`ErrorCode`)
| 錯誤碼 | 語意說明 |
| :--- | :--- |
| `VRAM_LOCK_TIMEOUT` | 等待跨程序鎖或租約逾時。 |
| `VRAM_INSUFFICIENT` | `VRAM_admission` 低於任務最低需求。 |
| `STALE_HARDWARE_SNAPSHOT` | `HardwareSnapshot` 超過新鮮度門檻或無法驗證新鮮度。 |
| `LEASE_STATE_CORRUPTED` | Lease Registry 狀態檔損壞（觸發 Fail-Closed）。 |
| `FENCING_TOKEN_MISMATCH` | Worker 提交之 `lease_generation` 已過期，拒絕寫入狀態。 |
| `LEASE_RECLAIM_BLOCKED` | 原 Worker 無法確認已停止，禁止重發衝突資源租約。 |
| `HARDWARE_UNVERIFIED` | GPU 或裝置層級 VRAM 讀值失敗，禁止放行 GPU 任務。 |
| `EVICTION_FAILED` | `EvictionProvider` 卸載失敗或複驗後 VRAM 未達標。 |
| `THERMAL_COOLING_WAIT` | 處於 `COOLING` 狀態暫緩核發。 |
| `THERMAL_CRITICAL_ABORT` | 處於 `CRITICAL` 狀態拒絕或安全中止任務。 |
| `THERMAL_SENSOR_DEGRADED` | 處於 `SENSOR_DEGRADED` 狀態，拒絕高負載 GPU 任務。 |
| `DISK_SPACE_INSUFFICIENT` | 工作目錄或暫存區可用磁碟空間不足。 |
| `WORKER_ENV_NOT_INSTALLED` | 目標 Worker 的 `.venv` Python 執行檔不存在。 |
| `WORKER_MANIFEST_INVALID` | `worker.json` 不符合 `Worker Manifest Contract v1.0`。 |
| `WORKER_CANCELLED` | Worker 已依取消協議完成清理並終止。 |
| `WORKER_EXECUTION_FAILED` | Worker 回傳非零 Exit Code 或逾時中止。 |
