# 📋 LocalGPU-OS 產品需求規格書 (PRD v1.1.2 — M0 Frozen)

> **專案名稱**：`LocalGPU-OS`  
> **版本**：`v1.1.2`（**M0 正式凍結版**，已完成全部 P0 契約收斂，進入 `M1-A` 實作階段）  
> **授權協議**：Apache License 2.0（Copyright 2026 `etrnya`；第三方套件與模型見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）  
> **核心開發原則**：**核心嚴謹、功能精簡、實測優先**  
> **術語對齊**：本文件嚴格遵循 [GLOSSARY.md](GLOSSARY.md) (`v1.1.2`)

---

## 1. 產品核心定位與邊界 (Vision & Scope)

### 1.1 一句話核心定位
> **`LocalGPU-OS` 是一個與 AI 模型及應用程式無關的本機運算資源仲裁層，負責管理 GPU 資源、工作負載、執行環境及硬體安全，讓不同應用程式可以可靠地共享本機運算能力。**

### 1.2 核心邊界：只做好四件事
1. **知道本機有什麼硬體**（`HardwareProfile` 靜態能力；M1 明確限定管理單一指定 NVIDIA GPU，避免過早多卡複雜化）。
2. **知道目前還有多少可用資源**（`HardwareSnapshot` 即時觀測，感知受控與外部程序）。
3. **安全地安排工作**（三層 `VRAM-Gate`、全有或全無多資源仲裁、溫控四態與 Fail-Safe）。
4. **可靠地啟動、監控與回收 Worker**（獨立 `.venv` 解析、標準契約、7 步驟取消與異常恢復）。

### 1.3 成本與離線承諾
- **成本定位**：**零雲端 API 使用費、無強制訂閱費、預設本機運算**（本機仍存在電費、硬體折舊、磁碟空間與安裝期下載流量）。
- **網路政策區分（軟體政策 vs. 系統強制）**：
  - `INSTALL_PHASE`：允許下載相依套件與模型權重至本機快取，強制鎖定版本與 SHA-256（禁止每次啟動自動追新版）。
  - `RUNTIME_OFFLINE`（M1 預設）：Core OS 與官方 Worker 不主動建立外部網路連線、不呼叫雲端 API、不發送遙測，並記錄各 Worker 之網路政策宣告。
  - `AIRGAPPED`（後續階段）：需搭配 OS 防火牆或系統層網路隔離機制強制執行，M1 不宣稱單靠 Python 參數即可攔截第三方套件連網。

---

## 2. 核心工程契約 (Core Contracts — Frozen)

### 2.1 硬體事實 (`HardwareProfile`) 與動態觀測 (`HardwareSnapshot`) 分離
- **`HardwareProfile`（靜態能力）**：`device_id`, `vendor`, `device_name`, `architecture`, `compute_capability`, `total_vram_mib`, `memory_bandwidth_gbps`, `power_limit_w`, `nvenc_supported`, `temperature_support`, `backend_support`。**無法取得或不支援的欄位一律設為 `null` / `UNKNOWN`，嚴禁填 `0`**。
- **`HardwareSnapshot`（動態觀測）**：`device_id`, `observed_at`（UTC ISO-8601 供 JSON 輸出與日誌）, `observed_monotonic`（程序內部單調時鐘 `time.monotonic()` 計算快照年齡）, `freshness_ttl_seconds`（預設 `10s`）, `available_vram_mib`（裝置層級即時 `VRAM_free,observed`）, `external_vram_mib`（程序層級僅作輔助歸因）, `temperature_c`, `utilization_percent`, `sensor_status`。
- **新鮮度與時鐘不變量**：
  - 內部一律使用 **monotonic clock** 判斷年齡，避免系統時鐘同步造成誤判。
  - 休眠喚醒、裝置重置、採樣失敗或無法確認新鮮度時，舊快照**一律視為 `STALE`**（`STALE_HARDWARE_SNAPSHOT`），禁止沿用放行 GPU 任務。
- **策略分離**：`ExecutionPolicyTier` 僅提供描述性預設配置與狀態說明；**真正決定是否放行的是 `AdmissionControl`**。
- **平台驗證順序**：M1 主驗證平台為 **Windows 11 + 單一指定 NVIDIA CUDA GPU**，次驗證平台為 Linux + NVIDIA CUDA；Apple Silicon (MPS) 僅保留 `BackendAdapter` 介面，不納入 M1 驗收。

---

### 2.2 三層 `VRAM-Gate` 與嚴格租約交易契約

#### A. 動態入場公式（統一 `MiB`、下限歸零、杜絕重複扣除）
$$VRAM_{admission} = \max\left(0,\; VRAM_{free,observed} - VRAM_{safety} - VRAM_{pending}\right)$$
- 內部統一以 **`MiB`**（對齊 `nvidia-smi`）運算。
- `VRAM_free,observed`：直接取自裝置層級即時 Free Memory（已自然反映外部程序佔用，**絕不再重複扣除** `external_vram_mib`）。
- `VRAM_safety = system_reserve_mib + runtime_overhead_mib + uncertainty_margin_mib`。
- **保證範圍不變量**：`estimated_vram_mib` 僅為排程估算；**任何 `Admission Grant` 均不宣稱保證絕不發生 OOM**，僅代表在當下可觀測資源與既定政策下符合放行條件。

#### B. `VRAM_pending` 生命週期與「觀測確認後轉換」契約
為避免新啟動 Worker 既被算進 `VRAM_pending` 又反映在 `VRAM_free,observed` 造成重複扣除（或過早移除導致搶跑）：

| `ReservationState` | 是否計入 `VRAM_pending` | 轉換規則 |
| :--- | :---: | :--- |
| `RESERVED` | ✅ 是 | Lease 已核發，Worker 尚未啟動。 |
| `STARTING` | ✅ 是 | Worker 已啟動，持續重採樣裝置層級 Free Memory。 |
| `ACTIVE`（觀測確認後） | ❌ 否 | **僅當**重新採樣確認實際佔用已反映預留量時，才將對應 `pending` 轉為已觀測佔用；無法確認時保守保留 `pending` 或觸發逾時處理。 |
| `RELEASING` | ⚠️ 依實測 | 依實際觀測與回收政策處理。 |
| `RELEASED` | ❌ 否 | 租約已結束。 |

#### C. 租約登錄檔 Read-Modify-Write 交易原子性
- **Registry Lock 單一寫入者原則**：Lease Registry 完整「讀取 → 判斷 → 修改 → 寫入」必須全部包覆於同一個 `RegistryLock` 內。
- **狀態檔結構**：含 `schema_version`、`generation`（單調遞增）、`updated_at` 與活躍租約清單。
- **同檔案系統原子置換**：暫存檔與正式檔位於同一目錄／檔案系統，經 `fsync` 後以 `os.replace` 置換。*(註：單檔 `fsync` 保證程序崩潰一致性，跨平台突然斷電之目錄項目持久化不作為 M1 強保證。)*
- **跨平台獨立驗證**：Windows `msvcrt.locking` 與 POSIX `fcntl.flock` 分別獨立驗證。

#### D. 防誤回收與 Fencing Token (`lease_generation`)
- **Heartbeat 逾時 ≠ Worker 已死亡**：回收前必須交叉驗證 `pid`、`pid_create_time` 與 OS 程序狀態；無法確認原 Worker 已停止前，**嚴禁重新核發衝突資源租約**（`LEASE_RECLAIM_BLOCKED`）。
- **Fencing Token (`lease_generation`)**：每次核發租約配發單調遞增世代號，Worker 回寫心跳或釋放租約須帶入驗證（不符回傳 `FENCING_TOKEN_MISMATCH`）。*(保護受控狀態，無法使已送進 GPU 的 CUDA kernel 失效。)*

#### E. 抽象化外部服務卸載 (`EvictionProvider`)
- Core OS 定義抽象介面 `EvictionProvider`（能力探測、卸載前狀態紀錄、逾時、失敗處理與**卸載後重採樣 `VRAM_free,observed` 複驗**）；`OllamaEvictionProvider` 為第一個實作。

---

### 2.3 多資源全有或全無原子性 & 相容性矩陣

- **全有或全無原則 (All-or-Nothing Admission)**：同一個任務申請清單 (`requests`) 中所有 `ResourceClass` 視為單一交易。所有資源均符合條件才核發 Lease；任一資源不符時，整筆拒絕且**不得留下任何部分有效租約**。
- **Core OS 政策最高原則**：Worker Manifest 所宣告之 `exclusive` 或 `concurrency_limit` **不得放寬 Core OS 的資源相容性政策**；最終互斥與併發規則以 Core OS Resource Policy 為準。

| 任務 A \ 任務 B | `GPU_COMPUTE` | `GPU_MEMORY` | `GPU_ENCODE` (M3) | `CPU_COMPUTE` | `SYSTEM_MEMORY` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`GPU_COMPUTE`** | ❌ 互斥 (M1) | ⚠️ 共享容量配額 | ⚠️ 條件並行 (M3) | ✅ 允許並行 | ⚠️ 受 RAM 配額限制 |
| **`GPU_MEMORY`** | ⚠️ 共享容量配額 | ⚠️ 共享容量配額 | ⚠️ 共享容量配額 | ✅ 允許並行 | ⚠️ 受 RAM 配額限制 |
| **`GPU_ENCODE` (M3)** | ⚠️ 條件並行 | ⚠️ 共享容量配額 | ⚠️ 依 NVENC Session | ✅ 允許並行 | ⚠️ 受 RAM 配額限制 |
| **`CPU_COMPUTE`** | ✅ 允許並行 | ✅ 允許並行 | ✅ 允許並行 | ⚠️ 依 CPU 配額 | ⚠️ 受 RAM 配額限制 |
| **`SYSTEM_MEMORY`** | ⚠️ 受 RAM 配額限制 | ⚠️ 受 RAM 配額限制 | ⚠️ 受 RAM 配額限制 | ⚠️ 受 RAM 配額限制 | ⚠️ 共享容量配額 |

- M1-B 實作 `GPU_COMPUTE` + `GPU_MEMORY` + `CPU_COMPUTE`（及 `SYSTEM_MEMORY` 基礎檢查）；`GPU_ENCODE` 保留介面與契約於 M3 啟用。

---

### 2.4 `ThermalGuard` 四態狀態機
`ThermalGuard` 調節合作式 Worker 排程節奏，**不取代** NVIDIA 驅動與筆電 BIOS 原生保護（保留未來擴充 CPU/Hotspot/電池感測器介面，M1 僅實作 GPU 溫度）。

| 狀態 (`ThermalState`) | 觸發條件 | 行為與恢復條件 |
| :--- | :--- | :--- |
| **`NORMAL`** | 感測器正常，溫度低於警戒值（參考 `82°C`） | 正常執行。 |
| **`COOLING`** | 超過警戒值 | 節流／拉長間隔／暫停新任務；須低於恢復值（參考 `72°C`）且滿足遲滯時間（預設 `15s`）回 `NORMAL`。 |
| **`CRITICAL`** | 超過緊急門檻（參考 `88°C`） | 停止新任務並通知安全中止；須降至恢復值以下且連續穩定達冷卻遲滯時間（預設 `60s`）方可重開核發。 |
| **`SENSOR_DEGRADED`** | 感測器不支援、讀取失敗或快照過期 | 保守模式，禁止高負載新 GPU 任務；須連續成功採樣有效且新鮮溫度讀值方可恢復。 |

---

### 2.5 `Worker Manifest Contract v1.0` 與 7 步驟跨平台取消協議
- **嚴格 `.venv` 解析**：Manifest 僅宣告 `module` 與 `args`（不依賴 `PATH`），由 Runtime 解析 `workers/<id>/.venv/Scripts/python.exe`（Windows）或 `workers/<id>/.venv/bin/python`（Linux），全倉 `0 shell=True`。
- **降級與大檔邊界**：明確宣告 `fallback_policy`（`cpu_supported`, `cpu_backend`, `max_cpu_threads`, `max_system_memory_mib`）；`json_file_manifest` 嚴禁內嵌 Base64 檔案本體，強制根目錄限制、檔案大小上限、`DISK_SPACE` 檢查與 Path Traversal 攔截。
- **跨平台統一取消協議（完整 7 步驟）**：
  1. Core OS 寫入原子取消控制檔（固定 JSON 格式、輪詢間隔 `≤ 1s` 與競態處理）。
  2. Worker 偵測取消通知，停止接收新工作。
  3. Worker 完成允許的清理工作並刪除暫存檔。
  4. Worker 確認 GPU 工作已停止，攜帶 `lease_generation` 釋放 Lease。
  5. Worker 寫入 `WORKER_CANCELLED` 狀態。
  6. Worker 正常退出，Core OS 確認程序已終止。
  7. 超過 `cancel_grace_seconds` 仍未退出時，由 Core OS 強制終止（`TerminateProcess` / `SIGKILL`）、執行清理並**重新採樣 VRAM 確認資源狀態後**，方可核發下一個 GPU 任務。

---

### 2.6 Fail-Safe 失效安全資源管理矩陣
| 異常情境 | 預期行為 |
| :--- | :--- |
| GPU / VRAM 讀取失敗 | Fail-Closed 拒絕 GPU 任務 (`HARDWARE_UNVERIFIED`)；允許純 `CPU_COMPUTE` 任務。 |
| `HardwareSnapshot` 過期或時鐘不可信 | 視為 `STALE`，拒絕 GPU 放行 (`STALE_HARDWARE_SNAPSHOT`)。 |
| Lease Registry 狀態檔損壞 | Fail-Closed (`LEASE_STATE_CORRUPTED`)。 |
| Worker Heartbeat 逾時但程序存活 | 禁止誤回收，回傳 `LEASE_RECLAIM_BLOCKED` 並嘗試隔離停止。 |
| **Core OS 意外崩潰** | OS 釋放 `RegistryLock`，但 **Lease 不得僅因 TTL 逾時即回收**；重啟時須重新驗證 Worker `pid`、`pid_create_time` 與程序狀態，若仍存活須先確認停止或隔離，否則保留佔用並拒絕衝突 Admission。 |
| `EvictionProvider` 卸載失敗 | 絕不假設已釋放；重採樣複驗未達標即回傳 `EVICTION_FAILED`。 |
| 溫度感測器失敗 | 進入 `SENSOR_DEGRADED`，禁止高負載新 GPU 任務。 |
| 磁碟空間不足 (M1-C+) | 拒絕啟動任務 (`DISK_SPACE_INSUFFICIENT`)。 |

---

## 3. 分期里程碑與量化驗收標準 (Roadmap & Acceptance)

- **M0（已凍結）**：`LICENSE`、`GLOSSARY.md` (`v1.1.2`)、`PRD.md` (`v1.1.2`)、`THIRD_PARTY_NOTICES.md`、`README.md`、`README_EN.md`。
- **M1-A：Hardware Foundation**（詳見 [docs/M1-A_DESIGN_AND_TEST_SPEC.md](docs/M1-A_DESIGN_AND_TEST_SPEC.md)）
  - **只做**：`HardwareProfile`、`HardwareSnapshot`、`HardwareDetector`（Windows NVIDIA）、`PolicyEngine`、`lgpu doctor`。
  - **不做**：`VramLease`、`RegistryLock`、排隊、Ollama 卸載、Worker 啟動與生命週期管理。
  - **驗收**：6 大邊界情境 Scenario Coverage 100% 通過 + 核心 `PolicyEngine` 高分支覆蓋率 + RTX 4060 實機 `lgpu doctor` 驗收通過。
- **M1-B：Resource Arbitration**：`RegistryLock` + 原子 RMW Lease Registry、`VRAM_pending` 觀測轉換、全有或全無多資源 `AdmissionControl`、`EvictionProvider` (`OllamaEvictionProvider`)、`ThermalGuard` 四態、`lgpu status`。
- **M1-C：Worker Runtime**：`WorkerRegistry`、`WorkerManifest` v1.0 驗證、`.venv` 解析、7 步驟取消協議、`DISK_SPACE` 檢查、`lgpu run` CLI ＋ Python SDK。
- **M1-D：分層式離線 OCR / PDF Worker (`ocr_pdf`)**：Document Classifier（原生 PDF 走 `PyMuPDF` 跳過 OCR）→ 預設 `RapidOCR` → 第二階段選配 `Docling`。正式測試集無 OOM、超限正確拒絕/降級、失敗隔離與租約回收、記錄前後 VRAM 峰值。
- **M2 ~ M4**：M2 離線 LLM & Embedding Worker + 本機安全 HTTP API（`127.0.0.1` + Origin/CSRF + Token + 白名單）；M3 NVENC (`GPU_ENCODE`) + 離線語音；M4 評估 MCP Server、監控頁與多語系 README。
