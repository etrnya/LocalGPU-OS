# `ocr_pdf` — Layered Offline OCR & PDF Worker (v1.0.0)

> `LocalGPU-OS` M1-D 第一個正式交付之隔離 Worker。  
> 嚴格隔離於 `workers/ocr_pdf/.venv`，Core OS 維持零重型 AI 依賴；預設 `RUNTIME_OFFLINE` 離線執行，並在載入模型前強制驗證 ONNX 檔案 SHA-256。

---

## 1. 處理管線 (Layered Pipeline)

1. **Document Classifier (`classifier.py`)**：
   - 原生文字 PDF → 直接走 **PyMuPDF (`pymupdf_native`)**，**完全跳過 OCR（不載入 RapidOCR 模型）**。
   - 掃描 PDF（無原生文字層）→ 逐頁渲染（預設 `200 DPI`）後交由 **RapidOCR (`rapidocr`)**。
   - 圖片（`.png`, `.jpg`, `.jpeg`, `.webp`, `.tif`, `.tiff`, `.bmp`）→ 直接交由 **RapidOCR**。
2. **Backend Audit & Pre-Load Integrity (`ocr_engine.py`)**：
   - 載入 `RapidOCR` 前強制比對 [model_checksums.json](model_checksums.json) 內鎖定的 3 顆離線 PP-OCRv4 ONNX 模型 SHA-256，不符即 Fail-Closed。
   - 實際檢查 ONNX Runtime `det` / `cls` / `rec` 三組 Session 的 `get_providers()`，誠實記錄 `cuda_verified`、`active_providers` 與 `model_precision`（`fp32_onnx_cuda` / `fp32_onnx_cpu`）。
3. **原子化輸出契約 (`output_writer.py`)**：
   - 先寫入 `.staging` 目錄、執行 `fsync` 與 JSON 完整性檢查，驗證通過後再以 `os.replace` 原子置換為正式目錄：
     ```text
     <output_dir>/<job_id>/
       document.md      # 供人閱讀（含 ## Page N）
       document.json    # 供下游 AI / RAG 使用（含頁碼、文字、bbox、confidence）
       metadata.json    # 來源、分類、Backend Audit、模型 SHA-256、耗時、網路政策
     ```
   - 中途取消、異常或強制終止時自動清除 `.staging`，**絕不覆蓋先前有效結果，也不留下半成品**。

---

## 2. 乾淨環境從頭安裝 (`INSTALL_PHASE`)

> **建議 Python 版本**：Python 3.12（Windows 11 + NVIDIA CUDA 單卡已實機驗證）。

```powershell
# 於 LocalGPU-OS 專案根目錄執行：

# 1. 建立獨立隔離虛擬環境
py -3.12 -m venv workers\ocr_pdf\.venv

# 2. 安裝鎖定版本相依套件（含 PyMuPDF、RapidOCR、ONNX Runtime GPU 與 NVIDIA CUDA 13 / cuDNN 9 Runtime wheel）
workers\ocr_pdf\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r workers\ocr_pdf\requirements.txt

# 3. 讓隔離 Worker 可直接匯入 LocalGPU-OS SDK（零額外 build 依賴）
Set-Content -Path "workers\ocr_pdf\.venv\Lib\site-packages\localgpu_os.pth" `
  -Value "$((Get-Location).Path)\src" -Encoding UTF8
```

*(註：無 NVIDIA 顯卡的純 CPU 環境可將 `requirements.txt` 中的 `onnxruntime-gpu` 與 `nvidia-*` 替換為 `onnxruntime`，執行時加上 `--allow-cpu-fallback` 即可。)*

---

## 3. 執行範例 (`RUNTIME_OFFLINE`)

```powershell
# 準備 input_manifest.json（嚴禁內嵌 Base64，所有檔案須位於 --allowed-root 內）
# {"files": [{"path": "sample.pdf"}]}

$env:PYTHONPATH="src"
python -m localgpu_os.presentation.cli run ocr_pdf `
  --input-manifest .\inputs\manifest.json `
  --allowed-root .\inputs `
  --allow-cpu-fallback `
  --json `
  -- --output-dir .\output --job-id job_001
```

---

## 4. RTX 4060 Laptop 實機基準與已知限制

- **實測環境**：Windows 11 + NVIDIA GeForce RTX 4060 Laptop GPU (`8188 MiB`, Driver `621.14`)
- **A4 200 DPI 掃描頁實測峰值 VRAM**：
  - 模型載入：`+134 MiB` → 首次推論：`+956 MiB` → 穩態峰值：**`+1468 MiB`**（`~578 ms/page`）
  - [worker.json](worker.json) 據此申請 `GPU_COMPUTE: 1 slot (exclusive)` + `GPU_MEMORY: 2048 MiB`。
- **已知限制 — Windows 程序樹範圍**：
  - `ocr_pdf` Worker 目前為**單程序**執行（PyMuPDF 與 ONNX Runtime 均在同一個 Worker 程序內運行，不額外衍生孫程序），因此 `WorkerRuntime` 的 `TerminateProcess` 強制終止可直接清除完整執行實體。
  - 若未來其他 Worker 會衍生多層子程序，將於 M2/M3 階段評估導入 Windows Job Object 進行程序樹管控。
