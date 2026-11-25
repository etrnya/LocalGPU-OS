# ⚖️ THIRD-PARTY NOTICES — LocalGPU-OS 第三方套件與模型授權清單

本專案 (`LocalGPU-OS`) 自身原始碼採用 [Apache License 2.0](LICENSE)（Copyright 2026 `etrnya`）。

> **重要聲明**：Apache License 2.0 **僅適用於本儲存庫自行撰寫之原始碼**，不自動涵蓋使用者於安裝期 (`INSTALL_PHASE`) 透過隔離 Worker 自行下載之第三方 Python 套件、系統二進位檔（如 FFmpeg）或 AI 模型權重。各第三方組件均受其原始授權條款約束；商用或再散佈前請務必逐一核對下表。

---

## 1. Core OS 相依套件（M1-A ~ M1-C 實作後鎖定版本並補齊）

| 套件名稱 | 用途 | 授權協議 | 備註 |
| :--- | :--- | :--- | :--- |
| Python Standard Library | 核心排程、跨程序檔案鎖、子程序管理 | PSF License | 隨 Python 3.10+ 內建 |
| *(待 M1-A 鎖定依賴後登記)* | — | — | 嚴禁引入 PyTorch / Paddle / TensorFlow |

---

## 2. 隔離 Workers 相依套件與外部執行檔（規劃追蹤表）

| 階段 | 模組 / 工具 | 類型 | 常見授權參考 | 商用與再散佈注意事項 |
| :--- | :--- | :--- | :--- | :--- |
| M1-D | `PyMuPDF` (`fitz`) | Python 套件 | **AGPL-3.0** / 商業授權 | ⚠️ **注意**：AGPL-3.0 具強網路傳染性；商業閉源發佈或網路服務需評估授權或改用 `pypdf` / `pdfium` (`Apache-2.0`/`BSD`)。 |
| M1-D | `rapidocr-onnxruntime` | Python 套件 + 模型 | Apache-2.0（套件）；內建模型權重依 PaddleOCR 原始授權 | 須核對具體下載版本之模型權重條款。 |
| M1-D (選配) | `Docling` | Python 套件 + 模型 | MIT（套件）；模型權重各自獨立授權 | 僅作為選配隔離 Worker，不隨 Core OS 散佈。 |
| M2 | `Ollama` | 外部服務 | MIT（引擎）；各下載 LLM 權重依原模型授權（Apache-2.0 / Qwen / Gemma 等） | 透過 HTTP 介面外部協作，不打包散佈。 |
| M3 | `FFmpeg` | 外部執行檔 | LGPL-2.1+ 或 GPL-2.0+（視編譯參數如 `--enable-gpl` / `libx264`） | 不隨本專案打包，由使用者依自身環境安裝。 |

---

## 3. 模型權重治理原則
1. 本儲存庫**絕不提交、不託管、不再散佈**任何第三方模型權重檔（`.onnx`, `.gguf`, `.safetensors`, `.bin`, `.pt`）。
2. 所有隔離 Worker 須附帶版本鎖定檔（如 `requirements.lock` / `uv.lock`），並於執行期預設採 `RUNTIME_OFFLINE` 模式。
