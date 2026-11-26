# 🛠️ M2-A Engineering Design & Test Specification: Local Quality Triage (v1.1 — Approved Revisions)

> **階段編號**：`M2-A`（獨立開發分支 `feat/m2a-local-quality-triage`，保留已封存之 M1 基線 `m1-d-ocr-pdf-v1.0.0` 不動）  
> **目標模組**：`workers/ocr_pdf`（本機品質分診器 `triage.py` + 選配複核圖片原子匯出）  
> **核心定位**：讓 `ocr_pdf` 成為可靠的**本機文件品質判斷器**；雲端 Vision API 屬於上游應用程式的可選後續責任，**嚴禁納入 `ocr_pdf` Worker**。

---

## 1. 權限邊界與核心不變量 (Boundary & Invariants)

1. **離線與雲端權限邊界**：`RUNTIME_OFFLINE` 的 `ocr_pdf` Worker 絕不呼叫雲端 API、不傳送原始 PDF、不上傳複核圖片，僅產出本機分診結果與（選配）本機 `review_pages/`。本機標記 `needs_vision_review: true` **不代表取得雲端傳輸授權**。
2. **風險提示性質**：`Quality Triage` 為風險提示器，不保證高 Confidence 即零辨識錯誤。
3. **區分無文字與執行故障**：OCR 模型執行失敗或回傳格式無效屬於**執行錯誤**（直接記錄為 Job 失敗），絕不可吞掉轉成 `NO_TEXT_DETECTED`。
4. **原始結果不可覆蓋**：本機 `document.md` / `document.json` / `metadata.json` 永遠保留；任何上游雲端結果必須另存（預留 `vision_review/`），禁止覆蓋原始 OCR。

---

## 2. 雙路徑品質指標與輸入契約

### 2.1 OCR Confidence 嚴格輸入契約 (`path == "ocr"`)
1. **空白文字區塊過濾**：`text.strip() == ""` 之區塊不計入有效 `block_count`。
2. **有效數值範圍**：僅接受有限實數（排除 `bool`、`NaN`、`±Infinity`、非數值）且位於 `[0.0, 1.0]`。
3. **無效 Confidence 處理**：絕不把無效 Confidence 偽填為 `0.0`；改計入 `invalid_confidence_count` 並附加 `INVALID_CONFIDENCE_DATA` 分診原因。
4. **零分母保護**：`block_count == 0` 或有效字數總和為 `0` 時，`avg_confidence`、`char_weighted_avg_confidence`、`min_confidence` 一律輸出 `null`（絕不填 `0.0`）。

### 2.2 原生 PDF 圖片覆蓋率精確計算契約 (`path == "native"`)
- 圖片包圍盒先裁切至頁面有效範圍 `[0, page_width] × [0, page_height]`，排除零面積與無效座標。
- 採用 **精確矩形聯集演算法**（2D Sweep-line / Coordinate Compression，`image_coverage_algorithm: "exact_rect_union_v1"`），**重疊圖片區域不重複計算**，最終 `image_coverage_ratio` 嚴格落於 `[0.0, 1.0]`。
- 原生路徑絕不偽造 OCR Confidence（OCR 專用指標一律為 `null`）。

### 2.3 統一外層 `quality_triage` 結構（上游無須依路徑猜欄位）
```json
{
  "triage_version": "1.0",
  "path": "ocr",
  "needs_vision_review": true,
  "reasons": ["LOW_AVG_CONFIDENCE"],
  "metrics": {
    "avg_confidence": 0.6812,
    "avg_confidence_method": "block_equal_weight",
    "char_weighted_avg_confidence": 0.6540,
    "min_confidence": 0.4120,
    "low_conf_ratio": 0.3333,
    "block_count": 6,
    "char_count": 42,
    "invalid_confidence_count": 0,
    "image_count": null,
    "image_coverage_ratio": null,
    "image_coverage_algorithm": null,
    "vector_drawing_count": null,
    "has_mixed_text_and_image": null
  }
}
```

---

## 3. 互斥 `TriageReason` 規則

- **文字量互斥判斷**：
  ```python
  if char_count == 0:
      reasons.append("NO_TEXT_DETECTED")
  elif char_count < low_text_char_threshold:
      reasons.append("LOW_TEXT_CONTENT")
  ```
  同一頁絕不同時出現 `NO_TEXT_DETECTED` 與 `LOW_TEXT_CONTENT`。
- **完整列舉**：`NO_TEXT_DETECTED`、`LOW_TEXT_CONTENT`、`LOW_AVG_CONFIDENCE`、`HIGH_LOW_CONF_RATIO`、`INVALID_CONFIDENCE_DATA`、`IMAGE_HEAVY_PAGE`。
- `metadata.json` 中 `triage_summary.vision_review_pages` 為 **1-based、遞增排序且去重** 的頁碼清單。

---

## 4. 交易式原子輸出與 Review Image 資源限制

### 4.1 Windows 安全目錄交易替換流程（失敗不破壞舊有效輸出）
1. 所有新檔案（`document.md`、`document.json`、`metadata.json` 與選配 `review_pages/`）先寫入同目錄 `.staging` 資料夾並 `fsync`。
2. 驗證 JSON 可無損解析、非空；若啟用圖片匯出，驗證每張 PNG 非空、未超過單頁大小上限且與 `review_manifest.json` 之 SHA-256 一致。
3. 若正式目標目錄 `output/<job_id>/` 已存在舊有效輸出，先將舊目錄 `os.replace` 移至同層 `.old` 備份保留。
4. 將 `.staging` 以 `os.replace` 提交為正式目標目錄。
5. 若提交失敗，自動將 `.old` 回復為正式目標並清除 `.staging`（確保舊有效輸出完好無損）。
6. 僅在提交完全成功後，才刪除 `.old` 備份。

### 4.2 `--export-review-images` 渲染與資源契約
- **預設不匯出**：未指定 `--export-review-images` 時，絕不建立 `review_pages/` 或任何 PNG。
- **僅匯出待複核頁**：只針對 `needs_vision_review == True` 的頁面匯出，絕不渲染整份文件。
- **統一來源與固定 DPI**：PDF 統一從原始 PDF 頁面以明確 `--review-dpi`（預設 `150`）渲染；圖片檔轉存為標準 PNG。
- **逐頁串流釋放**：逐頁渲染、寫入磁碟並立即釋放 Pixmap／影像物件，不同時在 RAM 堆積多頁影像。
- **單頁大小上限與 SHA-256**：單頁 PNG 超過 `--max-review-image-mib`（預設 `15 MiB`）即報錯中止；對最終 PNG 檔案位元組計算 64 個十六進位字元的 SHA-256（256-bit digest）寫入 `review_pages/review_manifest.json`。匯出失敗時整筆工作失敗，不留半成品 `review_pages/`。

---

## 5. 測試與驗收矩陣 (`tests/test_m2a_quality_triage.py`)

| 測試編號 | 測試情境 | 驗證重點 |
| :---: | :--- | :--- |
| **`Q1`** | 清晰標準文字頁（原生 PDF & 清晰掃描 PDF） | `needs_vision_review == False`, `reasons == []`（確認零誤報）。 |
| **`Q2-A`** | 低品質／模糊掃描頁 + 合成低 Confidence 單元驗證 | 明確驗證 `avg_confidence` / `char_weighted_avg_confidence` / `low_conf_ratio` 觸發 `LOW_AVG_CONFIDENCE` / `HIGH_LOW_CONF_RATIO`。 |
| **`Q2-B`** | 少字掃描頁（如僅含 `"CHAPTER 1"`） | 明確驗證觸發 `LOW_TEXT_CONTENT` 且與 `NO_TEXT_DETECTED` 互斥。 |
| **`Q3`** | 無效 Confidence 資料（`NaN`, `Inf`, `-0.1`, `1.5`, 空白文字塊） | 絕不偽填 `0.0`，記錄 `invalid_confidence_count` 並觸發 `INVALID_CONFIDENCE_DATA`；零分母回傳 `null`。 |
| **`Q4`** | 圖表／純圖掃描頁 & 完全空白頁 | 觸發 `NO_TEXT_DETECTED`，不拋例外、不誤判為執行失敗。 |
| **`Q5`** | 原生 PDF 圖片覆蓋率（小 Logo `<10%` vs 大圖 `>=30%` vs 重疊雙圖聯集） | 驗證精確矩形聯集不重複計面積；小 Logo 不誤報，大圖觸發 `IMAGE_HEAVY_PAGE` 且 OCR Confidence 為 `null`。 |
| **`Q6`** | 選配 `--export-review-images` 與交易回復 | 預設無 `review_pages/`；啟用後僅匯出待複核頁 PNG + SHA-256 `review_manifest.json`；匯出超限失敗時保留既有有效輸出。 |
| **`Q7`** | 全倉回歸相容性 | M2-A 新增測試不得使既有 M1 基線測試失敗；全倉測試數量以 M2-A 分支實際發現結果為準。 |
