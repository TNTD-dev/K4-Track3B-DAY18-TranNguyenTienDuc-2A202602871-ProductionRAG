# Đối chiếu ASSIGNMENT / RUBRIC

Đây là checklist bằng chứng, không phải điểm do giảng viên chấm. Các kết quả thực nghiệm được giữ nguyên, kể cả câu sai và metric giảm.

| Hạng mục | Triển khai / bằng chứng | Trạng thái |
|---|---|---|
| M1 | Semantic cosine; hierarchy parent/child; structure giữ headers, bảng, fenced code. `reports/chunking_comparison.json`, tests M1 + regressions | Hoàn thiện |
| M2 | underthesea, BM25, BGE-M3, Qdrant `query_points`, RRF đúng rank và identity | Hoàn thiện |
| M3 | BGE reranker-v2-m3, logits xếp giảm dần; empty/scalar handling; benchmark | Hoàn thiện |
| M4 | RAGAS 4 metrics, 20 rows; fail closed khi evaluator lỗi; bottom-5 và Diagnostic Tree có bằng chứng nguồn | Hoàn thiện |
| M5 | Summary, HyQA, contextual, metadata; combined 1 request/chunk, cache và trạng thái fallback | Hoàn thiện |
| Pipeline | Retrieve child → rerank nguồn gốc → 3 parent khác nhau → answer có nguồn; baseline và production đo với Qdrant server | Chạy thực tế |
| ≥3 metrics ≥0,70 | 4/4: 0,9100 / 0,7384 / 0,9500 / 0,9333 | Đạt ngưỡng thực đo |
| Failure insight | Phân biệt lỗi tròn tháng, thiếu tổng hợp cấp dữ liệu, evaluator disagreement và reference/version ambiguity | Có phân tích thủ công |
| Reflection | Mapping 5 modules, debugging có sự kiện cụ thể, project plan cho chính trợ lý chính sách trong lab | Hoàn thiện |
| Bonus Faithfulness ≥0,85 | 0,9100 | Đạt ngưỡng thực đo |
| Bonus mọi metric ≥0,75 | Answer Relevancy 0,7384 | Chưa đạt |
| Bonus combined | `_enrich_single_call()`; regression đảm bảo một request và provenance | Có bằng chứng |
| Bonus latency | `reports/latency_report.json` và bảng build/query/evaluation trong failure analysis | Có bằng chứng |

## Giới hạn cần trình bày khi bảo vệ

- Hai PDF scan chưa OCR; pipeline không tuyên bố đã đọc nội dung ảnh.
- Enrichment trong lượt đo dùng cache: 15 ms không đại diện chi phí cold LLM enrichment.
- Parent/child production là 26/100 khi chunk theo từng tài liệu; phép so sánh concatenated corpus là thực nghiệm riêng.
- Answer Relevancy dùng prompt Việt với generic examples; hiệu chỉnh đã thấy tập lab nên cần holdout độc lập.
- Một bài tính phí quá hạn vẫn trả lời sai reference. Phải khai báo giả định pro-rata và xác nhận convention với Tài chính.
- Baseline được chấm lại trên đáp án/context cố định sau lỗi evaluator. Không suy luận nguyên nhân lỗi chỉ từ việc rerun thành công.
- Model manifests có revision, dung lượng và SHA-256 hai weight BGE đúng đề. Bài làm được chuyển sang fork đúng Track 3B; chưa nộp link vào LMS.
