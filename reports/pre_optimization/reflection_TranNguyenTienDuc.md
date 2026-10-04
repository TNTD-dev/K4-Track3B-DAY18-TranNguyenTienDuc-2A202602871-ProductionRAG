# Reflection — Production RAG Pipeline

**Học viên:** Trần Nguyễn Tiến Đức · **MSSV:** 2A202602871 · K4 Track 3B
**Ngày triển khai:** 04/10/2026 (GMT+7)
Ghi chép dựa trên code và các lần chạy thực nghiệm trong repository này.

## 1. Lecture → code

| Concept | Module / hàm | Quan sát và quyết định |
|---|---|---|
| Semantic chunking | M1 `chunk_semantic()` | Tách câu, embed MiniLM một lần theo batch, normalize rồi so cosine giữa hai câu kề nhau. Trên phép so sánh nối corpus của `compare_strategies()`, threshold 0,85 cho 208 semantic chunks (trung bình 99 ký tự) so với basic 51 chunks (trung bình 410 ký tự). Ngưỡng cao tạo nhiều nhóm hơn; MiniLM không chuyên tiếng Việt, cần A/B với embedding đa ngôn ngữ trước khi chọn cho production. Model được cache trong process. |
| Hierarchical / structure-aware | M1 `chunk_hierarchical()`, `chunk_structure_aware()` | Pipeline chunk từng tài liệu: corpus đọc được 26 tài liệu → 26 parent và 100 child ở giới hạn 2048/256 ký tự. Phép compare nối corpus cho 90 child và 106 structure chunks, không tương đương chunk theo từng nguồn; dùng số pipeline khi báo cáo production. Giới hạn áp dụng cả paragraph dài; ID theo nguồn và nội dung. Parser không coi header trong fenced code là section. |
| BM25 + Dense fusion | M2 `segment_vietnamese()`, `reciprocal_rank_fusion()` | Chuẩn hóa lowercase, bỏ dấu gạch dưới segmentation, tokenize punctuation. RRF cộng `1/(60+rank+1)` theo danh tính chunk, không cộng trực tiếp score BM25 với cosine. Hai nguồn có text giống nhau vẫn được giữ riêng; cùng chunk không được tăng điểm hai lần trong một ranked list. |
| Cross-encoder | M3 `CrossEncoderReranker.rerank()`; pipeline `run_query()` | BGE reranker đọc cặp query–nội dung gốc, không coi summary sinh tự động là bằng chứng. Rerank candidates rồi chọn 3 parent khác nhau; tránh nhiều child cùng parent chiếm hết context budget. |
| RAGAS 4 metrics | M4 `evaluate_ragas()`, `failure_analysis()` | Judge gpt-4o-mini, embedding text-embedding-3-small; lưu từng câu hỏi, aggregate và trạng thái. Ground truth chỉ dùng trong đánh giá. Nếu judge lỗi, scores_valid=false; không kết luận pipeline tốt/xấu từ zero fallback. Diagnosis từ metric là giả thuyết, cần đọc context và câu trả lời để xác nhận. |
| Contextual embeddings / HyQA | M5 `_enrich_single_call()`, `enrich_chunks()` | 1 request tạo summary, questions, context, metadata. Index enrichment, rerank và generation dùng nguồn gốc. Cache theo source + content + phiên bản prompt, concurrency giới hạn 4; metadata do LLM sinh không được ghi đè ID/source gốc. |

## 2. Khó khăn và debug

### Python mặc định khác virtualenv

`python3 --version` trả `Python 3.9.6`, còn `.venv/bin/python --version` trả
`Python 3.11.16`. Dùng interpreter trong virtualenv cho mọi lần chạy RAGAS và pytest.
Không suy ra phiên bản Python từ tên shell hoặc trạng thái activate.

### Lỗi f-string trên Python 3.11

Exact error khi compile bản đầu của prompt:

```text
SyntaxError: f-string expression part cannot include a backslash
```

Nguyên nhân là đặt biểu thức `'\n\n'.join(contexts)` trực tiếp trong f-string.
Tách thành biến `context_text`, chạy lại `py_compile`, rồi ruff. Đây là lỗi cú pháp
trong quá trình triển khai, không phải lỗi chất lượng RAG.

### Model download và giới hạn môi trường

Kết nối tải thử BGE trả HTTP 206 nhưng chỉ khoảng 114 KB/s cho một request 1 MiB.
Tải lần đầu các weight khoảng 2,27 GB/model chậm hơn thời lượng inference.
Bật `HF_HUB_DISABLE_XET=1`; dùng download nhiều range và xác minh SHA-256 từ metadata
Hugging Face trước khi cache. Không thay BGE bằng model nhỏ rồi gọi đó là cùng thí nghiệm.

### Scaffold che khuất lỗi bằng test yếu

Một số test gốc có `if result:` nên kết quả rỗng vẫn qua test kiểu/sorted.
Bổ sung regression test kiểm tra dense index/search thật bằng Qdrant in-memory,
parent expansion, identity collision, code fence, scalar reranker và report thiếu key.
Script nộp cũ chỉ đếm file và không thất bại khi pytest lỗi; phiên bản hiện tại kiểm tra
20 samples, scores_valid và exit code của pytest/lint.

### Judge tiếng Việt và lỗi prompt adaptation

Một mẫu kiểm tra với đáp án đúng, giữ nguyên input, cho Answer Relevancy khoảng
0,596 với prompt mặc định tiếng Anh và gần 1,00 khi sinh câu hỏi tiếng Việt.
Đây là kiểm tra riêng, không phải điểm của 20 câu trong bài nộp.
RAGAS 0.1.x `adapt()` lưu output example dưới dạng fenced JSON string; lần nạp
cache gặp exact error:

```text
ValidationError: 1 validation error for Prompt
__root__
  output in example 1 is not in valid json format: Expecting value: line 1 column 1 (char 0)
```

Đọc implementation `Prompt.adapt()` và parser để tìm nguyên nhân. Sửa bằng
bản dịch các ví dụ chung của RAGAS (Einstein, ếch, Everest, điện thoại), giữ
`question`/`noncommittal` và công thức cosine + penalty. Không đưa câu hỏi lab
vào các ví dụ evaluator. Cùng evaluator này được dùng cho cả hai pipeline;
lưu cấu hình ở `reports/evaluator_prompt_vi.json`.

### Phủ định rõ ràng khác câu trả lời không chắc chắn

Ở lần baseline đầu, câu trả lời đúng rằng nhân viên **không nên tự xử lý malware**
bị Answer Relevancy chấm 0. Diagnostic rerun giữ nguyên câu trả lời/context, không
truyền question hoặc ground truth vào judge, cho cả 3 outputs của prompt cũ
`noncommittal=1`; prompt làm rõ phủ định cho cả 3 outputs `noncommittal=0`.
Đây là rerun, không phải outputs được giữ lại từ lượt evaluation đầu.

Thêm một ví dụ chung về thư viện đóng cửa Chủ nhật và hướng dẫn phân biệt
phủ định trực tiếp với sự né tránh/không biết. Không thêm đáp án của lab vào
few-shot evaluator; giữ nguyên công thức cosine và penalty. Cả hai pipeline
được chạy lại cùng cấu hình. `reports/negation_diagnostic.json` lưu bằng chứng;
`reports/initial_*` giữ lượt đầu. Prompt được hiệu chỉnh sau khi quan sát lỗi trên
tập lab, nên cần holdout để đánh giá hiệu quả ngoài 20 câu cố định.

### Download tự động ngoài dự kiến và evaluation timeout

Lượt production đầu hoàn tất 20 answers nhưng RAGAS báo `TimeoutError`.
Cùng lúc đó, source `transformers/modeling_utils.py` khởi tạo thread
`Thread-auto_conversion` sau khi load BGE-M3 `.bin` và tải thêm safetensors từ
`refs/pr/130` ngoài các weight đã xác minh. Thread không có `daemon=True`, làm
process vẫn chờ dù bước chính đã kết thúc.

Đặt `model_kwargs={"use_safetensors": False}` riêng cho embedding BGE-M3 để
sử dụng weight `.bin` chính thức đã tải; cross-encoder vẫn dùng safetensors của
chính model reranker. Giảm RAGAS concurrency xuống 2, đặt timeout ở client và
RunConfig, rồi chạy lại main. Không kết luận download nền là nguyên nhân duy
nhất của timeout: lỗi mạng/API cũng có thể góp phần. Báo cáo timeout được giữ
và gắn `scores_valid=false`, không dùng để tính điểm.

### Kiến thức cần bổ sung

- Phân biệt API Qdrant `query_points` với search API cũ; kích thước collection lấy
  từ vector thật, tránh mismatch khi đổi encoder.
- RAGAS 0.1.x dùng Hugging Face Dataset; đọc source/signature của bản cài trong
  virtualenv, không trộn schema của RAGAS mới vào lab cũ.
- Metric recall thấp khác faithfulness thấp: cải thiện retrieval không tự động
  giải quyết generation và ngược lại.
- PDF scan không có text layer: OCR là bước ingestion riêng. Hai PDF scan bị bỏ
  qua có cảnh báo, nên chưa thể dùng hệ thống để trả lời nội dung hai file này.

## 3. Action plan — Trợ lý tra cứu chính sách nội bộ

### Hiện trạng

Project chính là corpus HR, CNTT, tài chính và quy trình của bài lab. Baseline
paragraph+dense không có lexical fusion, reranking hay enrichment. Corpus có
chính sách nhiều phiên bản, phủ định, ngưỡng tiền và câu hỏi multi-hop; cần giữ
đúng đối tượng và phiên bản, không chỉ tìm đoạn có cùng từ khóa.

### Kế hoạch áp dụng

1. **Chunking:** dùng parent–child cho tra cứu; thêm section/title và effective-date
   vào metadata. A/B structure-aware cho bảng lương/phê duyệt. Kích thước lab
   tính theo ký tự; khi mở rộng cần chuyển sang token budget của model.
2. **Search:** BM25 tiếng Việt + BGE-M3 + RRF. Giữ policy cũ để trả lời câu hỏi lịch
   sử; metadata filter phải xét ngày hỏi/ngày hiệu lực và quan hệ supersedes,
   tránh xóa toàn bộ tài liệu cũ. Thêm truy hồi từng sub-question cho multi-hop.
3. **Reranking:** BGE reranker; đo warm p50/p95 và chất lượng trước khi cân nhắc
   Flashrank. Chọn parent distinct sau rerank, không cộng dồn điểm theo số child.
4. **Evaluation:** giữ 20 câu cố định làm regression, tạo thêm bộ holdout theo
   lookup/version/negation/multi-hop/numeric/ambiguous. Đo 4 RAGAS metrics cùng
   source hit-rate, độ đúng số học, chi phí/query, p95. Không đưa ground truth
   vào prompt trả lời để tăng điểm.
5. **Enrichment:** combined + content cache. Kiểm soát tính đúng summary/HyQA bằng
   audit mẫu; văn bản sinh tự động chỉ hỗ trợ retrieval. OCR hai PDF scan và
   kiểm tra độ đúng trước khi đưa vào embedding.

### Timeline và acceptance criteria

| Tuần | Việc làm | Điều kiện hoàn thành |
|---|---|---|
| 1 | Ingestion/version metadata/OCR; benchmark 3 chunkers trên từng tài liệu | Không mất nguồn, không trùng ID, bảng và phủ định được giữ; audit OCR mẫu |
| 2 | Hybrid + parent expansion + multi-hop retrieval, ablation enrichment | So sánh cùng corpus/model/prompt; không giảm source recall so với baseline |
| 3 | Holdout evaluation, failure triage, latency/cost monitoring | ≥3 RAGAS metrics đạt 0,70; riêng faithfulness mục tiêu ≥0,85; mỗi lỗi có owner/fix và regression |
| 4 | Pilot nội bộ với human review và chính sách abstain | Có citation, trả lời lịch sử đúng phiên bản, phát hiện tài liệu cập nhật và reindex |

Các ngưỡng trên là mục tiêu dự án. Kết quả thực đo, comparison và latency của
lần chạy lab được ghi riêng trong `analysis/failure_analysis.md` và `reports/`.

## Kết quả thực đo của lần chạy lab

| Metric | Baseline | Production |
|---|---:|---:|
| faithfulness | 0.7969 | 0.9100 |
| answer_relevancy | 0.7452 | 0.7384 |
| context_precision | 0.9250 | 0.9500 |
| context_recall | 0.9083 | 0.9333 |

Metric thấp nhất là `answer_relevancy` (0.7384); đọc bottom-5 trong failure analysis để phân biệt nguyên nhân retrieval và generation.
Cross-encoder trung bình 477.85 ms/query trong run này; không đồng nhất với latency tải model hay p95.

Production cải thiện Faithfulness +0,1131, Precision/Recall +0,0250 nhưng Relevancy giảm 0,0068. Không nên coi thêm module là mọi metric sẽ tăng. Bài tính phí tạm ứng còn lỗi suy diễn tròn tháng; tôi ưu tiên guard về giả định tài chính và kiểm tra bằng human review trước khi dùng cho quyết định thật.

Kiểm tra nộp bài cuối: `check_lab.py` exit 0, 47/47 tests trong 23,27 giây và lint pass khi dùng model đã cache với HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE. Lượt online trước đó timeout 300 giây, còn lượt offline đi qua toàn bộ test; ghi nhận khả năng chờ mạng, không coi đó là bằng chứng test implementation hỏng.
