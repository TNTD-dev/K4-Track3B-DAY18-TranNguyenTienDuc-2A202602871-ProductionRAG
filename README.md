# Lab 18: Production RAG Pipeline

**K4-Track3B · Ngày 18 · Production RAG**  
**Thời gian:** 2h implement + 30 phút reflection

---

## Tổng quan

Bài tập **cá nhân** — implement toàn bộ 5 modules:

```
M1 Chunking → M5 Enrichment → M2 Hybrid Search → M3 Reranking → LLM Answer → M4 RAGAS Eval
```

Xem **ASSIGNMENT.md** để biết chi tiết từng module và timeline.

## Prerequisites

| Dependency | Bắt buộc? | Dùng cho |
|-----------|-----------|----------|
| Docker (Qdrant) | ✅ Có | M2 Dense Search |
| Python 3.11+ | ✅ Có | Tất cả modules (RAGAS cần 3.11+ cho asyncio) |
| `OPENAI_API_KEY` | ⚠️ M4+M5 | RAGAS eval (M4), Enrichment LLM (M5) |

**Pre-download models** (tránh timeout trong lab):
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
```

## Quick Start

### 1. Clone repository & tạo môi trường ảo

**Linux / macOS / Git Bash:**
```bash
git clone <repo-url>
cd K4-Track3B-Production-RAG
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
git clone <repo-url>
cd K4-Track3B-Production-RAG
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Nếu dùng Windows CMD: chạy `.venv\Scripts\activate.bat`)*

### 2. Cài đặt dependencies & Khởi động dịch vụ

**Linux / macOS / Git Bash:**
```bash
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
cp .env.example .env                    # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```

**Windows (PowerShell):**
```powershell
docker compose up -d                    # Khởi động Qdrant vector database
pip install -r requirements.txt
Copy-Item .env.example .env             # Tạo file .env và điền OPENAI_API_KEY
python naive_baseline.py                # Khởi tạo baseline
```
*(Nếu dùng Windows CMD: dùng `copy .env.example .env` thay cho `Copy-Item`)*

## Chạy toàn bộ & Kiểm tra

```bash
python main.py                          # Chạy Naive + Production + In bảng so sánh
python check_lab.py                     # Script kiểm tra hợp lệ trước khi nộp (chạy được trên mọi OS)
```

## Cấu trúc repo

```
K4-Track3B-Production-RAG/
├── README.md                   # File này
├── ASSIGNMENT.md               # ★ Đề bài + timeline + reflection
├── RUBRIC.md                   # Hệ thống chấm điểm
│
├── main.py                     # Entry point: chạy toàn bộ pipeline
├── check_lab.py                # Kiểm tra định dạng trước khi nộp
├── naive_baseline.py           # Baseline (chạy trước)
├── config.py                   # Shared config
├── requirements.txt            # Dependencies
├── docker-compose.yml          # Qdrant local
├── .env.example                # API keys template
│
├── data/                       # Corpus tiếng Việt — 25 .md files + 3 PDFs (28 files total)
│   ├── nghi_phep_nam_v2023.md  # Nghỉ phép 12 ngày (v2023, superseded)
│   ├── nghi_phep_nam_v2024.md  # Nghỉ phép 15 ngày (v2024, hiện hành)
│   ├── mat_khau_v1.md          # Password policy 90 ngày (OLD)
│   ├── mat_khau_v2.md          # Password policy 120 ngày + MFA (NEW)
│   ├── ... (28 files total)    # 8 categories: leave, salary, IT, workflow, training, admin, safety, compliance
│   ├── so_tay_an_toan.pdf      # An toàn PCCC + sơ cứu (PDF text)
│   ├── BCTC.pdf                # Báo cáo tài chính (scan, cần OCR)
│   └── Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf # Nghị định BVDL (scan, cần OCR)
├── test_set.json               # 20 Q&A pairs (6 types: lookup, version, negation, multi-hop, numeric, ambiguous)
│
├── src/                        # ★ Scaffold code (có TODO markers)
│   ├── m1_chunking.py          # Module 1: Chunking
│   ├── m2_search.py            # Module 2: Hybrid Search
│   ├── m3_rerank.py            # Module 3: Reranking
│   ├── m4_eval.py              # Module 4: Evaluation
│   ├── m5_enrichment.py        # Module 5: Enrichment Pipeline
│   └── pipeline.py             # Ghép toàn bộ pipeline
│
├── tests/                      # Auto-grading
│   ├── test_m1.py
│   ├── test_m2.py
│   ├── test_m3.py
│   ├── test_m4.py
│   └── test_m5.py
│
├── analysis/                   # ★ Deliverable
│   ├── failure_analysis.md     # Phân tích failures (cá nhân)
│   └── reflections/            # Reflection cá nhân
│       └── reflection_TEMPLATE.md
│
├── reports/                    # ★ Auto-generated (bắt buộc: reports/ragas_report.json)
│   ├── ragas_report.json
│   └── naive_baseline_report.json
│
└── templates/                  # Templates gốc (backup)
    └── failure_analysis.md
```

## Timeline (Thời lượng ước tính)

| Thời lượng | Hoạt động |
|------------|-----------|
| 10 phút | Setup môi trường + chạy `naive_baseline.py` |
| 90 phút | Implement M1 → M2 → M3 → M4 → M5 |
| 20 phút | Chạy pipeline + RAGAS + failure analysis |
| 30 phút | Reflection: lecture mapping + project plan |

## Quy chuẩn đặt tên Repository & Nộp bài

- **Cấu trúc đặt tên repo:**  
  `K4-Track3B-DAY18-<HoVaTen>-<MSSV>-ProductionRAG`  
  *(Ví dụ: `K4-Track3B-DAY18-NguyenVanAn-AI20K001-ProductionRAG`)*
- **Hạn chót nộp bài:** **11h59 ngày hôm sau diễn ra bài lab (GMT+7)** trên cổng VLearn LMS / Codelab.
- **Chi tiết yêu cầu:** Xem tại [ASSIGNMENT.md](ASSIGNMENT.md) và [RUBRIC.md](RUBRIC.md).

## Bản triển khai của Trần Nguyễn Tiến Đức

Pipeline retrieve **child** bằng BM25 + BGE-M3 + RRF, rerank nội dung nguồn bằng
BGE reranker, rồi mở rộng thành tối đa 3 **parent khác nhau** để trả lời. ID parent
phụ thuộc nguồn và nội dung nên không trùng giữa tài liệu. Context của LLM dùng
văn bản gốc và tên nguồn; summary/HyQA được dùng để tìm kiếm.

Enrichment mặc định dùng combined JSON (1 request/chunk), tối đa 4 request đồng
thời. Cache theo nội dung, nguồn và phiên bản prompt nằm trong `.cache/enrichment/`
(được gitignore). Khi API không hoạt động, trạng thái fallback được ghi nhận.
Không có API key vẫn chạy retrieval, nhưng không có điểm RAGAS hợp lệ.

Answer Relevancy sinh câu hỏi tiếng Việt với ví dụ chung được dịch từ RAGAS và
ví dụ phủ định về giờ mở cửa thư viện (phân biệt trả lời “không” với “không biết”);
prompt được lưu tại `reports/evaluator_prompt_vi.json`, công thức metric giữ nguyên.
Baseline và production dùng cùng model trả lời, prompt, temperature=0, embedding
và bộ 20 câu hỏi. Ground truth chỉ được truyền vào evaluator, không vào retrieval
hoặc generation. Bộ test cố định nhỏ nên kết quả chưa chứng minh tổng quát hóa.

Dùng Python 3.11 của virtualenv; Python mặc định trên máy có thể là 3.9:

```bash
source .venv/bin/activate
# Nếu tải Hugging Face bị treo ở Xet:
export HF_HUB_DISABLE_XET=1
docker compose up -d
python main.py
python check_lab.py
```

- `reports/ragas_report.json`: aggregate, từng câu hỏi, bottom-5, trạng thái
  evaluator, model, backend và latency.
- `reports/naive_baseline_report.json`: cùng schema để so sánh baseline.
- `reports/latency_report.json`: thời gian build, retrieval, reranking, generation,
  evaluation (đơn vị ms). Model tải lần đầu và cache ảnh hưởng thời gian build.
- `analysis/failure_analysis.md`: kết quả thực nghiệm và Diagnostic Tree.
- `analysis/reflections/reflection_TranNguyenTienDuc.md`: mapping 5 modules,
  debug và kế hoạch cho trợ lý chính sách nội bộ của lab.

`check_lab.py` trả exit code khác 0 khi thiếu deliverable, test/lint thất bại hoặc
đánh giá RAGAS không thành công. Điểm 0 từ fallback không được coi là điểm thật.
Qdrant in-memory fallback được ghi trong report khi server không hoạt động.
Hai PDF scan cần OCR trước khi đưa vào corpus; không tạo văn bản giả để lấp chỗ trống.

API được đối chiếu với [Qdrant collections](https://qdrant.tech/documentation/manage-data/collections/),
[Qdrant search](https://qdrant.tech/documentation/search/) và
[RAGAS RunConfig](https://docs.ragas.io/en/v0.2.0/howtos/customizations/_run_config/).
BGE-M3 được load với `use_safetensors=False` để không tự download bản conversion
trong background của transformers; BGE reranker dùng safetensors chính thức.
Code dùng RAGAS 0.1.x theo requirements của đề; không sử dụng schema 0.2.x.

Nếu tải weight lớn bị gián đoạn, có thể chạy `python scripts/download_models.py`.
Script tải range từ CDN chính thức Hugging Face, tiếp tục các phần đã lưu, xác minh
SHA-256 trước khi đưa vào cache, rồi xóa các phần tạm. Cần khoảng 9 GB dung lượng
trống trong lúc ghép hai weight. Không cần token Hugging Face cho các model public.

Sau khi đã tải đầy đủ model, có thể kiểm tra hoàn toàn từ cache để tránh chờ mạng:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python check_lab.py
```

Xem thêm `analysis/rubric_checklist.md` để đối chiếu từng tiêu chí với bằng chứng,
và `analysis/failure_analysis.md` cho bảng kết quả cùng các giới hạn còn lại.

Bản chốt bổ sung `src/retrieval.py`: truy hồi từng ý của câu hỏi nhiều phần và
chọn phiên bản policy có hiệu lực theo ngày/năm hỏi trước rerank. `src/calculation.py`
tính số thập phân có giới hạn; trường hợp phí rõ quy định dùng nhánh tính từ
source và dữ kiện người hỏi, luôn ghi giả định pro-rata. Bản cuối đạt cả 4 metrics
≥0,75 và Faithfulness ≥0,85; các thí nghiệm trước được giữ riêng trong reports/.
