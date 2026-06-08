# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
[Vẽ diagram kiến trúc ở đây]
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Phạm Thị Thắm | 2A202600789| FE kiêm api router be cơ bản | |
| Hồ Thành Tiến | 2A202600868| Task service (mốc vào BE) từ task 1 -> task4 | |
| Nguyễn Trần Mạnh Thắng | 2A202600710 | Task service (nối tiếp task4) từ task 5 đêns task 8 | |
| Trần Mạnh Chánh Quân | 2A202600786| task 9 10 | |
| Nguyễn Thái Bảo | 2A202600763| merger code | |

---

## Hướng Dẫn Chạy Demo

### Yêu cầu hệ thống
- Python 3.10+
- Kết nối Internet (lần đầu để tải embedding model ~471 MB)
- API keys: `OPENAI_API_KEY`, `JINA_API_KEY` (tuỳ chọn), `PAGEINDEX_API_KEY` (tuỳ chọn)

---

### Bước 1 — Cài đặt môi trường

```bash
# Tạo virtual environment (nếu chưa có)
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (macOS / Linux)
source .venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
pip install flask faiss-cpu sentence-transformers langchain-text-splitters rank-bm25
```

---

### Bước 2 — Cấu hình API keys

Tạo file `.env` trong thư mục `group_project/`:

```env
OPENAI_API_KEY=sk-...          # Bắt buộc cho generation
JINA_API_KEY=jina_...          # Tuỳ chọn — reranking chất lượng cao
PAGEINDEX_API_KEY=pi_...       # Tuỳ chọn — vectorless fallback
```

---

### Bước 3 — Thu thập dữ liệu (Task 1 & 2)

> Bỏ qua nếu đã có file trong `data/landing/`

```bash
cd group_project

# Task 1: Crawl văn bản pháp luật
python -m src.task1_collect_legal_docs

# Task 2: Crawl bài báo
python -m src.task2_crawl_news
```

---

### Bước 4 — Convert sang Markdown (Task 3)

> Bỏ qua nếu đã có file trong `data/standardized/`

```bash
python -m src.task3_convert_markdown
```

---

### Bước 5 — Build FAISS Index (Task 4) ⚠️ Bắt buộc

```bash
python -m src.task4_chunking_indexing
```

Output mong đợi:
```
[OK] Loaded N documents
[OK] Created XXXX chunks
[OK] Embedded XXXX chunks
[OK] FAISS index co XXXX vectors
[OK] Saved FAISS index: data/.faiss.index
```

> Lần đầu sẽ tải model `paraphrase-multilingual-MiniLM-L12-v2` (~471 MB).
> Các lần sau load từ cache, rất nhanh.

---

### Bước 6 — Khởi động Flask API Server

```bash
# Chạy từ thư mục group_project/
python app.py

# Hoặc chỉ định port khác
python app.py --port 5001
```

Output mong đợi:
```
============================================================
RAG Pipeline v2 — Flask API Server
  URL: http://127.0.0.1:5000
  OpenAI key:  ✓
  Jina key:    ✓
============================================================
 * Running on http://127.0.0.1:5000
```

---

### Bước 7 — Mở giao diện Demo

Mở trình duyệt và truy cập:

```
http://127.0.0.1:5000
```

Hoặc mở trực tiếp file `index.html` (server phải đang chạy để các tính năng hoạt động).

---

### Các API Endpoint

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/api/status` | Kiểm tra trạng thái FAISS, BM25, các API keys |
| `POST` | `/api/search/semantic` | Semantic Search (Task 5) |
| `POST` | `/api/search/lexical` | BM25 Lexical Search (Task 6) |
| `POST` | `/api/rerank` | Reranking — Jina hoặc score sort (Task 7) |
| `POST` | `/api/retrieve` | Full hybrid retrieval + fallback (Task 9) |
| `POST` | `/api/generate` | RAG generation có citation (Task 10) |

**Ví dụ gọi API:**
```bash
# Kiểm tra status
curl http://127.0.0.1:5000/api/status

# Semantic search
curl -X POST http://127.0.0.1:5000/api/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "hình phạt tội tàng trữ ma tuý", "top_k": 5}'

# Generation với citation
curl -X POST http://127.0.0.1:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "Luật phòng chống ma tuý 2021 quy định gì?", "top_k": 5}'
```

---

### Chạy Evaluation (Bài Nhóm)

```bash
cd group_project
python evaluation/eval_pipeline.py
```

Kết quả lưu tại `evaluation/results.md`.

---

### Troubleshooting

| Lỗi | Nguyên nhân | Cách sửa |
|-----|-------------|----------|
| `No index available. Run task4 first.` | Chưa build FAISS index | Chạy `python -m src.task4_chunking_indexing` |
| `ModuleNotFoundError: No module named 'faiss'` | Thiếu dependency | `pip install faiss-cpu` |
| `Server offline` trên status bar | Flask chưa chạy | Chạy `python app.py` |
| Chat trả về raw context thay vì câu trả lời | Chưa có `OPENAI_API_KEY` | Thêm key vào file `.env` |
| Rerank dùng "score sort" thay vì Jina | Chưa có `JINA_API_KEY` | Thêm key vào file `.env` |

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
