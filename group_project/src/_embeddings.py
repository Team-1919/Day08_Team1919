"""
Embedding qua transformers (cùng model MiniLM), tránh import sentence_transformers
→ sklearn/pandas/pyarrow gây access violation trên Windows khi chạy pytest.
"""

import os
import threading

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

_lock = threading.Lock()
_tokenizer = None
_model = None


def _mean_pooling(model_output, attention_mask):
    import torch

    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def _get_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    with _lock:
        if _tokenizer is not None and _model is not None:
            return _tokenizer, _model

        from transformers import AutoModel, AutoTokenizer
        import torch

        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        _model = AutoModel.from_pretrained(EMBEDDING_MODEL)
        _model.eval()
        torch.set_num_threads(1)

    return _tokenizer, _model


def encode_texts(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed danh sách text, trả về array (n, 384) đã L2-normalize."""
    import torch

    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    tokenizer, model = _get_model()
    batches = range(0, len(texts), batch_size)
    if show_progress and len(texts) > batch_size:
        try:
            from tqdm import tqdm

            batches = tqdm(batches, desc="Embedding")
        except ImportError:
            pass

    all_embeddings = []
    with torch.no_grad():
        for start in batches:
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            outputs = model(**encoded)
            embeddings = _mean_pooling(outputs, encoded["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


def embed_text(text: str) -> list[float]:
    """Embed một đoạn text."""
    if not text or not str(text).strip():
        return [0.0] * EMBEDDING_DIM
    return encode_texts([text])[0].tolist()
