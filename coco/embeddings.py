import numpy as np

_model = None
_model_name = None


def get_model(name: str):
    global _model, _model_name
    if _model is None or _model_name != name:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(name)
        _model_name = name
    return _model


def embed(text: str, model_name: str) -> np.ndarray:
    model = get_model(model_name)
    v = model.encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Assumes both vectors are L2-normalized
    return float(np.dot(a, b))
