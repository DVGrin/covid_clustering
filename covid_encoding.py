import numpy as np

from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def encode_texts(texts: List[str], method: str) -> np.ndarray:
    try:
        methods = {
            'roberta': _encode_texts_roberta,
            'tf-idf': _encode_texts_tf_idf
        }
        function = methods[method.lower()]
    except KeyError as key:
        raise KeyError(f"Invalid encoding method name: {key}! Available methods are: {tuple(methods.keys())}")
    return function(texts)


def _encode_texts_roberta(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    sentence_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return sentence_embeddings


def _encode_texts_tf_idf(texts: List[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=4096)
    result = vectorizer.fit_transform(texts)
    return result


def _preprocess_texts(texts: List[str]) -> List[str]:
    pass


def vectorize(text, maxx_features):
    vectorizer = TfidfVectorizer(max_features=maxx_features)
    X = vectorizer.fit_transform(text)
    return X
