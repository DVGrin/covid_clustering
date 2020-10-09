import numpy as np

from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def encode_texts(texts: List[str], method: str, *, verbose: bool = False) -> np.ndarray:
    try:
        methods = {
            'roberta': _encode_texts_roberta,
            'tf-idf': _encode_texts_tf_idf, 'tfidf': _encode_texts_tf_idf,
        }
        function = methods[method.lower()]
    except KeyError as key:
        raise KeyError(f"Invalid encoding method name: {key}! Available methods are: {tuple(methods.keys())}")
    return function(texts, verbose)


def _encode_texts_roberta(texts: List[str], verbose: bool = False) -> np.ndarray:
    model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    if verbose:
        print("Started text encoding:")
    sentence_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=verbose)
    return sentence_embeddings


def _encode_texts_tf_idf(texts: List[str], verbose: bool = False) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=4096)
    result = vectorizer.fit_transform(texts)
    if verbose:
        print("Text encoding finished")
    return result.toarray()
