import numpy as np

from tqdm import tqdm
from typing import List
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


def encode_texts(texts: List[str], method: str, *, verbose: bool = False) -> np.ndarray:
    try:
        methods = {
            'roberta': _encode_texts_roberta,
            'tf-idf': _encode_texts_tf_idf, 'tfidf': _encode_texts_tf_idf,
            'doc2vec': _encode_texts_doc2vec
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


def _encode_texts_doc2vec(texts: List[str], verbose: bool = False) -> np.ndarray:
    model = Doc2Vec.load("doc2vec.model")
    if verbose:
        print("Doc2Vec encoding started:")
        texts = tqdm(texts)
    vectorized_texts = []
    for text in texts:
        tokenized_text = word_tokenize(text)
        vectorized_texts.append(model.infer_vector(tokenized_text))
    return np.vstack(vectorized_texts)
