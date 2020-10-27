import multiprocessing
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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


def train_doc2vec_model(file_path: str, verbose: bool = False, model_name: str = "doc2vec.model") -> Doc2Vec:
    article_data = pd.read_csv(file_path)
    if verbose:
        print(f"Opened file for doc2vec model training, Length: {article_data.shape[0]}")
    texts = list(article_data["content"].to_numpy())
    model = Doc2Vec(size=300,
                    alpha=0.025,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1,
                    workers=multiprocessing.cpu_count())

    if verbose:
        print("Started data tagging")
        texts = tqdm(texts)
    tagged_data = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(texts)]
    if verbose:
        print("Started building the vocabulary")
    model.build_vocab(tagged_data)

    epoch_iter = range(20)
    if verbose:
        print("Doc2Vec model training started:")
        epoch_iter = tqdm(epoch_iter)
    for epoch in epoch_iter:
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # Decrease the learning rate
        model.alpha -= 0.0002
        # Fix the learning rate, no decay
        model.min_alpha = model.alpha

    model.save(model_name)
    return model


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
