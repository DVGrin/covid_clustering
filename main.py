import os
import numpy as np
import pandas as pd

from typing import List, Dict, Union

from tqdm import tqdm
from summarizer import Summarizer
from rake_nltk import Rake, Metric
from sklearn.decomposition import PCA
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig

from covid_encoding import encode_texts
from covid_clustering import get_cluster_labels


# Parameters
VERBOSE = True
USE_SUMMARY = False
N_SAMPLES = 1000  # int or None


def main():
    article_data = open_csv_file("../datasets/covid_articles/covid19_articles.csv", n_samples=N_SAMPLES, verbose=VERBOSE)
    texts = list(article_data["content"].to_numpy())
    if USE_SUMMARY:
        texts = get_summary(texts, verbose=VERBOSE)

    embeddings = encode_texts(texts, 'roberta', verbose=VERBOSE)
    embeddings = dimensionality_reduction(embeddings)

    cluster_labels = get_cluster_labels(embeddings, method='hdbscan')
    if VERBOSE:
        print("Clustering finished")

    article_data["cluster_label"] = cluster_labels
    article_data_clustered = article_data.groupby("cluster_label").apply(dataframe_regroup)
    generate_html_report(article_data_clustered, "report.html")


def open_csv_file(path: str, *, n_samples: Union[None, int] = None, verbose: bool = False) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    if n_samples:
        dataframe = dataframe.sample(n=n_samples)
    if verbose:
        if n_samples:
            print(f"Chosen {n_samples} random samples from articles dataset")
        else:
            print(f"Articles dataset opened. Length: {dataframe.shape[0]}")
    return dataframe


def get_summary(texts: List[str], *, verbose: bool = False) -> List[str]:
    config = AutoConfig.from_pretrained('allenai/biomed_roberta_base')
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained('allenai/biomed_roberta_base')
    model = AutoModel.from_pretrained('allenai/biomed_roberta_base', config=config)
    summarizer = Summarizer(custom_model=model, custom_tokenizer=tokenizer)
    if verbose:
        print("Summary extraction started:")
        texts = tqdm(texts)
    result = [summarizer(text) for text in texts]
    return result


def dimensionality_reduction(embeddings: np.ndarray) -> np.ndarray:
    max_pca_components = 768
    pca = PCA(n_components=(min(N_SAMPLES or max_pca_components, max_pca_components)))
    result = pca.fit_transform(embeddings)
    print(f"Dimensions: {len(embeddings)}x{len(embeddings[0])} -> {result.shape[0]}x{result.shape[1]}")
    return pca.fit_transform(embeddings)


def dataframe_regroup(dataframe: pd.DataFrame) -> Dict:
    # text = '.\n'.join(dataframe["content"])
    titles = dataframe['title'].values
    size = dataframe.shape[0]
    # common_phrases = extract_common_phrases(text)[:10]

    return pd.Series({'titles': titles,
                      'cluster_size': size})
    #   'common_phrases': common_phrases})


def extract_common_phrases(text: str) -> List[str]:
    r = Rake(min_length=2, max_length=5, ranking_metric=Metric.WORD_FREQUENCY)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()


def generate_html_report(dataframe: pd.DataFrame, output_file: str) -> None:
    env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__))))
    template = env.get_template("template.html")
    original_list = dataframe.to_dict('records')
    new_list = [{key: values for key, values in record.items()
                 if key in ['cluster_size', 'titles']}
                for record in original_list]
    template_vars = {"values": new_list}
    html_out = template.render(template_vars)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_out)


if __name__ == "__main__":
    main()
