import os
import warnings
import numpy as np
import pandas as pd

from typing import List, Dict, Union

from tqdm import tqdm
from summarizer import Summarizer
from sklearn.decomposition import PCA
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig

from covid_encoding import encode_texts, train_doc2vec_model
from covid_preprocessing import clean_text
from covid_clustering import get_cluster_labels
from covid_keywords import extract_common_phrases


INPUT_FILE = "../datasets/covid_articles/covid19_articles.csv"


class ArticleClustering():
    def __init__(self, clustering_method: str = "hdbscan", encoding_method: str = "roberta",
                 keyword_extraction_method: str = "yake", num_keywords: int = 20,
                 verbose: bool = True, train_new_model: bool = False, use_summary: bool = False,
                 report_filename: str = "report.html"):
        self.clustering_method = clustering_method
        self.encoding_method = encoding_method
        self.keyword_extraction_method = keyword_extraction_method
        self.num_keywords = num_keywords
        self.verbose = verbose
        self.train_new_model = train_new_model
        self.use_summary = use_summary
        self.report_filename = report_filename

    def analyze(self, input_file: str, n_samples: Union[int, None] = None) -> None:
        self.n_samples = n_samples
        if self.encoding_method == "doc2vec" and self.train_new_model:
            train_doc2vec_model(input_file, verbose=self.verbose)
        article_data = self.open_csv_file(input_file, n_samples=n_samples, verbose=self.verbose)
        texts = list(article_data["content"].to_numpy())
        if self.use_summary:
            texts = self.get_summary(texts)

        texts = clean_text(texts, verbose=self.verbose)
        article_data["filtered_content"] = texts
        # print(texts[0])
        embeddings = encode_texts(texts, self.encoding_method, verbose=self.verbose)
        embeddings = self.dimensionality_reduction(embeddings, 300, verbose=self.verbose)

        cluster_labels = get_cluster_labels(embeddings, method=self.clustering_method)
        article_data["cluster_label"] = cluster_labels

        if self.verbose:
            print("Clustering finished\nKeyword extraction started")
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=FutureWarning)
                tqdm.pandas()
            article_data_clustered = article_data.groupby("cluster_label").progress_apply(self.dataframe_regroup)
        else:
            article_data_clustered = article_data.groupby("cluster_label").apply(self.dataframe_regroup)
        article_data_clustered = article_data_clustered.sort_values(by=["cluster_size"], ascending=False)
        self.generate_html_report(article_data_clustered)

    @staticmethod
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

    def get_summary(self, texts: List[str]) -> List[str]:
        config = AutoConfig.from_pretrained('allenai/biomed_roberta_base')
        config.output_hidden_states = True
        tokenizer = AutoTokenizer.from_pretrained('allenai/biomed_roberta_base')
        model = AutoModel.from_pretrained('allenai/biomed_roberta_base', config=config)
        summarizer = Summarizer(custom_model=model, custom_tokenizer=tokenizer)
        if self.verbose:
            print("Summary extraction started:")
            texts = tqdm(texts)
        result = [summarizer(text) for text in texts]
        return result

    def dimensionality_reduction(self, embeddings: np.ndarray, max_pca_components: int = 300, verbose: bool = False) -> np.ndarray:
        pca = PCA(n_components=(min(self.n_samples or max_pca_components, max_pca_components)))
        result = pca.fit_transform(embeddings)
        if verbose and len(embeddings[0]) != result.shape[1]:
            print(f"Dimensionality reduction: {len(embeddings)}x{len(embeddings[0])} -> {result.shape[0]}x{result.shape[1]}")
        return result

    def dataframe_regroup(self, dataframe: pd.DataFrame) -> Dict:
        titles = dataframe["title"].to_numpy()
        size = dataframe.shape[0]
        texts = dataframe["filtered_content"].to_list()
        common_phrases = extract_common_phrases(texts, self.keyword_extraction_method, n_keywords=self.num_keywords)

        return pd.Series({'titles': titles,
                          'cluster_size': size,
                          'common_phrases': common_phrases})

    def generate_html_report(self, dataframe: pd.DataFrame) -> None:
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__))))
        template = env.get_template("template.html")
        original_list = dataframe.to_dict('records')
        new_list = [{key: values for key, values in record.items()
                    if key in ['cluster_size', 'titles', 'common_phrases']}
                    for record in original_list]
        template_vars = {"values": new_list}
        html_out = template.render(template_vars)
        with open(self.report_filename, "w", encoding="utf-8") as f:
            f.write(html_out)


def main():
    pipeline = ArticleClustering(encoding_method="doc2vec")
    pipeline.analyze(INPUT_FILE, 1000)


if __name__ == "__main__":
    main()
