from typing import List

from yake import KeywordExtractor
from rake_nltk import Rake, Metric
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def extract_common_phrases(articles: List[str], method: str, n_keywords: int) -> List[str]:
    try:
        methods = {
            'rake': _extract_common_phrases_rake,
            'yake': _extract_common_phrases_yake,
            'lda': _extract_common_phrases_lda
        }
        function = methods[method.lower()]
    except KeyError as key:
        raise KeyError(f"Invalid keyword extraction method name: {key}! Available methods are: {tuple(methods.keys())}")
    return function(articles, n_keywords)


def _extract_common_phrases_rake(articles: List[str], n_keywords: int) -> List[str]:
    text = '\n'.join(articles)
    r = Rake(min_length=2, max_length=5, ranking_metric=Metric.WORD_FREQUENCY)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()[:n_keywords]


def _extract_common_phrases_yake(articles: List[str], n_keywords: int) -> List[str]:
    text = '\n'.join(articles)
    custom_kw_extractor = KeywordExtractor(lan='en',
                                           n=3,
                                           dedupLim=0.9,
                                           dedupFunc='seqm',
                                           windowsSize=1,
                                           top=n_keywords,
                                           features=None)
    keywords = custom_kw_extractor.extract_keywords(text)

    return [keyword[1] for keyword in keywords]


def _extract_common_phrases_lda(articles: List[str], n_keywords: int) -> List[str]:
    vectorizer = CountVectorizer(min_df=0.2, max_df=0.95,
                                 stop_words='english',
                                 lowercase=True,
                                 token_pattern='[-a-zA-Z][-a-zA-Z]{2,}')
    # TODO: Try using doc2vec, or other of our vectorizers here
    #       Or better yet, just use already vectorizer data?
    vectorized_data = vectorizer.fit_transform(articles)
    lda = LatentDirichletAllocation(n_components=20, max_iter=10,
                                    learning_method='online',
                                    verbose=False, random_state=42)
    lda.fit(vectorized_data)

    n_topics_to_use = 10
    current_words = set()
    keywords = []

    for topic in lda.components_:
        words = [(vectorizer.get_feature_names()[i], topic[i])
                 for i in topic.argsort()[:-n_topics_to_use - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.add(word[0])

    keywords.sort(key=lambda x: x[1], reverse=True)
    return [keyword[0] for keyword in keywords][:20]
