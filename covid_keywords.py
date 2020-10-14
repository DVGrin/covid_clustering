from typing import List
from yake import KeywordExtractor
from rake_nltk import Rake, Metric


def extract_common_phrases(text: str, method: str, *, verbose: bool = False) -> List[str]:
    try:
        methods = {
            'rake': _extract_common_phrases_rake,
            'yake': _extract_common_phrases_yake
        }
        function = methods[method.lower()]
    except KeyError as key:
        raise KeyError(f"Invalid keyword extraction method name: {key}! Available methods are: {tuple(methods.keys())}")
    return function(text, verbose)


def _extract_common_phrases_rake(text: str, verbose: bool = False) -> List[str]:
    r = Rake(min_length=2, max_length=5, ranking_metric=Metric.WORD_FREQUENCY)
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()


def _extract_common_phrases_yake(text: str, verbose: bool = False) -> List[str]:
    custom_kw_extractor = KeywordExtractor(lan='en',
                                           n=3,
                                           dedupLim=0.9,
                                           dedupFunc='seqm',
                                           windowsSize=1,
                                           top=20,
                                           features=None)
    keywords = custom_kw_extractor.extract_keywords(text)

    return [keyword[1] for keyword in keywords]
