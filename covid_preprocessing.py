import string
from tqdm import tqdm
from typing import List
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Custom stopword list added from:
# https://www.kaggle.com/maksimeren/covid-19-literature-clustering#Data-Pre-processing
STOPWORDS = stopwords.words('english') + [
    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
    'al.', 'Elsevier', 'PMC', 'CZI', 'www'
]


def clean_text(texts: List[str], *, verbose: bool = False) -> List[str]:
    if verbose:
        print("Text cleaning started:")
        texts = tqdm(texts)
    for text in texts:
        # lower text
        text = text.lower()
        # tokenize text and remove puncutation
        words = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        words = [word for word in words if not any(c.isdigit() for c in word)]
        # remove stop words
        words = [word for word in words if word not in STOPWORDS]
        # remove empty tokens
        words = [word for word in words if len(word) > 0]
        # lemmatize text
        pos_tags = pos_tag(words)
        words = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        words = [word for word in words if len(word) > 1]
        # join all
        text = " ".join(text)
    return texts


def get_wordnet_pos(pos_tag: str) -> str:
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
