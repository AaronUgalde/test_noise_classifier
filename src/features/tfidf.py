"""TF-IDF feature extraction."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(config: dict) -> TfidfVectorizer:
    tfidf_cfg = config["tfidf"]
    return TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=tfidf_cfg["min_df"],
        sublinear_tf=tfidf_cfg["sublinear_tf"],
    )


def fit_transform_tfidf(vectorizer: TfidfVectorizer, train_texts, test_texts):
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    return x_train, x_test
