
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

class Tier1Baseline:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ('clf', OneVsRestClassifier(LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)))
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
