import joblib

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB

NUM_SPLITS = 5
RANDOM_STATE = 42

class TrainClassifier:
    def __init__(self):
        self.NUM_SPLITS = 5
        self.RANDOM_STATE = 42
        self.kfold = StratifiedKFold(n_splits=self.NUM_SPLITS, shuffle=True,
                                    random_state=self.RANDOM_STATE)

    def print_metrics(self, y_true, y_pred):
        print(f"ACCURACY: {accuracy_score(y_true, y_pred)}")
        print(f"MCC: {matthews_corrcoef(y_true, y_pred)}")

    def train(self, df):
        for fold, (train_index, valid_index) in enumerate(self.kfold.split(df["Medical_Description"], df["Package"])):
            print("*"*40)
            print("*"+" "*16+f"FOLD {fold+1}"+" "*16+"*")
            print("*"*40, end="\n")

            X_train = df.iloc[train_index, :].reset_index(drop=True)
            X_valid = df.iloc[valid_index, :].reset_index(drop=True)

            y_train = X_train["Package"]
            y_valid = X_valid["Package"]

            vec = TfidfVectorizer()
            train_term_doc = vec.fit_transform(X_train["Medical_Description"])
            valid_term_doc = vec.transform(X_valid["Medical_Description"])

            naive_bayes = MultinomialNB()
            naive_bayes.fit(train_term_doc, y_train)

            valid_preds = naive_bayes.predict(valid_term_doc)
            self.print_metrics(y_valid, valid_preds)

            joblib.dump(vec, f"../pickles/tfidf_{fold}.joblib")
            joblib.dump(naive_bayes, f"../models/classifier_{fold}.joblib")


if __name__ == "__main__":
    classifier = TrainClassifier()
    df = pd.read_csv("../data/processed/processed_data.csv")
    classifier.train(df)
