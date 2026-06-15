"""
TextClass - Text Classification Training Script
Classifies short press texts into: business, entertainment, politics, sport, tech
"""

import os
import sys
import argparse
import joblib
import numpy as np
from pathlib import Path
from collections import Counter

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]
STOP_WORDS = set(stopwords.words("english"))


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_dataset(data_dir: str):
    """Load texts and labels from folder structure: data_dir/<category>/*.txt"""
    texts, labels = [], []
    data_path = Path(data_dir)
    for category in CATEGORIES:
        cat_path = data_path / category
        if not cat_path.exists():
            print(f"  [warn] Category folder not found: {cat_path}")
            continue
        files = sorted(cat_path.glob("*.txt"))
        for fpath in files:
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    texts.append(text)
                    labels.append(category)
            except Exception as e:
                print(f"  [warn] Could not read {fpath}: {e}")
    return texts, labels


# ──────────────────────────────────────────────
# Preliminary analysis
# ──────────────────────────────────────────────

def preliminary_analysis(texts, labels):
    print("\n=== PRELIMINARY ANALYSIS ===")
    label_counts = Counter(labels)
    total = len(labels)
    print(f"Total samples: {total}")
    print(f"{'Category':<16} {'Count':>6} {'%':>6}")
    print("-" * 30)
    for cat in CATEGORIES:
        n = label_counts.get(cat, 0)
        print(f"  {cat:<14} {n:>6} {100*n/total:>5.1f}%")

    lengths = [len(t.split()) for t in texts]
    print(f"\nText length (words): min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")


# ──────────────────────────────────────────────
# Pipeline builders
# ──────────────────────────────────────────────

def make_pipeline(clf_name: str, ngram_range=(1, 2)):
    """Return a sklearn Pipeline with TF-IDF + chosen classifier."""
    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        max_df=0.90,
        min_df=2,
        sublinear_tf=True,
    )
    classifiers = {
        "nb":      MultinomialNB(alpha=0.05),
        "svm":     CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000)),
        "lr":      LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs"),
        "rf":      RandomForestClassifier(n_estimators=300, random_state=42),
        "gb":      GradientBoostingClassifier(n_estimators=200, random_state=42),
    }
    if clf_name not in classifiers:
        raise ValueError(f"Unknown classifier: {clf_name}. Choose from {list(classifiers)}")
    return Pipeline([("tfidf", vectorizer), ("clf", classifiers[clf_name])])


# ──────────────────────────────────────────────
# Experiments
# ──────────────────────────────────────────────

def run_experiment(name, pipeline, texts, labels, cv=5):
    """Fit pipeline, compute CV accuracy, full-set accuracy, and report."""
    print(f"\n--- Experiment: {name} ---")
    cv_scores = cross_val_score(
        pipeline, texts, labels,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
    )
    pipeline.fit(texts, labels)
    train_acc = accuracy_score(labels, pipeline.predict(texts))
    print(f"  CV accuracy ({cv}-fold):  {np.mean(cv_scores):.4f}  ±{np.std(cv_scores):.4f}")
    print(f"  Training accuracy:       {train_acc:.4f}")
    return pipeline, np.mean(cv_scores)


def run_all_experiments(texts, labels):
    """Run all experiments and return the best pipeline."""
    print("\n=== CLASSIFICATION EXPERIMENTS ===")

    experiments = [
        # (label, clf_name, ngram_range)
        ("NaiveBayes  unigram  [baseline]", "nb",  (1, 1)),
        ("NaiveBayes  bigram",              "nb",  (1, 2)),
        ("SVM         unigram",             "svm", (1, 1)),
        ("SVM         bigram",              "svm", (1, 2)),
        ("LogReg      bigram",              "lr",  (1, 2)),
    ]

    best_pipe, best_cv, best_name = None, 0.0, ""
    for name, clf, ngram in experiments:
        pipe = make_pipeline(clf, ngram_range=ngram)
        pipe, cv_acc = run_experiment(name, pipe, texts, labels)
        if cv_acc > best_cv:
            best_cv, best_pipe, best_name = cv_acc, pipe, name

    print(f"\n  ✔ Best model: {best_name}  (CV={best_cv:.4f})")
    return best_pipe, best_name, best_cv


# ──────────────────────────────────────────────
# Detailed evaluation of best model
# ──────────────────────────────────────────────

def detailed_report(pipeline, texts, labels, cv=5):
    print("\n=== FINAL MODEL DETAILED EVALUATION ===")
    y_pred = pipeline.predict(texts)
    print("\nClassification Report (training set):")
    print(classification_report(labels, y_pred, target_names=CATEGORIES))
    print("Confusion Matrix (training set):")
    cm = confusion_matrix(labels, y_pred, labels=CATEGORIES)
    header = f"{'':>16}" + "".join(f"{c:>14}" for c in CATEGORIES)
    print(header)
    for i, row_label in enumerate(CATEGORIES):
        row = f"  {row_label:<14}" + "".join(f"{cm[i,j]:>14}" for j in range(len(CATEGORIES)))
        print(row)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TextClass classifier")
    parser.add_argument("data_dir", help="Path to dataset root (contains category folders)")
    parser.add_argument("--model-out", default="model.joblib",
                        help="Output path for trained model (default: model.joblib)")
    parser.add_argument("--cv", type=int, default=5, help="CV folds (default: 5)")
    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    texts, labels = load_dataset(args.data_dir)
    if not texts:
        print("ERROR: No texts loaded. Check data_dir path.")
        sys.exit(1)

    preliminary_analysis(texts, labels)
    best_pipeline, best_name, best_cv = run_all_experiments(texts, labels)
    detailed_report(best_pipeline, texts, labels, cv=args.cv)

    joblib.dump(best_pipeline, args.model_out)
    print(f"\n✔ Model saved to: {args.model_out}")


if __name__ == "__main__":
    main()
