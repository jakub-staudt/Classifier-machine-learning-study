"""
TextClass - Classifier / Inference Script
Usage:
  python classify.py <model.joblib> <text_or_dir> [--labels-dir <dir>]
"""

import os
import sys
import argparse
import joblib
from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]


def load_texts_from_dir(path: Path):
    """Load texts from a flat dir or a category-structured dir."""
    texts, labels, filenames = [], [], []
    # Check if it looks like a labelled structure
    subdirs = [d for d in path.iterdir() if d.is_dir() and d.name in CATEGORIES]
    if subdirs:
        for cat in CATEGORIES:
            cat_path = path / cat
            if not cat_path.exists():
                continue
            for fpath in sorted(cat_path.glob("*.txt")):
                txt = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if txt:
                    texts.append(txt)
                    labels.append(cat)
                    filenames.append(str(fpath))
    else:
        for fpath in sorted(path.glob("*.txt")):
            txt = fpath.read_text(encoding="utf-8", errors="replace").strip()
            if txt:
                texts.append(txt)
                labels.append(None)
                filenames.append(str(fpath))
    return texts, labels, filenames


def classify_and_report(pipeline, texts, true_labels, filenames, verbose=True):
    predictions = pipeline.predict(texts)
    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(texts)
        classes = pipeline.classes_

    if verbose:
        print(f"\n{'File':<50} {'Predicted':<16} {'True':<16} {'Match'}")
        print("-" * 100)
        for fname, pred, true in zip(filenames, predictions, true_labels or [None]*len(predictions)):
            match = ("✔" if pred == true else "✗") if true else ""
            print(f"  {Path(fname).name:<48} {pred:<16} {(true or '?'):<16} {match}")

    any_true = true_labels and any(l is not None for l in true_labels)
    if any_true:
        valid = [(t, p) for t, p in zip(true_labels, predictions) if t is not None]
        yt, yp = zip(*valid)
        acc = accuracy_score(yt, yp)
        print(f"\n=== EVALUATION METRICS ===")
        print(f"Accuracy: {acc:.4f}  ({int(acc*len(yt))}/{len(yt)} correct)")
        print("\nClassification Report:")
        print(classification_report(yt, yp, target_names=CATEGORIES, zero_division=0))
        print("Confusion Matrix:")
        cm = confusion_matrix(yt, yp, labels=CATEGORIES)
        header = f"{'':>16}" + "".join(f"{c:>14}" for c in CATEGORIES)
        print(header)
        for i, row_label in enumerate(CATEGORIES):
            row = f"  {row_label:<14}" + "".join(f"{cm[i,j]:>14}" for j in range(len(CATEGORIES)))
            print(row)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Classify texts using a pre-trained model")
    parser.add_argument("model", help="Path to model file (model.joblib)")
    parser.add_argument("input", help="Single .txt file, directory of .txt files, or category-structured dir")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file output, show summary only")
    args = parser.parse_args()

    # Load model
    if not Path(args.model).exists():
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    pipeline = joblib.load(args.model)
    print(f"Model loaded: {args.model}")

    # Load input
    input_path = Path(args.input)
    if input_path.is_file():
        text = input_path.read_text(encoding="utf-8", errors="replace").strip()
        texts = [text]
        labels = [None]
        filenames = [str(input_path)]
    elif input_path.is_dir():
        texts, labels, filenames = load_texts_from_dir(input_path)
        if not texts:
            print("ERROR: No .txt files found in the given directory.")
            sys.exit(1)
        print(f"Loaded {len(texts)} texts from {input_path}")
    else:
        print(f"ERROR: Input not found: {args.input}")
        sys.exit(1)

    classify_and_report(pipeline, texts, labels, filenames, verbose=not args.quiet)


if __name__ == "__main__":
    main()
