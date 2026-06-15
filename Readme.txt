TextClass - Short Text Classifier
===================================
Classifies press texts into: business, entertainment, politics, sport, tech

REQUIRED SOFTWARE
-----------------
Python 3.8+

Install dependencies:
    pip install -r requirements.txt

On first run, NLTK stopwords are downloaded automatically (~1 MB).

CLASSIFY (pre-trained model)
-----------------------------
Classify a single text file:
    python classify.py model.joblib path/to/article.txt

Classify a directory of .txt files:
    python classify.py model.joblib path/to/test_dir/

Classify a labelled directory (category subfolders) and compute metrics:
    python classify.py model.joblib mlarr_text/

RETRAIN FROM DATA
-----------------
    python train.py mlarr_text/ --model-out model.joblib

Options:
    --model-out   Output path for the trained model   (default: model.joblib)
    --cv          Number of cross-validation folds    (default: 5)

NOTE: The training data folder (mlarr_text/) is NOT included in this package.
Extract TextClass_text.tgz to get it.

FILES
-----
  train.py          Training script (all experiments + best model saved)
  classify.py       Inference script (single file or directory)
  model.joblib      Pre-trained SVM+TF-IDF model (best CV: linear SVM, unigrams)
  requirements.txt  Python package list
  Readme.txt        This file
