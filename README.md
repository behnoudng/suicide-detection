# Suicide Risk Detection

A hybrid ensemble model combining XGBoost and fine-tuned BERT for detecting suicidal ideation in text. Optimized for recall, because missing someone in crisis is not an acceptable error.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Performance

| Model        | Recall    | Precision | F1-Score |
| ------------ | --------- | --------- | -------- |
| XGBoost      | 0.894     | —         | —        |
| BERT         | 0.985     | —         | —        |
| **Ensemble** | **0.988** | 0.963     | 0.975    |
|              |           |           |          |

Final ensemble achieves **98% accuracy** with a decision threshold tuned for high recall (0.4).

---
## Why Recall?
In suicide risk detection, a false negative means a person in crisis goes undetected. A false positive means extra review. The cost asymmetry is obvious. This model is tuned accordingly.

---

## Architecture


                    ┌─────────────────┐
                    │   Input Text    │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │   Preprocessing │          │   Raw Text      │
     │   + TF-IDF      │          │   (for BERT)    │
     │   + Features    │          │                 │
     └────────┬────────┘          └────────┬────────┘
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │    XGBoost      │          │   Fine-tuned    │
     │                 │          │   BERT          │
     └────────┬────────┘          └────────┬────────┘
              │                             │
              │      P(suicide)             │ P(suicide)
              │         0.3                 │    0.7
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Weighted Average│
                    │   Threshold=0.4 │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Prediction    │
                    └─────────────────┘

**Ensemble weights:** 30% XGBoost, 70% BERT

---

## Features

### Text Preprocessing
- URL, email, and Reddit-specific notation removal
- Lemmatization with NLTK
- Stopword removal *with negation preservation* (`not`, `never`, `nothing`, etc. are kept. They matter here)

### Engineered Features (XGBoost)
| Feature Type | Examples |
|--------------|----------|
| Structural | text length, word count |
| Punctuation | `!`, `?`, `...` counts |
| Linguistic | first-person pronoun ratio, negation ratio |
| Domain-specific | death-related words (`die`, `kill`, `suicide`, `end`) |
| Affect | sadness indicators (`hopeless`, `empty`, `lonely`) |

### TF-IDF
- 15,000 features
- 1-3 ngrams
- Sublinear term frequency scaling

---

## Installation
```bash
bash
git clone https://github.com/behnoudng/suicide-detection.git
cd suicide-detection
pip install -r requirements.txt
```

### Requirements (key dependencies)
- Python 3.10+
- PyTorch 2.9
- Transformers 4.57
- XGBoost 3.1
- scikit-learn 1.7
- NLTK 3.9

---

## Usage
### Training Pipeline

Run in order:
```bash
# 1. Prepare and split data
python src/data_prep.py

# 2. Preprocess text
python src/text_preprocessing.py

# 3. Generate features
python src/feature_engineering.py

# 4. Train XGBoost
python src/train_xgboost.py

# 5. Fine-tune BERT (see notebook)
# I recommend using Google Colab's free GPU
jupyter notebook notebooks/finetune_bert.ipynb

# 6. Run ensemble evaluation
python src/ensemble_model.py
```
### Inference
```python
from src.ensemble_model import HybridEnsemble
from scipy.sparse import load_npz

ensemble = HybridEnsemble(
    xgboost_path='data/models/xgboost_model.pkl',
    bert_path='data/models/suicide_bert_final',
    weights=(0.3, 0.7)
)

# For single prediction, you'll need to preprocess and featurize first
# See src/feature_engineering.py for the pipeline
prediction = ensemble.predict(X_features, texts, threshold=0.4)
```


---

## Project Structure

├── api/                    # FastAPI backend (coming soon)
├── app/                    # Streamlit interface (coming soon)
├── data/
│   ├── raw/                # Original dataset
│   ├── processed/          # Preprocessed splits
│   └── models/             # Trained models & artifacts
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_evaluation.ipynb
│   └── finetune_bert.ipynb
├── src/
│   ├── data_prep.py
│   ├── text_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_xgboost.py
│   ├── ensemble_model.py
│   └── tune_ensemble.py
└── tests/

---
## API & Interface

🚧 **Coming Soon**

- RESTful API with FastAPI
- Streamlit demo interface
---

## Dataset

This project uses the [Suicide and Depression Detection Dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) licensed under CC BY-SA 4.0.

- **Source:** Reddit posts from r/SuicideWatch and r/depression (labeled "suicide") vs. other subreddits (labeled "non-suicide")
- **Size:** ~232K samples
- **Balance:** Roughly 50/50 split

---

## Limitations

- Trained on Reddit data. May not generalize to other platforms or clinical text
- English only
- Not a substitute for professional mental health assessment

---

## License
MIT

---

## Acknowledgments

- Dataset by [Nikhileswar Komati](https://www.kaggle.com/nikhileswarkomati)
- BERT base model from Hugging Face

---

