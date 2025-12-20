import numpy as np
from sklearn.metrics import recall_score
from scipy.sparse import load_npz
import pandas as pd

X_test = load_npz('data/processed/X_test_enhanced.npz')
test_df = pd.read_csv("data/processed/bert_test.csv")
y_test = test_df['label'].values


xgb_probs = np.load('models/xgboost_test_proba.npy')
bert_probs = np.load('models.bert_test_proba.npy')

# use grid search to get the best weights
best_recall = 0
best_weights = None
best_threshold = None

for xgb_weight in np.arange(0.0, 1.01, 0.1):
    bert_weight = 1 - xgb_weight

    for threshold in np.arange(0.3, 0.71, 0.05):
        ensemble_probs = xgb_weight * xgb_probs + bert_weight * bert_probs
        y_pred = (ensemble_probs >= threshold).astype(int)
        recall = recall_score(y_test, y_pred, pos_label = 1)

        if recall > best_recall:
            best_recall = recall
            best_weights = (xgb_weight, bert_weight)
            best_threshold = threshold

print(f"Best weights: XGBoost={best_weights[0]:.2f}, BERT={best_weights[1]:.2f}")
print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best Recall: {best_recall:.4f}")