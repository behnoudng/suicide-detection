import numpy as np
from scipy.sparse import load_npz
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, recall_score
import pickle

X_train = load_npz("data/processed/X_train_enhanced.npz")
X_test = load_npz("data/processed/X_test_enhanced.npz")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

print(f"Training samples: {X_train.shape[0]}")
print(f"Columns: {X_train.shape[1]}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    scale_pos_weight=1,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("XGBoost Performance")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['non-suicide', 'suicide']))
suicide_recall = recall_score(y_test, y_pred, pos_label=1)
print(f"\nSuicide Class Recall: {suicide_recall:.4f}")

with open('data/models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

np.save('data/models/xgboost_test_proba.npy', y_pred_proba)

print("\nModel saved to models/xgboost_model.pkl")
print(f"Test set predictions saved")