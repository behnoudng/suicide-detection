import numpy as np
import pandas as pd
import pickle
from scipy.sparse import load_npz
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, recall_score, confusion_matrix, precision_score

class HybridEnsemble:
    def __init__(self, xgboost_path, bert_path, weights=(0.4, 0.6)):
        with open(xgboost_path, 'rb') as f:
            self.xgb_model = pickle.load(f)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
        self.bert_model.eval()

        self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bert_model.to(self.device)

    def predict_xgboost(self, X_features):
        return self.xgb_model.predict_proba(X_features)[:, 1]
    
    def predict_bert(self, texts, batch_size=64):
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = self.bert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt',
            ).to(self.device)

            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def predict_proba(self, X_features, texts):
        xgb_probs = self.predict_xgboost(X_features)
        bert_probs = self.predict_bert(texts)

        ensemble_probs = (
            self.weights[0] * xgb_probs +
            self.weights[1] * bert_probs
        )

        return ensemble_probs, xgb_probs, bert_probs
    
    def predict(self, X_features, texts, threshold=0.5):
        probs, _, _ = self.predict_proba(X_features, texts)
        return (probs >= threshold).astype(int)
    
if __name__ == '__main__':
    X_test = load_npz('data/processed/X_test_enhanced.npz')
    print(f"X_test:\n {X_test}")
    test_df = pd.read_csv("data/processed/bert_test.csv")
    print(f"test_df:\n {test_df}")
    y_test = test_df['label'].values
    print(f"y_test:\n {y_test}")
    texts_test = test_df['text'].astype(str).tolist()
    print(f"texts_test: \n {texts_test}")    


    ensemble = HybridEnsemble(
        xgboost_path='data/models/xgboost_model.pkl',
        bert_path='data/models/suicide_bert_final',
        weights=(0.3, 0.7)
    )

    print("Generating ensemble predictions...")
    ensemble_probs, xgb_probs, bert_probs = ensemble.predict_proba(X_test, texts_test)
    np.save("models/ensemble_test_proba.npy", ensemble_probs)

    print("\n" + "=" * 60)
    print("ENSEMBLE PERFORMANCE")
    print("=" * 60)

    for threshold in [0.3, 0.3, 0.5, 0.6, 0.7]:
        y_pred = (ensemble_probs >= threshold).astype(int)
        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1)

        print(f"\nThreshold: {threshold}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")

    y_pred_final = (ensemble_probs >= 0.4).astype(int)

    print("\n" + "=" * 60)
    print("final ensemble results (threshold=0.4 for high recall)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_final, target_names=['non-suicide', 'suicide']))

    xgb_pred = (xgb_probs >= 0.5).astype(int)
    bert_pred = (bert_probs >= 0.5).astype(int)

    print("\n" + "=" * 60)
    print("model comparison")
    print("=" * 60)
    print(f"XGBoost recall: {recall_score(y_test, xgb_pred, pos_label=1):.4f}")
    print(f"BERT recall: {recall_score(y_test, bert_pred, pos_label=1):.4f}")
    print(f"Ensemble recall: {recall_score(y_test, y_pred_final, pos_label=1):.4f}")
