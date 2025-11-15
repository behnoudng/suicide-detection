#3
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self, max_features=5000):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
            stop_words="english"
        )
        self.encoder = LabelEncoder()
    def fit_transform(self, X_train, y_train):
        X_vec = self.vectorizer.fit_transform(X_train)
        y_enc = self.encoder.fit_transform(y_train)
        print(f"TF-IDF matrix shape: {X_vec.shape}")
        return X_vec, y_enc
    
    def transform(self, X_test, y_test=None):
        X_vec = self.vectorizer.transform(X_test)
        y_enc = self.encoder.transform(y_test) if y_test is not None else None
        print(f"Transformed test data: {X_vec.shape}")
        return X_vec, y_enc
    
    def save(self, path="models/feature_engineering.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "encoder": self.encoder
            }, f)
        print(f"Saved vectorizer + encoder inside {path}")
    
    @staticmethod
    def load(path="models/feature_engineering.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        fe = FeatureEngineer()
        fe.vectorizer = data["vectorizer"]
        fe.encoder = data["encoder"]
        return fe
    
def main():
    print("FEATURE ENGINEERING")
    data_dir = Path("data/processed")
    train_csv = data_dir / "train_preprocessed.csv"
    test_csv = data_dir / "test_preprocessed.csv"
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(
        train_df["processed_text"], train_df["class"]
    )
    X_test, y_test = fe.transform(
        test_df["processed_text"], train_df["class"]
    )
    fe.save()

    print("\n Top 15 vocab terms:")
    vocab = list(fe.vectorizer.vocabulary_.keys())[:15]
    print(vocab)
    print("\nEncoded labels:", fe.encoder.classes_)
    print("Train matrix shape: ", X_train.shape, y_train.shape)
    print("Test matrix shape:", X_test.shape, y_test.shape)
    print("\nfeature engineering complete\n")
    print("Next step: train multiple models")

if __name__ == "__main__":
    main()