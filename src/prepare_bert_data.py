import pandas as pd
from sklearn.model_selection import train_test_split

train_o = pd.read_csv("data/processed/train_preprocessed.csv")
test_o = pd.read_csv("data/processed/test_preprocessed.csv")

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_o["text"].values,
    train_o['class'].values,
    test_size = 0.1,
    stratify=train_o['class'],
    random_state=42
)

bert_train = pd.DataFrame({
    'text': train_texts,
    'label': [1 if l == 'suicide' else 0 for l in train_labels]
})

bert_val = pd.DataFrame({
    'text': val_texts,
    'label': [1 if l == 'suicide' else 0 for l in val_labels]
})

bert_test = pd.DataFrame({
    'text': test_o['text'].values,
    'label': [1 if l == 'suicide' else 0 for l in test_o['class'].values]
})

bert_train.to_csv("data/processed/bert_train.csv", index=False)
bert_val.to_csv("data/processed/bert_val.csv", index=False)
bert_test.to_csv("data/processed/bert_test.csv", index=False)