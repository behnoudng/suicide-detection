import pandas as pd
import numpy as np
from pathlib import Path
import json
def load_raw_data(filepath="data/raw/suicide_data.csv"):
    print(f"loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"loaded {len(df)} rows")
    return df
def clean_data(df):
    print('\nclearning data...')
    df = df.drop(columns=['Unnamed: 0'], axis=1)
    missing = df.isnull().sum()
    if missing.any():
        print("found missing values:\n{missing[missing>0]}")
        df = df.dropna()
        print(f"dropped rows with missing values")
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    after = len(df)
    if before != after:
        print(f"removed {before - after} duplicate texts")
    df = df[df['text'].str.strip() != '']
    after_em = df
    if after != len(after_em):
        print("dataset now has {len(df)} rows")
    print("no empty rows found.")
    df['class'] = df['class'].str.strip().str.lower()
    return df
def check_class_balance(df):
    print("\n" + "="*50)
    print("class distribution")
    print("="*50)
    counts = df['class'].value_counts()
    percentages = df['class'].value_counts(normalize=True) * 100
    for key, count in counts.items():
        pct = percentages[key]
        print(f"{key:15s}: {count:5d} ({pct:5.1}%)")
    ratio = counts.max() / counts.min()
    if ratio > 3:
        print(f"Imbalanced dataset (ration: {ratio:.1f}:1)")
        print("Consider using SMOTE or class weights")
    else:
        print("\n Dataset is reasonably balanced")
    return counts
def split_data(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    print(f"\n" + "="*50)
    print(f"Splitting data (test_size={test_size})")
    print("="*50)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['class'])
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    return train_df, test_df

def save_processed_data(train_df, test_df, output_dir='data/processed'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n" + "="*50)
    print("Saving processed data...")
    print("="*50)
    train_path = output_path / 'train.csv'
    test_path = output_path / 'test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    metadata = {
        'train_size': len(train_df),
        'test_size': len(test_df),
        'total_size': len(train_df) + len(test_df),
        'classes': train_df['class'].value_counts().to_dict(),
        'test_split': 0.2,
    }
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")

def main():
    print("\n" + "="*50)
    print("SUICIDE DETECTION DATA PREPROCESSING")
    print("="*50)
    df = load_raw_data()
    df = clean_data(df)
    check_class_balance(df)
    train_df, test_df = split_data(df)
    save_processed_data(train_df, test_df)
    print("\n", "="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Open notebook/01_eda.ipynb to explore the data")
    print("2. Then run notebooks/02_modeling.ipynb to train models")
if __name__ == '__main__':
    main()