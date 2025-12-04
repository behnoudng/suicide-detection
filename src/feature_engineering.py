#3
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re

class FeatureEngineer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )
        self.encoder = LabelEncoder()

    def extract_custom_features(self, texts):
        """Extract psychological features"""
        features = []
        
        for text in texts:
            feat_dict = {}
            
            feat_dict['text_length'] = len(text)
            feat_dict['word_count'] = len(text.split())
            
            feat_dict['exclamation_count'] = text.count('!')
            feat_dict['question_count'] = text.count('?')
            feat_dict['ellipsis_count'] = text.count('...')
            
            first_person = ['i ', 'me ', 'my ', 'myself ', 'mine ']
            text_lower = ' ' + text.lower() + ' '
            feat_dict['first_person_count'] = sum(text_lower.count(p) for p in first_person)
            
            negations = ['not ', 'no ', "n't ", 'never ', 'nothing ']
            feat_dict['negation_count'] = sum(text_lower.count(n) for n in negations)
            
            death_words = ['die', 'death', 'kill', 'suicide', 'end', 'gone']
            feat_dict['death_word_count'] = sum(text_lower.count(w) for w in death_words)
            
            sadness = ['sad', 'depressed', 'hopeless', 'empty', 'lonely']
            feat_dict['sadness_count'] = sum(text_lower.count(w) for w in sadness)
            
            if feat_dict['word_count'] > 0:
                feat_dict['first_person_ratio'] = feat_dict['first_person_count'] / feat_dict['word_count']
                feat_dict['negation_ratio'] = feat_dict['negation_count'] / feat_dict['word_count']
                feat_dict['death_ratio'] = feat_dict['death_word_count'] / feat_dict['word_count']
            else:
                feat_dict['first_person_ratio'] = 0
                feat_dict['negation_ratio'] = 0
                feat_dict['death_ratio'] = 0
            features.append(feat_dict)
        return pd.DataFrame(features)
    
    def fit_transform(self, texts, labels):
        tfidf_features = self.tfidf.fit_transform(texts)
        custom_features = self.extract_custom_features(texts)
        encoded_labels = self.encoder.fit_transform(labels)
        return tfidf_features, custom_features, encoded_labels
    
    def transform(self, texts):
        tfidf_features = self.tfidf.transform(texts)
        custom_features = self.extract_custom_features(texts)
        return tfidf_features, custom_features
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'label_encoder': self.encoder
            }, f)


if __name__ == '__main__':
    train = pd.read_csv('data/processed/train_preprocessed.csv')
    test = pd.read_csv('data/processed/test_preprocessed.csv')
    
    fe = FeatureEngineer()
    
    tfidf_train, custom_train, y_train = fe.fit_transform(
        train['processed_text'].values,
        train['class'].values
    )
    
    tfidf_test, custom_test = fe.transform(test['processed_text'].values)
    y_test = fe.encoder.transform(test['class'].values)
    
    from scipy.sparse import save_npz, hstack
    from scipy.sparse import csr_matrix

    X_train = hstack([tfidf_train, csr_matrix(custom_train.values)])
    X_test = hstack([tfidf_test, csr_matrix(custom_test.values)])
    
    save_npz('data/processed/X_train_enhanced.npz', X_train)
    save_npz('data/processed/X_test_enhanced.npz', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    fe.save('models/feature_engineering_v2.pkl')
    
    print(f"Enhanced features shape: {X_train.shape}")
    print(f"TF-IDF features: {tfidf_train.shape[1]}")
    print(f"Custom features: {custom_train.shape[1]}")