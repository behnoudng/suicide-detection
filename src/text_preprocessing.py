#2
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path

def download_nltk_data():
    print("Downloading NLTK data...")
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("wordnet", quiet=True)
        nltk.download("stopwords", quiet=True)
        print("NLTK data downloaded")
    except Exception as e:
        print(f"Warning: {e}")
class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
            self.stop_words -= {"not", "no", "never", "nothing", "nobody", "nowhere", "none", "cannot", "don't", "won't", "ain't", "wouldn't"}
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        # remove urls
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # remove reddit notations
        text = re.sub(r"\[removed\]|\[deleted\]", "", text)
        # remove email addresses
        text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\'\-]', '', text)
        return text
    
    def tokenize(self, text):
        try: 
            tokens = word_tokenize(text)
        except:
            print("Tokenization failed.")
        return tokens
    
    def remove_stop_words(self, tokens):
        if not self.remove_stopwords:
            return tokens
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        if not self.lemmatize:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text):
        text = self.clean_text(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.lemmatize_tokens(tokens)
        # remove single characters and number-only tokens
        tokens = [t for t in tokens if len(t) > 1 and not t.isdigit()]
        return ' '.join(tokens)
    
def preprocess_dataset(input_path, output_path, preprocessor):
    print(f"Processing {input_path}")
    df = pd.read_csv(input_path)
    print(f"Preprocessing {len(df)} texts...")
    df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    before = len(df)
    df = df[df['processed_text'].str.strip() != '']
    after = len(df)
    if before != after:
        print(f"Removed {before - after} empty texts after preprocessing")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df

def main():
    print("\n" + "=" * 50)
    print("TEXT PREPROCESSING")
    print("="*50)
    download_nltk_data()
    preprocessor = TextPreprocessor(
        remove_stopwords = True,
        lemmatize = True
    )
    # exmaple to understand how it works
    print("\n" + "="*50)
    print("EXAMPLE PREPROCESSING")
    print("="*50)
    example = "I'm feeling really depressed today and I don't know what to do. https://example.com"
    print(f"\nOriginal:\n{example}")
    processed = preprocessor.preprocess(example)
    print(f"\nProcessed:\n{processed}")
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    print("\n"+"="*50)
    print("PROCESSING DATASETS")
    print("=" * 50)
    data_dir = Path('data/processed')
    train_df = preprocess_dataset(
        input_path = data_dir / 'train.csv',
        output_path = data_dir / 'train_preprocessed.csv',
        preprocessor = preprocessor
    )
    test_df = preprocess_dataset(
        input_path = data_dir / 'test.csv',
        output_path = data_dir / 'test_preprocessed.csv',
        preprocessor = preprocessor
    )

    print("\n" + '='*50)
    print("PREPROCESSING STATS")
    print("="*50)
    def show_stats(df, name):
        print(f"\n{name}:")
        avg_len_before = df['text'].str.split().str.len().mean()
        avg_len_after = df['processed_text'].str.split().str.len().mean()
        print(f"Avg words before: {avg_len_before:.1f}")
        print(f"Avg words after: {avg_len_after:.1f}")
        print(f"Reduction: {(1 - avg_len_after / avg_len_before) * 100:.1f}%")
    show_stats(train_df, "Training set")
    show_stats(test_df, "Test set")

    print("\n"+"="*50)
    print("TEXT PREPROCESSING COMPLETE")
    print("="*50)
    print("\nNext step: Feature Engineering (TF-IDF/Word Embeddings)")

if __name__ == '__main__':
    main()