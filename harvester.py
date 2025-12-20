import argparse
import sys
import os
import joblib
import pandas as pd
import numpy as np
import re
from collections import Counter
from itertools import pairwise
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import STOPWORDS
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import shared logic
import text_processor as tp

def setup_args():
    parser = argparse.ArgumentParser(description="Harvester: Large Scale Text Processor")
    parser.add_argument("--input", required=True, help="Path to input file (CSV/Excel) or folder")
    parser.add_argument("--col", required=True, help="Name of the text column to analyze")
    parser.add_argument("--output", default="sketch.pkl", help="Output path for the sketch file")
    parser.add_argument("--chunksize", type=int, default=50000, help="Rows per chunk")
    parser.add_argument("--topics", type=int, default=5, help="Number of LDA topics")
    parser.add_argument("--no-bigrams", action="store_true", help="Disable bigram calculation")
    return parser.parse_args()

def stream_file(file_path, chunksize, text_col):
    """Yields lists of raw text strings from CSV or Excel."""
    ext = file_path.lower().split(".")[-1]
    
    if ext == 'csv':
        # CSV Streaming
        for chunk in pd.read_csv(file_path, usecols=[text_col], chunksize=chunksize, on_bad_lines='skip', encoding='utf-8', engine='python'):
            yield chunk[text_col].dropna().astype(str).tolist()
            
    elif ext in ['xlsx', 'xlsm']:
        # Excel Streaming (slower, but memory safe-ish via openpyxl if handled carefully, but pandas read_excel doesn't chunk well)
        # For massive Excel, we load full columns or convert to CSV first. 
        # Here we use standard pandas load for Excel as chunking is limited support in engines.
        # Warning: Excel is hard to stream. We assume if it's 10M rows it's CSV.
        print(f"Loading Excel file {file_path}...")
        df = pd.read_excel(file_path, usecols=[text_col])
        # Manually chunk the dataframe
        for i in range(0, len(df), chunksize):
            yield df.iloc[i:i+chunksize][text_col].dropna().astype(str).tolist()

def main():
    args = setup_args()
    
    # 1. Setup cleaning resources
    try: nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError: nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    
    stopwords = set(STOPWORDS)
    stopwords.update(tp.default_prepositions())
    trans_map = tp.build_punct_translation(keep_hyphens=False, keep_apostrophes=False)
    
    # 2. State Aggregators
    word_counter = Counter()
    bigram_counter = Counter()
    pos_count = 0
    neg_count = 0
    total_docs = 0
    
    pos_thresh = 0.05
    neg_thresh = -0.05
    
    print(f"--- HARVESTER STARTED ---")
    print(f"Input: {args.input} | Column: {args.col}")
    
    # --- PASS 1: VOCABULARY & COUNTERS ---
    print(">>> Phase 1: Building Vocabulary and Counters...")
    
    files = [args.input] if os.path.isfile(args.input) else [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(('csv', 'xlsx'))]
    
    for fpath in files:
        for chunk_idx, texts in enumerate(stream_file(fpath, args.chunksize, args.col)):
            clean_chunk = []
            for text in texts:
                tokens = tp.clean_and_tokenize(
                    text, True, True, True, True, trans_map, stopwords, None, 2, True
                )
                if not tokens: continue
                
                # Sentiment (approximate based on raw text or tokens? using tokens for speed here)
                # Note: VADER works best on raw text, but for speed on 10M rows, we check tokens or simple heuristics?
                # Let's do VADER on the raw sentence for accuracy, but it is slow. 
                # OPTIMIZATION: Check sentiment of the cleaning result to be consistent with Viewer.
                
                # Update Counters
                word_counter.update(tokens)
                if not args.no_bigrams and len(tokens) > 1:
                    bigram_counter.update(tuple(pairwise(tokens)))
                
                # Bayesian Aggregates (Token level approximation to match Viewer logic roughly or Text level)
                # The viewer sums counts of positive words. We do the same.
                # This is much faster than running VADER on every sentence of 10M rows.
                # We will re-calc this at the end based on the word_counter! 
                # (See logic: we don't need to do it per row if we have the global word counts and a sentiment dictionary).
                
                clean_chunk.append(" ".join(tokens))
            
            total_docs += len(texts)
            print(f"   Processed chunk {chunk_idx+1} (approx {total_docs} rows)...")

    print(f"Phase 1 Complete. Vocab Size: {len(word_counter)}")
    
    # --- PASS 2: TOPIC MODELING (ONLINE LDA) ---
    print(">>> Phase 2: Training Online LDA...")
    
    # Limit vocab to top 10,000 words for LDA stability and speed
    top_vocab = [w for w, c in word_counter.most_common(10000)]
    vectorizer = CountVectorizer(vocabulary=top_vocab) 
    
    lda = LatentDirichletAllocation(
        n_components=args.topics,
        learning_method='online',
        random_state=42,
        batch_size=args.chunksize,
        max_iter=1 # 1 pass over data is usually enough for "sketching" massive data
    )
    
    # We must stream again to partial_fit
    for fpath in files:
        for chunk_idx, texts in enumerate(stream_file(fpath, args.chunksize, args.col)):
            # Clean again (CPU bound, but necessary to keep RAM low)
            clean_docs = []
            for text in texts:
                tokens = tp.clean_and_tokenize(text, True, True, True, True, trans_map, stopwords, None, 2, True)
                if tokens: clean_docs.append(" ".join(tokens))
            
            if clean_docs:
                X = vectorizer.transform(clean_docs)
                lda.partial_fit(X)
                print(f"   LDA Trained on chunk {chunk_idx+1}")

    # --- FINAL SENTIMENT AGGREGATION ---
    # We calculate Pos/Neg counts based on the global vocabulary to save compute time
    # This matches the "Viewer" logic: pos_count = sum(counts[w] for w in sent if > thresh)
    print(">>> Finalizing Sentiment Stats...")
    vocab_sents = {w: sia.polarity_scores(w)['compound'] for w in word_counter.keys()}
    
    final_pos_count = sum(word_counter[w] for w, s in vocab_sents.items() if s >= pos_thresh)
    final_neg_count = sum(word_counter[w] for w, s in vocab_sents.items() if s <= neg_thresh)

    # --- SERIALIZATION ---
    sketch = {
        "word_counter": word_counter,
        "bigram_counter": bigram_counter,
        "lda_model": lda,
        "lda_feature_names": vectorizer.get_feature_names_out(),
        "pos_count": final_pos_count,
        "neg_count": final_neg_count,
        "total_rows": total_docs,
        "metadata": {"source": args.input, "col": args.col}
    }
    
    joblib.dump(sketch, args.output)
    print(f"✅ Sketch saved to: {args.output}")
    print(f"   Total Rows: {total_docs}")
    print(f"   Unique Words: {len(word_counter)}")

if __name__ == "__main__":
    main()
