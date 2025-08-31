import logging
from typing import List, Dict, Any, Set

from twitter_scraper import TwitterScraper

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# =========================
# Logging Configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)

# =========================
# Constants & Config
# =========================
HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
MIN_TWEETS = 2000
DATA_PATH = "tweets.parquet"
TIME_WINDOW_HOURS = 24

# =========================
# Utility Functions
# =========================

def deduplicate_tweets(tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    deduped = []
    for tweet in tweets:
        key = tweet["username"] + tweet["timestamp"] + tweet["content"]
        if key not in seen:
            deduped.append(tweet)
            seen.add(key)
    return deduped

# =========================
# Data Processing & Storage
# =========================

def process_and_store(tweets: List[Dict[str, Any]], path: str):
    # Deduplicate
    tweets = deduplicate_tweets(tweets)
    # DataFrame construction
    df = pd.DataFrame(tweets)
    # Clean and normalize
    if "content" in df.columns:
        df["content"] = df["content"].astype(str)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Handle Unicode and special characters
        df["content"] = df["content"].apply(lambda x: x.encode("utf-8", "ignore").decode("utf-8"))
        # Store as Parquet
        df.to_parquet(path, index=False)
        logging.info(f"Stored {len(df)} tweets to {path}")
    return df

# =========================
# Text-to-Signal Conversion
# =========================

def text_to_signal(df: pd.DataFrame) -> np.ndarray:
    # Use TF-IDF for vectorization
    vectorizer = TfidfVectorizer(
        max_features=256,
        stop_words="english",
        strip_accents="unicode",
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(df["content"].values)
    return tfidf_matrix.toarray()

def aggregate_signals(signals: np.ndarray) -> Dict[str, Any]:
    # Composite signal: mean, std, confidence interval
    mean_signal = np.mean(signals, axis=0)
    std_signal = np.std(signals, axis=0)
    n = signals.shape[0]
    ci95 = 1.96 * std_signal / np.sqrt(n) if n > 0 else 0
    return {
        "mean": mean_signal,
        "std": std_signal,
        "ci95": ci95
    }

# =========================
# Memory-Efficient Visualization
# =========================

def plot_signals(signals: np.ndarray, sample_size: int = 500):
    # Sample for memory efficiency
    if signals.shape[0] > sample_size:
        idx = np.random.choice(signals.shape[0], sample_size, replace=False)
        sampled = signals[idx]
    else:
        sampled = signals
    plt.figure(figsize=(10, 4))
    plt.plot(np.mean(sampled, axis=1))
    plt.title("Mean TF-IDF Signal per Tweet (Sampled)")
    plt.xlabel("Sampled Tweet Index")
    plt.ylabel("Mean TF-IDF Value")
    plt.tight_layout()
    plt.show()

# =========================
# Main Pipeline
# =========================

def main():
    scraper = TwitterScraper(HASHTAGS, MIN_TWEETS, TIME_WINDOW_HOURS)
    scraper.run()
    if len(scraper.tweets) < MIN_TWEETS:
        logging.warning(f"Only {len(scraper.tweets)} tweets collected, less than target {MIN_TWEETS}")
    print(scraper.tweets)
    if scraper.tweets:
        df = process_and_store(scraper.tweets, DATA_PATH)
        signals = text_to_signal(df)
        agg = aggregate_signals(signals)
        logging.info(f"Signal mean shape: {agg['mean'].shape}, std: {agg['std'].mean():.4f}, ci95: {agg['ci95'].mean():.4f}")
        plot_signals(signals)
        # Save signals for downstream trading models
        np.save("signals.npy", signals)

if __name__ == "__main__":
    main()
