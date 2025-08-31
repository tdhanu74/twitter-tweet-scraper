import re
import time
import logging
import random
import threading
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from queue import Queue

from twitter_scraper import TwitterScraper

import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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
SEARCH_URL = "https://twitter.com/search?q={query}&src=typed_query&f=live"
HEADERS_LIST = [
    # List of user agents to rotate for anti-bot
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]
DATA_PATH = "tweets.parquet"
MAX_THREADS = 1
TWEETS_PER_HASHTAG = MIN_TWEETS // len(HASHTAGS)
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
