import re
import time
import logging
import random
import threading
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from queue import Queue

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
MAX_THREADS = 8
TWEETS_PER_HASHTAG = MIN_TWEETS // len(HASHTAGS)
TIME_WINDOW_HOURS = 24

# =========================
# Twitter Login Credentials
# =========================
# You can set these as environment variables for security, or hardcode for testing (not recommended)
TWITTER_USERNAME = os.environ.get("TWITTER_USERNAME", "")
TWITTER_PASSWORD = os.environ.get("TWITTER_PASSWORD", "")

def extract_hashtags(text: str) -> List[str]:
    return re.findall(r"#(\w+)", text)

def extract_mentions(text: str) -> List[str]:
    return re.findall(r"@(\w+)", text)

def clean_text(text: str) -> str:
    # Remove URLs, mentions, emojis, and normalize whitespace
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# =========================
# Twitter Login Helper
# =========================

def twitter_login(driver, username: str, password: str, timeout: int = 30) -> bool:
    """
    Log in to Twitter using Selenium WebDriver.
    Returns True if login is successful, False otherwise.
    """
    try:
        driver.get("https://twitter.com/login")
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.NAME, "text"))
        )
        # Enter username
        user_input = driver.find_element(By.NAME, "text")
        user_input.clear()
        user_input.send_keys(username)
        user_input.send_keys(Keys.RETURN)
        time.sleep(2)
        # Twitter may ask for email/phone confirmation, handle if needed
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.NAME, "password"))
            )
        except TimeoutException:
            # Sometimes Twitter asks for phone/email confirmation
            try:
                alt_input = driver.find_element(By.NAME, "text")
                alt_input.clear()
                alt_input.send_keys(username)
                alt_input.send_keys(Keys.RETURN)
                time.sleep(2)
            except Exception:
                pass
        # Enter password
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.NAME, "password"))
        )
        pass_input = driver.find_element(By.NAME, "password")
        pass_input.clear()
        pass_input.send_keys(password)
        pass_input.send_keys(Keys.RETURN)
        # Wait for home page to load
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//a[@href='/home']"))
        )
        logging.info("Twitter login successful.")
        return True
    except Exception as e:
        logging.error(f"Twitter login failed: {e}")
        return False

# =========================
# Scraper Implementation (Selenium)
# =========================

class TwitterScraper:
    def __init__(self, hashtags: List[str], min_tweets: int, time_window_hours: int = 24):
        self.hashtags = hashtags
        self.min_tweets = min_tweets
        self.time_window = timedelta(hours=time_window_hours)
        self.tweets: List[Dict[str, Any]] = []
        self.seen_ids: Set[str] = set()

    def get_driver(self):
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--headless")
        # Set a random user agent for each session
        chrome_options.add_argument(f'user-agent={random.choice(HEADERS_LIST)}')
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(60)
        return driver

    def scrape_hashtag(self, hashtag: str):
        logging.info(f"Scraping for hashtag: {hashtag}")
        collected = 0
        max_attempts = 30
        attempts = 0
        scroll_attempts = 0
        last_height = 0
        try:
            driver = self.get_driver()
            # --- Twitter login before scraping ---
            if not TWITTER_USERNAME or not TWITTER_PASSWORD:
                logging.error("Twitter credentials not set. Set TWITTER_USERNAME and TWITTER_PASSWORD as environment variables.")
                return
            login_success = twitter_login(driver, TWITTER_USERNAME, TWITTER_PASSWORD)
            if not login_success:
                logging.error("Twitter login failed. Skipping hashtag scraping.")
                try:
                    driver.quit()
                except Exception:
                    pass
                return
            # --- End login ---
            url = SEARCH_URL.format(query=hashtag.replace("#", "%23") + "%20lang%3Aen")
            driver.get(url)
            time.sleep(random.uniform(3, 6))
            while collected < TWEETS_PER_HASHTAG and attempts < max_attempts and scroll_attempts < 50:
                try:
                    articles = driver.find_elements(By.TAG_NAME, "article")
                    if not articles:
                        logging.warning(f"No articles found for {hashtag} on attempt {attempts}")
                        time.sleep(random.uniform(2, 5))
                        attempts += 1
                        continue
                    for article in articles:
                        try:
                            tweet = self.parse_tweet(article)
                            if not tweet:
                                continue
                            tweet_id = tweet.get("tweet_id")
                            if tweet_id and tweet_id in self.seen_ids:
                                continue
                            self.tweets.append(tweet)
                            if tweet_id:
                                self.seen_ids.add(tweet_id)
                            collected += 1
                            if collected >= TWEETS_PER_HASHTAG:
                                break
                        except Exception as e:
                            logging.debug(f"Error parsing tweet: {e}")
                    # Scroll to load more tweets
                    driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
                    time.sleep(random.uniform(2.5, 3.5))
                    new_height = driver.execute_script("return document.body.scrollHeight")
                    if new_height == last_height:
                        scroll_attempts += 1
                    else:
                        scroll_attempts = 0
                    last_height = new_height
                    attempts += 1
                except Exception as e:
                    logging.error(f"Error scraping {hashtag}: {e}")
                    time.sleep(random.uniform(3, 7))
                    attempts += 1
        except Exception as e:
            logging.error(f"WebDriver error for {hashtag}: {e}")
        finally:
            try:
                driver.quit()
            except Exception:
                pass
        logging.info(f"Finished scraping {hashtag}: {collected} tweets collected.")
        return collected

    def parse_tweet(self, article) -> Dict[str, Any]:
        # Selenium WebElement parsing
        try:
            # Username
            username = "unknown"
            try:
                username_tag = article.find_element(By.XPATH, ".//a[starts-with(@href, '/') and not(contains(@href, '/status/'))]")
                username = username_tag.text.strip()
            except NoSuchElementException:
                pass
            # Timestamp
            try:
                time_tag = article.find_element(By.XPATH, ".//time")
                timestamp = time_tag.get_attribute("datetime")
            except NoSuchElementException:
                timestamp = datetime.utcnow().isoformat()
            # Tweet content
            try:
                content_div = article.find_element(By.XPATH, ".//div[@data-testid='tweetText']")
                content = content_div.text.strip()
            except NoSuchElementException:
                content = ""
            # Engagement metrics (likes, retweets, replies)
            metrics = {"likes": 0, "retweets": 0, "replies": 0}
            try:
                like_divs = article.find_elements(By.XPATH, ".//div[@data-testid='like']")
                retweet_divs = article.find_elements(By.XPATH, ".//div[@data-testid='retweet']")
                reply_divs = article.find_elements(By.XPATH, ".//div[@data-testid='reply']")
                for div in like_divs:
                    txt = div.get_attribute("aria-label") or div.text
                    metrics["likes"] = self.extract_metric(txt)
                for div in retweet_divs:
                    txt = div.get_attribute("aria-label") or div.text
                    metrics["retweets"] = self.extract_metric(txt)
                for div in reply_divs:
                    txt = div.get_attribute("aria-label") or div.text
                    metrics["replies"] = self.extract_metric(txt)
            except Exception:
                pass
            # Mentions and hashtags
            mentions = extract_mentions(content)
            hashtags = extract_hashtags(content)
            # Tweet ID
            tweet_id = None
            try:
                tweet_link = article.find_element(By.XPATH, ".//a[contains(@href, '/status/')]")
                href = tweet_link.get_attribute("href")
                match = re.search(r"/status/(\d+)", href)
                if match:
                    tweet_id = match.group(1)
            except NoSuchElementException:
                pass
            return {
                "tweet_id": tweet_id or "",
                "username": username,
                "timestamp": timestamp,
                "content": clean_text(content),
                "likes": metrics["likes"],
                "retweets": metrics["retweets"],
                "replies": metrics["replies"],
                "mentions": mentions,
                "hashtags": hashtags
            }
        except Exception as e:
            logging.debug(f"parse_tweet error: {e}")
            return {}

    def extract_metric(self, text: str) -> int:
        # Extracts numbers from engagement text
        match = re.search(r"(\d[\d,]*)", text)
        if match:
            return int(match.group(1).replace(",", ""))
        return 0

    def run(self):
        total_collected = 0
        for hashtag in self.hashtags:
            count = self.scrape_hashtag(hashtag)
            total_collected += count
        logging.info(f"Total tweets collected: {total_collected}")
