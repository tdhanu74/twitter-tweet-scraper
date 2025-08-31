# Twitter Hashtag Scraper & Signal Extractor

This project scrapes tweets for Indian stock market hashtags (e.g., #nifty50, #sensex, #intraday, #banknifty) from Twitter, processes and deduplicates the data, and converts tweet text into numerical signals for downstream analysis or trading models.

## Approach and Features

- Scraping using Headless Selenium WebDriver to search different hashtags and scroll through all the posts for those hashtag and extract their data
- Rotates user agents for anti-bot evasion
- Delays scrolling with random intervals for anti-bot evasion and not hitting the Rate limit for the page
- Cleans, deduplicates, and stores tweets in Parquet format
- Extracts hashtags, mentions, and engagement metrics
- Converts tweet text to TF-IDF signals
- Provides memory-efficient visualization of signals
- Logs all activity to `scraper.log`

## Requirements

- Python 3.8+
- Chrome browser (or compatible with Selenium)
- ChromeDriver (matching your Chrome version)
- Twitter account credentials (for login)
- The following Python packages:
  - selenium
  - pandas
  - pyarrow
  - numpy
  - scikit-learn
  - matplotlib



## Steps for running the project with uv:

Create `.env` file with and add a dummy Twitter account username and password as shown in the `.env.local` file

```bash
uv sync
uv run --env-file=.env main.py
```

## Steps for running the project with pip:

### Windows: 
Update `venv\Scripts\activate`, add `set TWITTER_USERNAME="YOUR_TWITTER_USERNAME"` and `set TWITTER_PASSWORD="YOUR_TWITTER_PASSWORD"`
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
### Linux:
Update `venv/bin/activate`, add `export TWITTER_USERNAME="YOUR_TWITTER_USERNAME"` and `export TWITTER_PASSWORD="YOUR_TWITTER_PASSWORD"` 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```