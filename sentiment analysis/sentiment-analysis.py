import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('vader_lexicon')

def analyze_sentiment(headline):
    # Initialize the NLTK sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(headline)
    
    # Interpret the sentiment scores
    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Example headlines
headlines = [
"Competition probe launched into Sun and Daily Mail publishers' printing tie-up", "Prince Harry drops libel case against Daily Mail after pretrial ruling", "Labour withdraws candidate support after 'Israel remarks' controversy", "£800m budget plan to cut red tape and boost NHS and police efficiency"
]

# Analyze each headline
for headline in headlines:
    sentiment = analyze_sentiment(headline)
    print(f"Headline: {headline}")
    print(f"Sentiment: {sentiment}\n")