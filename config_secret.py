# After filling API credentials, change the file name to config.py from config_secret.py

# Twitter API credentials
# Fill these in after setting up your Twitter Developer account
TWITTER_API_KEY = ""
TWITTER_API_SECRET = ""
TWITTER_ACCESS_TOKEN = ""
TWITTER_ACCESS_SECRET = ""
TWITTER_BEARER_TOKEN = ""

# Search parameters
SEARCH_TERMS = {
    "iphone": ["iPhone", "Apple iPhone", "#iPhone", "iphone", "iPhones", "iphones", "Iphone", "IPhone"],
    "galaxy": ["Samsung Galaxy", "Galaxy S", "Galaxy Note", "#SamsungGalaxy", "samsung", "samsung galaxy", "#Samsung", "S23"]
}

# Number of tweets to collect per search term (change to avoid rate limits)
MAX_TWEETS = 25

# Date ranges - use recent dates (last 7 days)
# Leave these blank to use the default 7-day window
START_DATE = "2023-01-01"  # Twitter will use 7 days ago by default
END_DATE = ""    # Twitter will use now by default

# Dashboard settings
DASHBOARD_TITLE = "Smartphone Sentiment Analysis"
DASHBOARD_SUBTITLE = "Analyzing consumer sentiment for iPhone vs. Samsung Galaxy"