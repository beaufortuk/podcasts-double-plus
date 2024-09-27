# config.py

import os

# Hugging Face Token
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

if HUGGING_FACE_TOKEN is None:
    print("Hugging Face token not found. Please set the 'HUGGING_FACE_TOKEN' environment variable.")
    # Optionally, you can exit the script or set a default value
    # sys.exit(1)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory where episodes are stored
EPISODES_DIR = os.path.join(BASE_DIR, 'episodes')

# Directory for embeddings (e.g., speaker embeddings)
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')

# Directory for assets (e.g., known speaker audio samples)
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Other configuration variables can be added here
PODCAST_RSS_FEED = "https://feeds.economist.com/v1/rss/the-economist-podcasts/52522270-d2b1-484e-929f-d9bf2356ab7e"
