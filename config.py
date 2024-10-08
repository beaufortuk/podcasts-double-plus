# config.py

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the RSS feed URL from the environment variable
PODCAST_RSS_FEED = os.getenv('PODCAST_RSS_FEED')

# Get the Hugging Face API token from the environment variable
HUGGING_FACE_TOKEN = os.getenv('HUGGING_FACE_TOKEN')

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory where episodes are stored
EPISODES_DIR = os.path.join(BASE_DIR, 'episodes')

# Directory for embeddings (e.g., speaker embeddings)
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings')

# Path to episode_metadata.json
EPISODE_METADATA_PATH = os.path.join(BASE_DIR, 'episode_metadata.json')

# Directory for assets (e.g., known speaker audio samples)
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# How many eps will be analysed. 
NUMBER_OF_EPS_TO_ANALYSE = 1000
