# recommendations.py

import os
import json
import random
from collections import defaultdict
from difflib import SequenceMatcher
from nltk.corpus import wordnet
import nltk

# Import configurations
import config

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load metadata
def load_metadata():
    base_dir = config.BASE_DIR

    # Paths to metadata files
    speakers_path = os.path.join(base_dir, 'speakers.json')
    episode_topics_path = os.path.join(base_dir, 'episode_topics.json')
    episode_entities_path = os.path.join(base_dir, 'episode_entities.json')
    episode_sentiments_path = os.path.join(base_dir, 'episode_sentiments.json')
    episode_humor_scores_path = os.path.join(base_dir, 'episode_humor_scores.json')
    episode_metadata_path = os.path.join(base_dir, 'episode_metadata.json')  # Added

    # Load data
    with open(speakers_path, 'r', encoding='utf-8') as f:
        speakers = json.load(f)

    with open(episode_topics_path, 'r', encoding='utf-8') as f:
        episode_topics = json.load(f)

    with open(episode_entities_path, 'r', encoding='utf-8') as f:
        episode_entities = json.load(f)

    with open(episode_sentiments_path, 'r', encoding='utf-8') as f:
        episode_sentiments = json.load(f)

    with open(episode_humor_scores_path, 'r', encoding='utf-8') as f:
        episode_humor_scores = json.load(f)

    # Load episode metadata
    with open(episode_metadata_path, 'r', encoding='utf-8') as f:
        episode_metadata = json.load(f)

    return speakers, episode_topics, episode_entities, episode_sentiments, episode_humor_scores, episode_metadata

def integrate_metadata():
    (
        speakers,
        episode_topics,
        episode_entities,
        episode_sentiments,
        episode_humor_scores,
        episode_metadata
    ) = load_metadata()

    # Create a dictionary for episodes using episode_id as the key
    episodes = {}

    # Create a mapping from episode_id
