# preprocess_recommendations.py

import os
import json
from collections import defaultdict
from nltk.corpus import wordnet
import nltk
import config

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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

def get_synonyms(keyword):
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != keyword.lower():
                synonyms.add(synonym)
    return synonyms

def preprocess_recommendations():
    speakers, episode_topics, episode_entities, episode_sentiments, episode_humor_scores, episode_metadata = load_metadata()

    # Create a dictionary for episodes using episode_id as the key
    episodes = {}

    # Create a mapping from episode_id to enclosure_url and title
    id_to_url_title = {}
    for ep in episode_metadata:
        ep_id = ep.get('episode_id')
        title = ep.get('title')
        url = ep.get('enclosure_url')
        if ep_id and title and url:
            id_to_url_title[ep_id] = {'title': title, 'enclosure_url': url}

    # Get a list of all episode_ids
    episode_ids = set()
    episode_ids.update(episode_topics.keys())
    episode_ids.update(episode_entities.keys())
    episode_ids.update(episode_sentiments.keys())
    episode_ids.update(episode_humor_scores.keys())

    # Initialize episodes dictionary using episode_id
    for ep_id in episode_ids:
        ep_info = id_to_url_title.get(ep_id, {})
        episodes[ep_id] = {
            'topics': episode_topics.get(ep_id, []),
            'entities': episode_entities.get(ep_id, []),
            'sentiments': episode_sentiments.get(ep_id, {}),
            'humor_scores': episode_humor_scores.get(ep_id, {}),
            'speakers': [],
            'title': ep_info.get('title', 'Unknown Title'),
            'enclosure_url': ep_info.get('enclosure_url', '#')
        }

    # Add speakers to episodes
    for speaker_name, data in speakers.items():
        for ep in data.get('episodes', []):
            ep_id = ep.get('episode_id')  # Changed from 'episode_name' to 'episode_id'
            if not ep_id:
                continue  # Skip if 'episode_id' is missing

            if ep_id in episodes:
                episodes[ep_id]['speakers'].append(speaker_name)
            else:
                # If the episode_id is not already in episodes, add it
                ep_info = id_to_url_title.get(ep_id, {})
                episodes[ep_id] = {
                    'topics': [],
                    'entities': [],
                    'sentiments': {},
                    'humor_scores': {},
                    'speakers': [speaker_name],
                    'title': ep_info.get('title', 'Unknown Title'),
                    'enclosure_url': ep_info.get('enclosure_url', '#')
                }

    # Build inverted index: keyword/synonym -> set of episode_ids
    inverted_index = defaultdict(set)

    for ep_id, data in episodes.items():
        # Extract keywords from title, topics, and entities
        keywords = set()

        # From title
        title_tokens = data['title'].lower().split()
        keywords.update(title_tokens)

        # From topics
        for topic in data['topics']:
            if isinstance(topic, (list, tuple)) and len(topic) > 1:
                keywords.add(topic[1].lower())  # Assuming topic structure
            elif isinstance(topic, str):
                keywords.add(topic.lower())
            else:
                continue  # Skip unexpected formats

        # From entities
        for entity in data['entities']:
            if isinstance(entity, (list, tuple)) and len(entity) > 0:
                keywords.add(entity[0].lower())  # Assuming entity structure
            elif isinstance(entity, str):
                keywords.add(entity.lower())
            else:
                continue  # Skip unexpected formats

        # Generate synonyms for each keyword and add to the inverted index
        for kw in keywords:
            inverted_index[kw].add(ep_id)
            synonyms = get_synonyms(kw)
            for syn in synonyms:
                inverted_index[syn].add(ep_id)

    # Save the preprocessed data
    preprocessed_data = {
        'episodes': episodes,
        'inverted_index': {k: list(v) for k, v in inverted_index.items()},
        'speakers': speakers  # Include speakers in the preprocessed data
    }

    output_path = os.path.join(config.BASE_DIR, 'preprocessed_recommendations.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_data, f, indent=4)

    print(f"Preprocessing complete. Data saved to {output_path}")

if __name__ == '__main__':
    preprocess_recommendations()
