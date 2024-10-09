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

    # Create a dictionary for episodes
    episodes = {}

    # Create a mapping from episode title to enclosure_url
    title_to_url = {}
    for ep in episode_metadata:
        title = ep.get('title')
        url = ep.get('enclosure_url')
        if title and url:
            title_to_url[title] = url

    # Get a list of all episode names
    episode_names = set()
    episode_names.update(episode_topics.keys())
    episode_names.update(episode_entities.keys())
    episode_names.update(episode_sentiments.keys())
    episode_names.update(episode_humor_scores.keys())

    # Initialize episodes dictionary
    for episode_name in episode_names:
        episodes[episode_name] = {
            'topics': episode_topics.get(episode_name, []),
            'entities': episode_entities.get(episode_name, []),
            'sentiments': episode_sentiments.get(episode_name, {}),
            'humor_scores': episode_humor_scores.get(episode_name, {}),
            'speakers': [],
            'enclosure_url': title_to_url.get(episode_name, '#')  # Added
        }

    # Add speakers to episodes
    for speaker_name, data in speakers.items():
        for ep in data.get('episodes', []):
            ep_name = ep.get('episode_name')
            if not ep_name:
                continue  # Skip if 'episode_name' is missing

            if ep_name in episodes:
                episodes[ep_name]['speakers'].append(speaker_name)
            else:
                # If the episode is not already in episodes, add it
                episodes[ep_name] = {
                    'topics': [],
                    'entities': [],
                    'sentiments': {},
                    'humor_scores': {},
                    'speakers': [speaker_name],
                    'enclosure_url': title_to_url.get(ep_name, '#')  # Added
                }

    return episodes, speakers

def get_synonyms(keyword):
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' ').lower())
    return synonyms

def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def recommend_by_keyword(episodes, keyword):
    recommended_episodes = []
    relevance_scores = defaultdict(float)

    keyword_lower = keyword.lower()
    synonyms = get_synonyms(keyword_lower)
    all_keywords = set([keyword_lower]) | synonyms

    for episode_name, data in episodes.items():
        score = 0.0

        # Prepare data
        topic_strings = [t[1] for t in data['topics']]  # Topic descriptions
        entities = [ent[0] for ent in data['entities']]  # Entity texts
        episode_title = episode_name
        enclosure_url = data.get('enclosure_url', '#')  # Get URL

        # Search in episode title
        title_similarity = similarity(keyword_lower, episode_title.lower())
        if title_similarity > 0.8:  # Threshold for fuzzy matching
            score += 4 * title_similarity  # Higher weight for title match

        # Search in topics
        for topic in topic_strings:
            for kw in all_keywords:
                topic_similarity = similarity(kw, topic.lower())
                if topic_similarity > 0.7:
                    score += 3 * topic_similarity  # Weight for topic match

        # Search in entities
        for entity in entities:
            for kw in all_keywords:
                entity_similarity = similarity(kw, entity.lower())
                if entity_similarity > 0.7:
                    score += 2 * entity_similarity  # Weight for entity match

        if score > 0:
            relevance_scores[episode_name] = score

    # Sort episodes based on relevance score
    sorted_episodes = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build list of recommended episodes with titles and URLs
    for ep in sorted_episodes:
        ep_title = ep[0]
        ep_url = episodes[ep_title].get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})

    return recommended_episodes

def recommend_by_speaker(episodes, speakers, input_speaker_name):
    recommended_episodes = []
    relevance_scores = defaultdict(float)

    input_name_lower = input_speaker_name.lower()

    for speaker_name, data in speakers.items():
        speaker_name_lower = speaker_name.lower()
        # Tokenize speaker name and input name
        speaker_tokens = speaker_name_lower.split()
        input_tokens = input_name_lower.split()

        # Initialize maximum similarity score for this speaker
        max_similarity = 0.0

        # Compare each token in input name with tokens in speaker name
        for input_token in input_tokens:
            for speaker_token in speaker_tokens:
                sim = similarity(input_token, speaker_token)
                if sim > max_similarity:
                    max_similarity = sim

        # If similarity exceeds threshold, consider it a match
        if max_similarity > 0.7:
            # Assign a relevance score based on similarity
            relevance_scores[speaker_name] = max_similarity

    # If no speakers matched, return empty list
    if not relevance_scores:
        return []

    # Find the best matching speaker(s)
    sorted_speakers = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    best_speaker = sorted_speakers[0][0]

    # Collect episodes featuring the best matching speaker with durations
    speaker_episodes = speakers[best_speaker].get('episodes', [])

    # Create a list of tuples (episode_name, duration_spoken)
    episode_durations = []
    for ep in speaker_episodes:
        episode_name = ep.get('episode_name')
        duration = ep.get('duration_spoken', 0)
        episode_durations.append((episode_name, duration))

    # Sort the episodes by duration_spoken descending
    sorted_episodes = sorted(episode_durations, key=lambda x: x[1], reverse=True)

    # Extract sorted episode names and URLs
    for ep in sorted_episodes:
        ep_name = ep[0]
        ep_url = episodes.get(ep_name, {}).get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_name, 'url': ep_url})

    return recommended_episodes

def recommend_surprise_me(episodes):
    if not episodes:
        return []
    episode_name = random.choice(list(episodes.keys()))
    episode_url = episodes[episode_name].get('enclosure_url', '#')
    return [{'title': episode_name, 'url': episode_url}]

def recommend_explore_more(episodes, user_keywords):
    recommended_episodes = []
    relevance_scores = defaultdict(float)

    user_keywords_lower = [kw.lower() for kw in user_keywords]
    all_keywords = set(user_keywords_lower)
    for kw in user_keywords_lower:
        synonyms = get_synonyms(kw)
        all_keywords.update(synonyms)

    for episode_name, data in episodes.items():
        score = 0.0

        # Prepare data
        topic_strings = [t[1] for t in data['topics']]
        entities = [ent[0] for ent in data['entities']]
        episode_title = episode_name
        enclosure_url = data.get('enclosure_url', '#')

        # Combine all text for matching
        combined_text = ' '.join(topic_strings + entities + [episode_title]).lower()

        for kw in all_keywords:
            # Fuzzy matching
            text_similarity = similarity(kw, combined_text)
            if text_similarity > 0.7:
                score += text_similarity

        if score > 0:
            relevance_scores[episode_name] = score

    # Sort and return top 5 episodes
    sorted_episodes = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    
    for ep in sorted_episodes[:5]:
        ep_title = ep[0]
        ep_url = episodes[ep_title].get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})

    return recommended_episodes

def recommend_understand_more(episodes):
    # Recommend episodes with high compound sentiment scores (assuming they are more in-depth)
    sorted_episodes = sorted(
        episodes.items(),
        key=lambda x: x[1]['sentiments'].get('compound', 0),
        reverse=True
    )
    top_episodes = sorted_episodes[:5]  # Top 5 episodes
    recommended_episodes = []
    for ep in top_episodes:
        ep_title = ep[0]
        ep_url = ep[1].get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})
    return recommended_episodes
