# recommendations.py

import os
import json
import random
from collections import defaultdict
import Levenshtein  # Ensure this library is installed: pip install python-Levenshtein
import config

# Define the path to the preprocessed data
PREPROCESSED_DATA_PATH = os.path.join(config.BASE_DIR, 'preprocessed_recommendations.json')

# Initialize global variables to store preprocessed data
episodes = {}
inverted_index = {}
speakers = {}

def load_preprocessed_data():
    """
    Load preprocessed recommendation data from a JSON file.
    This includes the episodes dictionary, inverted index, and speakers data for quick lookups.
    
    Returns:
        tuple: A tuple containing the episodes dictionary, inverted index, and speakers dictionary.
    """
    global episodes, inverted_index, speakers
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Preprocessed data file not found at {PREPROCESSED_DATA_PATH}. "
                                "Ensure that you have run the preprocessing script.")
    
    with open(PREPROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    episodes = data.get('episodes', {})
    inverted_index = {k: set(v) for k, v in data.get('inverted_index', {}).items()}
    speakers = data.get('speakers', {})
    print(f"Loaded {len(episodes)} episodes, {len(inverted_index)} keywords, and {len(speakers)} speakers from preprocessed data.")
    
    return episodes, inverted_index, speakers

def similarity(a, b):
    """
    Calculate the similarity ratio between two strings using Levenshtein distance.
    Returns a float between 0 and 1.
    """
    return Levenshtein.ratio(a.lower(), b.lower())

def recommend_by_keyword(episodes_dict, inverted_idx, keyword):
    """
    Recommend episodes based on a keyword search.
    
    Args:
        episodes_dict (dict): Dictionary of episodes with their metadata.
        inverted_idx (dict): Inverted index mapping keywords/synonyms to episode IDs.
        keyword (str): The search keyword input by the user.
    
    Returns:
        list: A list of recommended episodes, each represented as a dictionary with 'title' and 'url'.
    """
    recommended_episodes = []
    relevance_scores = defaultdict(float)

    keyword_lower = keyword.lower()
    
    # Retrieve matching episode IDs from the inverted index
    matched_episode_ids = inverted_idx.get(keyword_lower, set())

    # Iterate only over matched episodes to calculate relevance scores
    for ep_id in matched_episode_ids:
        data = episodes_dict.get(ep_id, {})
        score = 0.0

        # Extract relevant fields
        episode_title = data.get('title', '').lower()
        topics = data.get('topics', [])
        entities = data.get('entities', [])

        # Extract topic descriptions correctly
        # Assuming each topic is a list or tuple where the second element is the description
        topic_strings = []
        for topic in topics:
            if isinstance(topic, (list, tuple)) and len(topic) > 1:
                topic_description = topic[1].lower()
                topic_strings.append(topic_description)
            elif isinstance(topic, str):
                topic_strings.append(topic.lower())
            else:
                # Handle unexpected formats gracefully
                continue

        # Extract entities correctly
        # Assuming each entity is a list or tuple where the first element is the entity name
        entity_strings = []
        for entity in entities:
            if isinstance(entity, (list, tuple)) and len(entity) > 0:
                entity_name = entity[0].lower()
                entity_strings.append(entity_name)
            elif isinstance(entity, str):
                entity_strings.append(entity.lower())
            else:
                # Handle unexpected formats gracefully
                continue

        # Search in episode title
        title_similarity = similarity(keyword_lower, episode_title)
        if title_similarity > 0.8:
            score += 4 * title_similarity  # Higher weight for title match

        # Search in topics
        for topic in topic_strings:
            topic_similarity = similarity(keyword_lower, topic)
            if topic_similarity > 0.7:
                score += 3 * topic_similarity  # Weight for topic match

        # Search in entities
        for entity in entity_strings:
            entity_similarity = similarity(keyword_lower, entity)
            if entity_similarity > 0.7:
                score += 2 * entity_similarity  # Weight for entity match

        if score > 0:
            relevance_scores[ep_id] = score

    # Sort episodes based on relevance score in descending order
    sorted_episodes = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Build the list of recommended episodes with titles and URLs
    for ep in sorted_episodes:
        ep_id = ep[0]
        ep_title = episodes_dict.get(ep_id, {}).get('title', 'Unknown Title')
        ep_url = episodes_dict.get(ep_id, {}).get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})

    return recommended_episodes

def recommend_by_speaker(episodes_dict, speakers_dict, input_speaker_name):
    """
    Recommend episodes based on a speaker's name.
    
    Args:
        episodes_dict (dict): Dictionary of episodes with their metadata.
        speakers_dict (dict): Dictionary of speakers with their associated episodes.
        input_speaker_name (str): The speaker's name input by the user.
    
    Returns:
        list: A list of recommended episodes featuring the best matching speaker.
    """
    recommended_episodes = []
    relevance_scores = defaultdict(float)

    input_name_lower = input_speaker_name.lower()

    for speaker_name, data in speakers_dict.items():
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
    speaker_episodes = speakers_dict[best_speaker].get('episodes', [])

    # Create a list of tuples (episode_id, duration_spoken)
    episode_durations = []
    for ep in speaker_episodes:
        episode_id = ep.get('episode_id')  # Changed from 'episode_name' to 'episode_id'
        duration = ep.get('duration_spoken', 0)
        episode_durations.append((episode_id, duration))

    # Sort the episodes by duration_spoken descending
    sorted_episodes = sorted(episode_durations, key=lambda x: x[1], reverse=True)

    # Extract sorted episode ids and URLs
    for ep in sorted_episodes:
        ep_id = ep[0]
        ep_url = episodes_dict.get(ep_id, {}).get('enclosure_url', '#')
        ep_title = episodes_dict.get(ep_id, {}).get('title', 'Unknown Title')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})

    return recommended_episodes
'''
def recommend_surprise_me(episodes_dict):
    """
    Recommend a random episode.
    
    Args:
        episodes_dict (dict): Dictionary of episodes with their metadata.
    
    Returns:
        list: A list containing a single randomly selected episode.
    """
    if not episodes_dict:
        return []
    ep_id = random.choice(list(episodes_dict.keys()))
    ep_title = episodes_dict[ep_id].get('title', 'Unknown Title')
    ep_url = episodes_dict[ep_id].get('enclosure_url', '#')
    return [{'title': ep_title, 'url': ep_url}]

def recommend_explore_more(episodes_dict, user_keywords):
    """
    Recommend episodes based on a list of user-provided keywords.
    
    Args:
        episodes_dict (dict): Dictionary of episodes with their metadata.
        user_keywords (list): List of keywords input by the user.
    
    Returns:
        list: A list of recommended episodes based on the provided keywords.
    """
    recommended_episodes = []
    relevance_scores = defaultdict(float)

    user_keywords_lower = [kw.lower() for kw in user_keywords]
    all_keywords = set(user_keywords_lower)
    for kw in user_keywords_lower:
        synonyms = get_synonyms(kw)
        all_keywords.update(synonyms)

    for ep_id, data in episodes_dict.items():
        score = 0.0

        # Prepare data
        topics = data.get('topics', [])
        entities = data.get('entities', [])
        episode_title = data.get('title', '').lower()

        # Extract topic descriptions correctly
        topic_strings = []
        for topic in topics:
            if isinstance(topic, (list, tuple)) and len(topic) > 1:
                topic_description = topic[1].lower()
                topic_strings.append(topic_description)
            elif isinstance(topic, str):
                topic_strings.append(topic.lower())
            else:
                # Handle unexpected formats gracefully
                continue

        # Extract entities correctly
        entity_strings = []
        for entity in entities:
            if isinstance(entity, (list, tuple)) and len(entity) > 0:
                entity_name = entity[0].lower()
                entity_strings.append(entity_name)
            elif isinstance(entity, str):
                entity_strings.append(entity.lower())
            else:
                # Handle unexpected formats gracefully
                continue

        # Combine all text for matching
        combined_text = ' '.join(topic_strings + entity_strings + [episode_title])

        for kw in all_keywords:
            # Fuzzy matching
            text_similarity = similarity(kw, combined_text)
            if text_similarity > 0.7:
                score += text_similarity

        if score > 0:
            relevance_scores[ep_id] = score

    # Sort and return top 5 episodes
    sorted_episodes = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    
    for ep in sorted_episodes[:5]:
        ep_id = ep[0]
        ep_title = episodes_dict.get(ep_id, {}).get('title', 'Unknown Title')
        ep_url = episodes_dict.get(ep_id, {}).get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})

    return recommended_episodes

def recommend_understand_more(episodes_dict):
    """
    Recommend episodes with high compound sentiment scores, assuming they are more in-depth.
    
    Args:
        episodes_dict (dict): Dictionary of episodes with their metadata.
    
    Returns:
        list: A list of recommended episodes with high compound sentiment scores.
    """
    # Sort episodes based on the 'compound' sentiment score in descending order
    sorted_episodes = sorted(
        episodes_dict.items(),
        key=lambda x: x[1].get('sentiments', {}).get('compound', 0),
        reverse=True
    )
    top_episodes = sorted_episodes[:5]  # Top 5 episodes
    recommended_episodes = []
    for ep in top_episodes:
        ep_id = ep[0]
        ep_title = ep[1].get('title', 'Unknown Title')
        ep_url = ep[1].get('enclosure_url', '#')
        recommended_episodes.append({'title': ep_title, 'url': ep_url})
    return recommended_episodes

def get_synonyms(keyword):
    """
    Generate a set of synonyms for a given keyword using WordNet.
    
    Args:
        keyword (str): The keyword for which to generate synonyms.
    
    Returns:
        set: A set of synonyms excluding the original keyword.
    """
    from nltk.corpus import wordnet
    synonyms = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != keyword.lower():
                synonyms.add(synonym)
    return synonyms

'''

# Load preprocessed data when the module is imported
try:
    episodes, inverted_index, speakers = load_preprocessed_data()
except Exception as e:
    print(f"Error loading preprocessed data: {e}")
    episodes = {}
    inverted_index = {}
    speakers = {}
