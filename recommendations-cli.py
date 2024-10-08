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

    return speakers, episode_topics, episode_entities, episode_sentiments, episode_humor_scores

def integrate_metadata():
    speakers, episode_topics, episode_entities, episode_sentiments, episode_humor_scores = load_metadata()

    # Create a dictionary for episodes
    episodes = {}

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
            'speakers': []
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
                    'speakers': [speaker_name]
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
    recommended_episodes = [ep[0] for ep in sorted_episodes]

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

    # Extract sorted episode names
    recommended_episodes = [ep[0] for ep in sorted_episodes]

    return recommended_episodes


def recommend_surprise_me(episodes):
    if not episodes:
        return []
    episode_name = random.choice(list(episodes.keys()))
    return [episode_name]

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
    recommended_episodes = [ep[0] for ep in sorted_episodes[:5]]

    return recommended_episodes

def recommend_understand_more(episodes):
    # Recommend episodes with high compound sentiment scores (assuming they are more in-depth)
    sorted_episodes = sorted(
        episodes.items(),
        key=lambda x: x[1]['sentiments'].get('compound', 0),
        reverse=True
    )
    top_episodes = [ep[0] for ep in sorted_episodes[:5]]  # Top 5 episodes
    return top_episodes

def main():
    episodes, speakers = integrate_metadata()

    # Simple command-line interface
    while True:
        print("\nWelcome to The Economist Podcast Recommendation System!")
        print("Please select an option:")
        print("1. Search for Episodes")
        print("2. Recommend by Speaker")
        print("3. Surprise Me")
        print("4. Explore More")
        print("5. Understand More")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            keyword = input("Enter a keyword (e.g., AI, Elon Musk, Climate Change): ").strip()
            recommended_episodes = recommend_by_keyword(episodes, keyword)
            if recommended_episodes:
                print(f"\nEpisodes related to '{keyword}' (sorted by relevance):")
                for ep in recommended_episodes:
                    print(f"- {ep}")
            else:
                print(f"\nNo episodes found related to '{keyword}'.")
        elif choice == '2':
            speaker_name = input("Enter the speaker's name (e.g., Zanny): ").strip()
            recommended_episodes = recommend_by_speaker(episodes, speakers, speaker_name)
            if recommended_episodes:
                print(f"\nEpisodes featuring '{speaker_name}' (sorted by airtime):")
                for ep in recommended_episodes:
                    print(f"- {ep}")
            else:
                print(f"\nNo episodes found featuring '{speaker_name}'.")
        elif choice == '3':
            recommended_episodes = recommend_surprise_me(episodes)
            if recommended_episodes:
                print("\nSurprise! Here's an episode you might enjoy:")
                for ep in recommended_episodes:
                    print(f"- {ep}")
            else:
                print("\nNo episodes available to recommend.")
        elif choice == '4':
            user_keywords = input("Enter keywords you're interested in (comma-separated): ").split(',')
            user_keywords = [t.strip() for t in user_keywords if t.strip()]
            if user_keywords:
                recommended_episodes = recommend_explore_more(episodes, user_keywords)
                if recommended_episodes:
                    print("\nExplore more with these episodes (sorted by relevance):")
                    for ep in recommended_episodes:
                        print(f"- {ep}")
                else:
                    print("\nNo episodes found to explore more.")
            else:
                print("\nNo keywords entered.")
        elif choice == '5':
            recommended_episodes = recommend_understand_more(episodes)
            if recommended_episodes:
                print("\nDeepen your understanding with these episodes:")
                for ep in recommended_episodes:
                    print(f"- {ep}")
            else:
                print("\nNo episodes available for deeper understanding.")
        elif choice == '6':
            print("Thank you for using The Economist Podcast Recommendation System. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please select a valid option.")

if __name__ == '__main__':
    main()
