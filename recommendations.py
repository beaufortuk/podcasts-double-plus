# recommendations.py

import os
import json
import random

# Import configurations
import config

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
        for ep_name in data['episodes']:
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

def recommend_by_speaker(episodes, speaker_name):
    recommended_episodes = []
    for episode_name, data in episodes.items():
        if speaker_name in data['speakers']:
            recommended_episodes.append(episode_name)
    return recommended_episodes

def recommend_by_topic(episodes, topic_keyword):
    recommended_episodes = []
    for episode_name, data in episodes.items():
        topic_strings = [t[1] for t in data['topics']]  # Extract topic descriptions
        if any(topic_keyword.lower() in topic.lower() for topic in topic_strings):
            recommended_episodes.append(episode_name)
    return recommended_episodes

def recommend_surprise_me(episodes):
    episode_name = random.choice(list(episodes.keys()))
    return [episode_name]

def recommend_explore_more(episodes, user_topics):
    recommended_episodes = []
    for episode_name, data in episodes.items():
        episode_topics = [t[1] for t in data['topics']]
        if any(user_topic.lower() in topic.lower() for user_topic in user_topics for topic in episode_topics):
            recommended_episodes.append(episode_name)
    # Shuffle and return a few episodes
    random.shuffle(recommended_episodes)
    return recommended_episodes[:5]  # Return top 5

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
        print("1. Recommend by Speaker")
        print("2. Recommend by Topic")
        print("3. Surprise Me")
        print("4. Explore More")
        print("5. Understand More")
        print("6. Exit")
        choice = input("Enter your choice (1-6): ").strip()

        if choice == '1':
            speaker_name = input("Enter the speaker's name: ").strip()
            recommended_episodes = recommend_by_speaker(episodes, speaker_name)
            if recommended_episodes:
                print(f"\nEpisodes featuring {speaker_name}:")
                for ep in recommended_episodes:
                    print(f"- {ep}")
            else:
                print(f"\nNo episodes found featuring '{speaker_name}'.")
        elif choice == '2':
            topic_keyword = input("Enter a topic keyword: ").strip()
            recommended_episodes = recommend_by_topic(episodes, topic_keyword)
            if recommended_episodes:
                print(f"\nEpisodes discussing '{topic_keyword}':")
                for ep in recommended_episodes:
                    print(f"- {ep}")
            else:
                print(f"\nNo episodes found discussing '{topic_keyword}'.")
        elif choice == '3':
            recommended_episodes = recommend_surprise_me(episodes)
            print("\nSurprise! Here's an episode you might enjoy:")
            for ep in recommended_episodes:
                print(f"- {ep}")
        elif choice == '4':
            user_topics = input("Enter topics you're interested in (comma-separated): ").split(',')
            user_topics = [t.strip() for t in user_topics if t.strip()]
            if user_topics:
                recommended_episodes = recommend_explore_more(episodes, user_topics)
                if recommended_episodes:
                    print("\nExplore more with these episodes:")
                    for ep in recommended_episodes:
                        print(f"- {ep}")
                else:
                    print("\nNo episodes found to explore more.")
            else:
                print("\nNo topics entered.")
        elif choice == '5':
            recommended_episodes = recommend_understand_more(episodes)
            print("\nDeepen your understanding with these episodes:")
            for ep in recommended_episodes:
                print(f"- {ep}")
        elif choice == '6':
            print("Thank you for using The Economist Podcast Recommendation System. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please select a valid option.")

if __name__ == '__main__':
    main()
