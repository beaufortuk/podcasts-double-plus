# generate-taxonomies.py

import os
import json
import numpy as np
import re 
import torch
import gensim
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Import configurations
import config

# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK resources
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Initialize spaCy model
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text
def preprocess_text(text):
    # Tokenize and lowercase
    tokens = gensim.utils.simple_preprocess(text, deacc=True)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def load_episode_metadata(metadata_path):
    """
    Load episode metadata and create a mapping from sanitized episode names to episode_ids.
    """
    if not os.path.exists(metadata_path):
        print(f"Episode metadata file '{metadata_path}' does not exist.")
        return {}
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading '{metadata_path}': {e}")
        return {}
    
    # Create a mapping: sanitized_title -> episode_id
    mapping = {}
    for episode in metadata:
        title = episode.get('title', '')
        episode_id = episode.get('episode_id', 'N/A')
        sanitized_title = sanitize_filename(title)
        if episode_id == 'N/A':
            print(f"Warning: Episode '{title}' does not have a valid 'episode_id'.")
        mapping[sanitized_title] = episode_id
    
    return mapping

def sanitize_filename(name):
    """
    Sanitize the filename by removing invalid characters and replacing spaces with underscores.
    """
    # Remove invalid characters
    valid_name = re.sub(r'[\\/:"*?<>|]+', '', name)
    # Replace spaces with underscores
    valid_name = valid_name.replace(' ', '_')
    # Limit length to prevent OS errors
    return valid_name[:255]

def main():
    # Retrieve configurations
    episodes_dir = config.EPISODES_DIR
    base_dir = config.BASE_DIR  # Assuming BASE_DIR is defined in config.py
    if not os.path.exists(episodes_dir):
        print(f"Episodes directory '{episodes_dir}' does not exist.")
        return
    
    # Path to episode_metadata.json
    episode_metadata_path = os.path.join(base_dir, 'episode_metadata.json')
    
    # Load episode metadata mapping
    episode_mapping = load_episode_metadata(episode_metadata_path)
    if not episode_mapping:
        print("No episode metadata available. Exiting.")
        return
    
    # Dictionaries to hold metadata
    speakers = {}
    episode_topics = {}
    episode_entities = {}
    episode_sentiments = {}
    episode_humor_scores = {}
    
    # Iterate over episodes
    for episode_name in os.listdir(episodes_dir):
        episode_dir = os.path.join(episodes_dir, episode_name)
        if not os.path.isdir(episode_dir):
            continue  # Skip if not a directory
        
        print(f"\nProcessing episode folder: {episode_name}")
        
        # Retrieve episode_id from mapping
        episode_id = episode_mapping.get(episode_name, 'N/A')
        if episode_id == 'N/A':
            print(f"Warning: No 'episode_id' found for folder '{episode_name}'. Skipping this episode.")
            continue  # Skip processing if no episode_id
        
        # Initialize variables for this episode
        all_text = ''
        episode_speakers = {}
        
        merged_transcript_path = os.path.join(
            episode_dir, 'merged_speaker_transcript.json'
        )
        if os.path.exists(merged_transcript_path):
            with open(merged_transcript_path, 'r', encoding='utf-8') as f:
                merged_transcript = json.load(f)
        
            # Process each segment in the merged transcript
            for segment in merged_transcript:
                speaker_name = segment['speaker']
                text = segment['text']
                duration = segment['end'] - segment['start']
        
                # Update overall text for the episode
                all_text += ' ' + text
        
                # Update episode speakers
                if speaker_name not in episode_speakers:
                    episode_speakers[speaker_name] = {
                        'total_speaking_time': 0.0,
                        'text_corpus': '',
                        'episodes': []
                    }
        
                episode_speakers[speaker_name]['total_speaking_time'] += duration
                episode_speakers[speaker_name]['text_corpus'] += ' ' + text
        
        # After processing all segments, filter speakers based on speaking time
        for speaker_name, data in episode_speakers.items():
            if data['total_speaking_time'] >= 180:  # 3 minutes = 180 seconds
                # Prepare episode-specific data for the speaker
                episode_data = {
                    'episode_id': episode_id,  # Use episode_id instead of episode_name
                    'text': data['text_corpus'],
                    'duration_spoken': data['total_speaking_time']
                }
        
                # Update global speakers dictionary
                if speaker_name not in speakers:
                    speakers[speaker_name] = {
                        'episodes': [],
                        'total_speaking_time': 0.0,
                        'text_corpus': ''
                    }
        
                speakers[speaker_name]['episodes'].append(episode_data)
                speakers[speaker_name]['total_speaking_time'] += data['total_speaking_time']
                speakers[speaker_name]['text_corpus'] += ' ' + data['text_corpus']
        
        # Topic Module
        tokens = preprocess_text(all_text)
        if tokens:
            dictionary = corpora.Dictionary([tokens])
            corpus = [dictionary.doc2bow(tokens)]
            num_topics = 6  # Adjust as needed
        
            # Build LDA model
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                 random_state=42, passes=10)
        
            # Extract topics
            topics = lda_model.print_topics(num_words=5)
            episode_topics[episode_id] = topics  # Use episode_id as key
        else:
            episode_topics[episode_id] = []
        
        # Entity Extraction Module
        doc = nlp(all_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        episode_entities[episode_id] = entities  # Use episode_id as key
        
        # Tone Module
        # Sentiment Analysis
        sentiment_scores = sia.polarity_scores(all_text)
        episode_sentiments[episode_id] = sentiment_scores  # Use episode_id as key
        
        # Humor Detection (simple keyword-based approach)
        humor_keywords = ['joke', 'funny', 'humor', 'laugh', 'hilarious', 'comedy']
        humor_count = sum(all_text.lower().count(word) for word in humor_keywords)
        total_words = len(all_text.split())
        humor_density = humor_count / total_words if total_words > 0 else 0
        episode_humor_scores[episode_id] = {
            'humor_count': humor_count,
            'total_words': total_words,
            'humor_density': humor_density
        }
    
    # Integration and Saving Results
    
    # Process and save speaker profiles
    # Save speaker profiles
    speakers_output_path = os.path.join(base_dir, 'speakers.json')
    with open(speakers_output_path, 'w', encoding='utf-8') as f:
        json.dump(speakers, f, ensure_ascii=False, indent=4)
    print(f"Speaker profiles saved to {speakers_output_path}")
    
    # Save episode topics
    episode_topics_output_path = os.path.join(base_dir, 'episode_topics.json')
    with open(episode_topics_output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_topics, f, ensure_ascii=False, indent=4)
    print(f"Episode topics saved to {episode_topics_output_path}")
    
    # Save episode entities
    episode_entities_output_path = os.path.join(base_dir, 'episode_entities.json')
    with open(episode_entities_output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_entities, f, ensure_ascii=False, indent=4)
    print(f"Episode entities saved to {episode_entities_output_path}")
    
    # Save episode sentiments
    episode_sentiments_output_path = os.path.join(base_dir, 'episode_sentiments.json')
    with open(episode_sentiments_output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_sentiments, f, ensure_ascii=False, indent=4)
    print(f"Episode sentiments saved to {episode_sentiments_output_path}")
    
    # Save episode humor scores
    episode_humor_scores_output_path = os.path.join(base_dir, 'episode_humor_scores.json')
    with open(episode_humor_scores_output_path, 'w', encoding='utf-8') as f:
        json.dump(episode_humor_scores, f, ensure_ascii=False, indent=4)
    print(f"Episode humor scores saved to {episode_humor_scores_output_path}")

if __name__ == '__main__':
    main()
