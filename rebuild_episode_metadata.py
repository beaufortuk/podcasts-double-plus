import feedparser
import json
import os
import re
import requests
from requests.exceptions import HTTPError, RequestException

# Import configurations
import config

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

def parse_rss_feed(rss_feed_url, num_episodes):
    """
    Parse the RSS feed and extract metadata for the specified number of episodes.
    Returns a list of dictionaries containing episode metadata.
    """
    try:
        # Fetch the RSS feed
        print(f"Fetching RSS feed from: {rss_feed_url}")
        response = requests.get(rss_feed_url, timeout=10)
        response.raise_for_status()
        
        # Parse the RSS feed content
        feed = feedparser.parse(response.content)
        
        # Check for parsing issues
        if feed.bozo:
            raise ValueError(f"Malformed feed: {feed.bozo_exception}")
        
        # Validate that the feed contains entries
        if not hasattr(feed, 'entries') or not feed.entries:
            raise ValueError("Feed does not contain any entries")
        
        # Extract the specified number of episodes
        episodes = feed.entries[:num_episodes]
        print(f"Successfully fetched and parsed {len(episodes)} episodes.")
        
    except HTTPError as http_err:
        print(f"HTTP error occurred while fetching the RSS feed: {http_err}")
        return []
    except RequestException as req_err:
        print(f"Network-related error occurred while fetching the RSS feed: {req_err}")
        return []
    except ValueError as val_err:
        print(f"Data parsing error: {val_err}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []
    
    # Extract metadata for each episode
    episode_metadata = []
    for entry in episodes:
        title = entry.get('title', 'N/A')
        link = entry.get('link', 'N/A')
        
        # Extract enclosure URL
        enclosure_url = None
        if 'enclosures' in entry and entry.enclosures:
            enclosure_url = entry.enclosures[0].get('href')
        else:
            print(f"No enclosure found for episode '{title}'. Skipping this episode.")
            continue  # Skip episodes without enclosure
        
        # Extract acast:episodeId
        episode_id = entry.get('acast_episodeid', 'N/A')  # feedparser converts <acast:episodeId> to 'acast_episodeid'
        if episode_id == 'N/A':
            print(f"No 'acast:episodeId' found for episode '{title}'. Setting as 'N/A'.")
        
        # Compile episode data
        episode_data = {
            'title': title,
            'link': link,
            'enclosure_url': enclosure_url,
            'episode_id': episode_id
        }
        episode_metadata.append(episode_data)
    
    return episode_metadata

def save_metadata_to_json(metadata, output_file='episode_metadata.json'):
    """
    Save the list of episode metadata dictionaries to a JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        print(f"Episode metadata successfully saved to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred while saving metadata to '{output_file}': {e}")

def main():
    # Retrieve configurations
    rss_feed_url = config.PODCAST_RSS_FEED
    num_episodes = config.NUMBER_OF_EPS_TO_ANALYSE
    
    if not rss_feed_url:
        print("RSS feed URL is not configured. Please set 'PODCAST_RSS_FEED' in config.py.")
        return
    
    # Parse the RSS feed and extract metadata
    episode_metadata = parse_rss_feed(rss_feed_url, num_episodes)
    
    if not episode_metadata:
        print("No episode metadata extracted. Exiting.")
        return
    
    # Save the metadata to JSON
    save_metadata_to_json(episode_metadata, 'episode_metadata.json')

if __name__ == "__main__":
    main()
