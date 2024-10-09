import feedparser
import whisper
from pyannote.audio import Pipeline, Inference
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchaudio
import os
import json
import nltk
import re
import requests
from nltk.stem import WordNetLemmatizer
import numpy as np
from pydub import AudioSegment
from pyannote.core import Annotation
from pyannote.database.util import load_rttm
from requests.exceptions import HTTPError, RequestException
import gc
import io
import itertools

from typing import BinaryIO, Union

import av

# Import configurations
import config

# Determine if device is CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CuDNN Benchmarking for potential speed-up
torch.backends.cudnn.benchmark = True

# Load Whisper model once globally
print("Loading Whisper model...")
whisper_model = whisper.load_model("base").to(device)

# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize lemmatizer and embedding model
lemmatizer = WordNetLemmatizer()

# Get Hugging Face token
hf_token = config.HUGGING_FACE_TOKEN
if hf_token is None:
    print("Hugging Face token not found. Please set the 'HUGGING_FACE_TOKEN' environment variable.")
else:
    print("Hugging Face token found.")

# Initialize the embedding model once
print("Initializing embedding model...")
embedding_model = Inference(
    "pyannote/embedding",
    window="whole",
    device=device,  # specify device
    use_auth_token=hf_token
)

# Decode audio functions
def decode_audio(input_file: Union[str, BinaryIO], sampling_rate: int = 16000):
    """Decodes the audio, returns a float32 Numpy array."""
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    raw_buffer = io.BytesIO()
    dtype = None

    with av.open(input_file, metadata_errors="ignore") as container:
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)
        frames = _group_frames(frames, 500000)
        frames = _resample_frames(frames, resampler)

        for frame in frames:
            array = frame.to_ndarray()
            dtype = array.dtype
            raw_buffer.write(array)

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype).astype(np.float32) / 32768.0
    return audio


def _ignore_invalid_frames(frames):
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(frames, num_samples=None):
    fifo = av.audio.fifo.AudioFifo()
    for frame in frames:
        frame.pts = None  # Ignore timestamp check.
        fifo.write(frame)
        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler):
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


# Optimized get_speaker_embedding function
def get_speaker_embedding(audio_data_np):
    """Compute speaker embedding from numpy audio data."""
    try:
        with torch.no_grad():
            waveform = torch.from_numpy(audio_data_np).float().to(device)
            waveform = waveform.unsqueeze(0)  # Add batch dimension
            audio_data = {'waveform': waveform, 'sample_rate': 16000}
            embedding = embedding_model(audio_data)

            if embedding is None:
                raise ValueError("Received None from embedding_model.")

            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            elif isinstance(embedding, (list, tuple)):
                embedding = np.array(embedding)
            elif isinstance(embedding, (float, int)):
                embedding = np.array([embedding])

            print(f"Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")
            return embedding.squeeze()

    except Exception as e:
        print(f"Error computing embedding: {e}")
        return None


# Function to compute and save known speaker embeddings
def compute_and_save_known_speaker_embeddings(known_speaker_files, embeddings_dir='embeddings'):
    os.makedirs(embeddings_dir, exist_ok=True)
    known_speakers_embeddings = {}

    for name, audio_file in known_speaker_files.items():
        embedding_file = os.path.join(embeddings_dir, f"{name}_embedding.npy")

        if os.path.exists(embedding_file):
            print(f"Loading existing embedding for {name} from disk.")
            embedding = np.load(embedding_file)
        else:
            print(f"Computing embedding for {name}...")
            audio = decode_audio(audio_file, 16000)
            embedding = get_speaker_embedding(audio)
            if embedding is not None:
                np.save(embedding_file, embedding)
                print(f"Saved embedding for {name} to {embedding_file}")
            else:
                print(f"Failed to compute embedding for {name}. Skipping.")
                continue

        known_speakers_embeddings[name] = embedding

    return known_speakers_embeddings


# Function to identify speaker
def identify_speaker(embedding, known_speakers):
    if np.isnan(embedding).any():
        print("Embedding contains NaN values. Cannot identify speaker.")
        return 'Unknown', 0.0
    max_similarity = -1
    identified_speaker = None
    for name, known_embedding in known_speakers.items():
        if np.isnan(known_embedding).any():
            print(f"Known embedding for speaker {name} contains NaN values. Skipping.")
            continue
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            identified_speaker = name
    return identified_speaker, max_similarity


# Function to extract audio segments in-memory
def extract_segment(audio_np, segment):
    """Extracts a segment from numpy audio data."""
    start_sample = int(segment.start * 16000)
    end_sample = int(segment.end * 16000)
    return audio_np[start_sample:end_sample]


# Function to relabel diarization
def relabel_diarization(diarization, speaker_names):
    new_diarization = Annotation()
    for segment, _, speaker_label in diarization.itertracks(yield_label=True):
        speaker_name = speaker_names.get(speaker_label, 'Unknown')
        # Replace spaces with underscores to make it RTTM-compatible
        rttm_speaker_name = speaker_name.replace(' ', '_')
        new_diarization[segment] = rttm_speaker_name
    return new_diarization


# Align Transcription Segments with Diarization Segments
def align_diarization_with_transcript(diarization, transcription_result):
    speaker_transcript = []

    # Iterate over each transcription segment
    for t_segment in transcription_result['segments']:
        t_start = t_segment['start']
        t_end = t_segment['end']
        t_text = t_segment['text']

        # Find the speaker for this transcription segment
        max_overlap = 0
        speaker = 'Unknown'

        # Iterate over diarization segments
        for d_segment, _, d_speaker in diarization.itertracks(yield_label=True):
            d_start = d_segment.start
            d_end = d_segment.end

            # Compute overlap between transcription segment and diarization segment
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)

            # Update speaker if this diarization segment has more overlap
            if overlap > max_overlap:
                max_overlap = overlap
                speaker = d_speaker

        # Append the speaker-attributed segment
        speaker_transcript.append({
            'start': t_start,
            'end': t_end,
            'speaker': speaker.replace('_', ' '),  # Replace underscores with spaces for readability
            'text': t_text.strip()
        })

    return speaker_transcript


# Generate a Speaker-Attributed Transcript
def format_speaker_transcript(speaker_transcript):
    formatted_transcript = ""
    for segment in speaker_transcript:
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        text = segment['text']

        # Format the timestamp (e.g., HH:MM:SS)
        formatted_start = format_timestamp(start_time)
        formatted_end = format_timestamp(end_time)

        # Append to the formatted transcript
        formatted_transcript += f"[{formatted_start} - {formatted_end}] {speaker}: {text}\n"

    return formatted_transcript


def format_timestamp(seconds):
    # Convert seconds to HH:MM:SS format
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


# Save the Formatted Transcript as JSON
def save_speaker_transcript_json(speaker_transcript, merged_speaker_transcript, output_dir):
    # Paths for saving
    original_output_path = os.path.join(output_dir, 'speaker_transcript.json')
    merged_output_path = os.path.join(output_dir, 'merged_speaker_transcript.json')

    # Save original speaker transcript
    if os.path.exists(original_output_path):
        print(f"Speaker-attributed transcript already exists at {original_output_path}. Skipping save.")
    else:
        with open(original_output_path, "w", encoding="utf-8") as f:
            json.dump(speaker_transcript, f, ensure_ascii=False, indent=4)
        print(f"Speaker-attributed transcript saved to {original_output_path}")

    # Save merged speaker transcript
    if os.path.exists(merged_output_path):
        print(f"Merged speaker-attributed transcript already exists at {merged_output_path}. Skipping save.")
    else:
        with open(merged_output_path, "w", encoding="utf-8") as f:
            json.dump(merged_speaker_transcript, f, ensure_ascii=False, indent=4)
        print(f"Merged speaker-attributed transcript saved to {merged_output_path}")


# Tokenize the merged text into sentences
def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


# Function to sanitize filenames
def sanitize_filename(name):
    # Remove invalid characters and limit length
    valid_name = re.sub(r'[\\/:"*?<>|]+', '', name)
    valid_name = valid_name.replace(' ', '_')  # Replace spaces with underscores
    valid_name = valid_name[:255]  # Limit length to prevent OS errors
    return valid_name


# Function to download MP3 files
def download_mp3(enclosure_url, save_path):
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping download.")
        return True
    try:
        with requests.get(enclosure_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {save_path}")
        return True
    except HTTPError as http_err:
        print(f"HTTP error occurred while downloading {enclosure_url}: {http_err}")
    except RequestException as req_err:
        print(f"Network error occurred while downloading {enclosure_url}: {req_err}")
    except Exception as e:
        print(f"An error occurred while downloading {enclosure_url}: {e}")
    return False


# Merging of Speaker Segments
def merge_speaker_segments(speaker_transcript):
    merged_transcript = []
    if not speaker_transcript:
        return merged_transcript

    # Initialize with the first segment
    current_segment = speaker_transcript[0].copy()

    for segment in speaker_transcript[1:]:
        if segment['speaker'] == current_segment['speaker']:
            # Concatenate text and update end time
            current_segment['text'] += ' ' + segment['text']
            current_segment['end'] = segment['end']
        else:
            # Perform sentence splitting on the current segment's text
            current_segment['sentences'] = split_into_sentences(current_segment['text'])
            # Append the completed segment and start a new one
            merged_transcript.append(current_segment)
            current_segment = segment.copy()

    # Handle the last segment
    current_segment['sentences'] = split_into_sentences(current_segment['text'])
    merged_transcript.append(current_segment)
    return merged_transcript


# Step 1: Transcribe the audio file with Whisper (with check for existing transcription)
def transcribe_audio(audio_file, transcript_path):
    if os.path.exists(transcript_path):
        print(f"Transcript already exists for {audio_file}. Loading from {transcript_path}...")
        with open(transcript_path, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        # Use the globally loaded Whisper model
        print(f"Transcribing {audio_file}...")
        with torch.no_grad():
            result = whisper_model.transcribe(audio_file, language='en', word_timestamps=False)

        # Save the transcript data to a file for future use
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Transcript data saved to {transcript_path}")

    return result


# Step 2: Optimized Speaker Diarization using pyannote.audio
def identify_speakers(audio_file, output_dir, known_speaker_files):
    rttm_filename = 'audio_with_names.rttm'
    rttm_path = os.path.join(output_dir, rttm_filename)

    if os.path.exists(rttm_path):
        print(f"Diarization already exists for {audio_file}. Loading from {rttm_path}...")
        # Load the diarization from the RTTM file
        diarization_dict = load_rttm(rttm_path)
        diarization = next(iter(diarization_dict.values()))
        return diarization

    # Load diarization pipeline
    print("Loading diarization pipeline...")
    with torch.no_grad():
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token).to(device)

        # Format audio file for diarization
        audio = decode_audio(audio_file, 16000)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': 16000
        }

        # Perform diarization
        print(f"Performing diarization on {audio_data}...")
        diarization = pipeline(audio_data)

    # Compute known speaker embeddings
    print("Computing known speaker embeddings...")
    known_speakers = compute_and_save_known_speaker_embeddings(known_speaker_files)

    # Decode audio once
    print("Decoding audio for embedding extraction...")
    audio_np = decode_audio(audio_file, 16000)
    total_duration = len(audio_np) / 16000  # in seconds

    # Extract embeddings for diarized segments
    speaker_embeddings = {}
    processed_duration = 0  # To track processed duration

    for segment, _, speaker_label in diarization.itertracks(yield_label=True):
        # Extract segment audio in-memory
        segment_audio = extract_segment(audio_np, segment)
        if len(segment_audio) == 0:
            print(f"Empty segment detected: {segment}. Skipping.")
            continue

        # Compute embedding
        embedding = get_speaker_embedding(segment_audio)

        if embedding is not None:
            speaker_embeddings.setdefault(speaker_label, []).append(embedding)
        else:
            print(f"Could not compute embedding for segment {segment} and speaker {speaker_label}")

        # Update processed duration
        processed_duration = segment.end

        # Calculate percentage
        percentage = (processed_duration / total_duration) * 100

        # Format timecodes
        current_timecode = format_timestamp(processed_duration)
        total_timecode = format_timestamp(total_duration)

        # Print progress update
        print(f"Progress: {current_timecode}/{total_timecode} ({percentage:.2f}%) processed.")

    # Identify speakers
    print("Identifying speakers...")
    speaker_names = {}
    for speaker_label, embeddings in speaker_embeddings.items():
        print(f"Processing embeddings for speaker label: {speaker_label}")
        embeddings_array = np.stack(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)
        if np.isnan(avg_embedding).any():
            print(f"Average embedding for speaker {speaker_label} contains NaN values. Skipping.")
            speaker_names[speaker_label] = 'Unknown'
            continue
        name, similarity = identify_speaker(avg_embedding, known_speakers)
        print(f"Speaker {speaker_label} identified as {name} with similarity {similarity:.4f}")
        speaker_names[speaker_label] = name if similarity > 0.7 else 'Unknown'

    # Relabel diarization
    print("Relabeling diarization with identified speaker names...")
    new_diarization = relabel_diarization(diarization, speaker_names)

    # Save the new diarization
    print(f"Saving diarization to {rttm_path}...")
    with open(rttm_path, "w") as rttm:
        new_diarization.write_rttm(rttm)
    print(f"Diarization saved to {rttm_path}")

    return new_diarization


# Main execution
if __name__ == "__main__":
    # Define known speaker files once
    known_speaker_files = {
        'John Prideaux': os.path.join('assets', 'samples', "john-prideaux-sample.mp3"),
        'Charlotte Howard': os.path.join('assets', 'samples', "charlotte-howard-sample.mp3"),
        'Idrees Kahloon': os.path.join('assets', 'samples', "idrees-kahloon-sample.mp3"),
        'Alok Jha': os.path.join('assets', 'samples', "alok-jha-sample.mp3"),
        'Shailesh Chitnis': os.path.join('assets', 'samples', "shailesh-chitnis-sample.mp3"), 
        'Dave Patterson': os.path.join('assets', 'samples', "dave-patterson-sample.mp3"),
        'Alice Fullwood': os.path.join('assets', 'samples', "alice-fullwood-sample.mp3"),
        'Cerian Richmond-Jones': os.path.join('assets', 'samples', "cerian-richmond-jones-sample.mp3"),
        'Mike Bird': os.path.join('assets', 'samples', "mike-bird-sample.mp3"),      
        'Tom Lee-Devlin': os.path.join('assets', 'samples', "tom-lee-devlin-sample.mp3"),
        'David Rennie': os.path.join('assets', 'samples', "david-rennie-sample.mp3"),
        'Anton La-Guardia': os.path.join('assets', 'samples', "anton-la-guardia-sample.mp3"),
        'Hollie Berman': os.path.join('assets', 'samples', "hollie-berman-sample.mp3"),
        'Henry Tricks': os.path.join('assets', 'samples', "henry-tricks-sample.mp3"),
        'Rosie Blau': os.path.join('assets', 'samples', "rosie-blau-sample.mp3"),
        'Annie Crabill': os.path.join('assets', 'samples', "annie-crabill-sample.mp3"),
        'Jan Piotrowski': os.path.join('assets', 'samples', "jan-piotrowski-sample.mp3"),
        'Jason Palmer': os.path.join('assets', 'samples', "jason-palmer-sample.mp3"),
        'Adam Roberts': os.path.join('assets', 'samples', "adam-roberts-sample.mp3"),
        'Vendeline von Bredow': os.path.join('assets', 'samples', "vendeline-von-bredow-sample.mp3"),
        'Natasha Loder': os.path.join('assets', 'samples', "natasha-loder-sample.mp3"),
        'Zanny Minton Beddoes': os.path.join('assets', 'samples', "zanny-sample.mp3")           
        # Add more as needed
    }

    # Parse the RSS feed
    print(f"\nParsing the RSS feed")
    rss_feed_url = config.PODCAST_RSS_FEED
    try:
        # Attempt to fetch the RSS feed using requests
        response = requests.get(rss_feed_url, timeout=10)  # Added timeout for better control
        response.raise_for_status()  # Raises HTTPError for bad HTTP status codes (4xx, 5xx)

        # Parse the RSS feed content
        feed = feedparser.parse(response.content)

        # Check if feedparser encountered any parsing issues
        if feed.bozo:
            raise ValueError(f"Malformed feed: {feed.bozo_exception}")

        # Validate that the feed contains entries
        if not hasattr(feed, 'entries') or not feed.entries:
            raise ValueError("Feed does not contain any entries")

        # Extract the first N episodes
        episodes = feed.entries[:config.NUMBER_OF_EPS_TO_ANALYSE]
        print("\nFinished parsing the RSS feed successfully.")

    except HTTPError as http_err:
        print(f"HTTP error occurred while fetching the RSS feed: {http_err}")
        episodes = []
    except RequestException as req_err:
        print(f"Network-related error occurred while fetching the RSS feed: {req_err}")
        episodes = []
    except ValueError as val_err:
        print(f"Data parsing error: {val_err}")
        episodes = []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        episodes = []

    if not episodes:
        print("No episodes to process. Exiting.")
        exit(1)

    # Record episode metadata
    print(f"\nRecording episode metadata")
    episode_metadata = []
    for entry in episodes:
        title = entry.title
        link = entry.link
        enclosure_url = entry.enclosures[0].href if entry.enclosures else None

        # Extract acast:episodeId
        episode_id = entry.get('acast_episodeid')  # feedparser converts <acast:episodeId> to 'acast_episodeid'

        if enclosure_url:
            episode_data = {
                'title': title,
                'link': link,
                'enclosure_url': enclosure_url,
                'episode_id': episode_id  # Add the episode_id here
            }
            episode_metadata.append(episode_data)
        else:
            print(f"No enclosure found for episode '{title}'. Skipping.")
    print(f"\nFinished recording episode metadata")

    if not episode_metadata:
        print("No valid episodes to process. Exiting.")
        exit(1)

    # Process each episode
    for episode in episode_metadata:
        sanitized_title = sanitize_filename(episode['title'])
        episode_dir = os.path.join('episodes', sanitized_title)
        os.makedirs(episode_dir, exist_ok=True)

        # Path for original MP3
        original_mp3_path = os.path.join(episode_dir, 'original.mp3')

        # Download MP3 file if it doesn't exist
        mp3_url = episode['enclosure_url']
        if download_mp3(mp3_url, original_mp3_path):
            episode['local_path'] = original_mp3_path
        else:
            episode['local_path'] = None
            continue  # Skip processing if download failed

        if original_mp3_path and os.path.exists(original_mp3_path):
            print(f"\nProcessing '{episode['title']}'")

            # Paths for transcripts and diarization
            transcript_filename = 'transcript.json'
            transcript_path = os.path.join(episode_dir, transcript_filename)

            # Check if the merged speaker-attributed transcript already exists
            merged_speaker_transcript_filename = 'merged_speaker_transcript.json'
            merged_speaker_transcript_path = os.path.join(episode_dir, merged_speaker_transcript_filename)
            if os.path.exists(merged_speaker_transcript_path):
                print(f"Merged speaker-attributed transcript already exists for '{episode['title']}'. Skipping processing.")
                continue  # Skip to next episode

            # Step 1: Transcribe
            transcription_result = transcribe_audio(original_mp3_path, transcript_path)

            # Step 2: Speaker Identification
            diarization = identify_speakers(original_mp3_path, episode_dir, known_speaker_files)

            # Step 3: Align Diarization with Transcript
            speaker_transcript = align_diarization_with_transcript(diarization, transcription_result)

            # Step 4: Merge speaker segments
            merged_speaker_transcript = merge_speaker_segments(speaker_transcript)

            # Step 5: Save the Speaker-Attributed Transcripts
            save_speaker_transcript_json(speaker_transcript, merged_speaker_transcript, episode_dir)

            # Clear GPU cache
            torch.cuda.empty_cache()

            # **Added code to delete original.mp3**
            try:
                if os.path.exists(original_mp3_path):
                    os.remove(original_mp3_path)
                    print(f"Deleted original MP3 file: {original_mp3_path}")
            except Exception as e:
                print(f"Error deleting file {original_mp3_path}: {e}")
        else:
            print(f"Original MP3 not available for '{episode['title']}'")

    # Save episode metadata
    with open('episode_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(episode_metadata, f, ensure_ascii=False, indent=4)
    print("Episode metadata saved to 'episode_metadata.json'")
