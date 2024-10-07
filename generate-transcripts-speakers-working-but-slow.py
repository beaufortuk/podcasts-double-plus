import feedparser
import whisper
from pyannote.audio import Pipeline, Inference
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchaudio
import os
import json
import nltk
import re
import requests
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import Counter
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
from pydub import AudioSegment
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from sklearn.metrics.pairwise import cosine_similarity
import requests
from requests.exceptions import HTTPError, RequestException
import gc
import io
import itertools

from typing import BinaryIO, Union

import av

# Import configurations
import config

#  Determine if device is CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable CuDNN Benchmarking for potential speed-up
torch.backends.cudnn.benchmark = True

# Load Whisper model once globally
whisper_model = whisper.load_model("tiny").to(device)

# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab')

# Initialize lemmatizer and embedding model
lemmatizer = WordNetLemmatizer()

# Get Hugging Face token
hf_token = config.HUGGING_FACE_TOKEN
if hf_token is None:
    print("Hugging Face token not found. Please set the 'HUGGING_FACE_TOKEN' environment variable.")
else:
    print("Hugging Face token found.")

# Initialize the embedding model once
embedding_model = Inference(
    "pyannote/embedding",
    window="whole",
    device=device, #specify device
    use_auth_token=hf_token
)

# Decode audio functions
def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
):
    """Decodes the audio.

    Args:
      input_file: Path to the input file or a file-like object.
      sampling_rate: Resample the audio to this sample rate.
      split_stereo: Return separate left and right channels.

    Returns:
      A float32 Numpy array.

      If `split_stereo` is enabled, the function returns a 2-tuple with the
      separated left and right channels.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

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

    # It appears that some objects related to the resampler are not freed
    # unless the garbage collector is manually run.
    del resampler
    gc.collect()

    audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

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
    # Add None to flush the resampler.
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)

def get_speaker_embedding(audio_file):
    try:
        # Read audio data using torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)
        print(f"Original waveform device: {waveform.device}")
    
        # Move waveform to the same device as the embedding model
        waveform = waveform.to(device)
        print(f"Waveform moved to device: {waveform.device}")
    
        # Prepare audio data dictionary
        audio_data = {'waveform': waveform, 'sample_rate': sample_rate}
    
        # Check if the embedding model is on the correct device
        print(f"Embedding model is on device: {embedding_model.device}")
    
        # (Optional) Check the device of the underlying model
        model_device = next(embedding_model.model_.parameters()).device
        print(f"Underlying model is on device: {model_device}")
    
        # Call embedding_model with audio data
        embedding = embedding_model(audio_data)
    
        # Check if embedding is None
        if embedding is None:
            raise ValueError("Received None from embedding_model.")
    
        # Handle torch.Tensor
        if isinstance(embedding, torch.Tensor):
            print(f"Embedding tensor device: {embedding.device}")
            embedding = embedding.detach().cpu().numpy()
        # Handle list or other iterable
        elif isinstance(embedding, (list, tuple)):
            embedding = np.array(embedding)
        # Handle single float or int
        elif isinstance(embedding, (float, int)):
            embedding = np.array([embedding])
    
        # Ensure the embedding is a 1D numeric array
        embedding = embedding.squeeze()
    
        print(f"Embedding shape for {audio_file}: {embedding.shape}")
        print(f"Embedding dtype: {embedding.dtype}")
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        embedding = None
    return embedding



# Function to compute and save known speaker embeddings
def compute_and_save_known_speaker_embeddings(known_speaker_files, embeddings_dir='embeddings'):
    os.makedirs(embeddings_dir, exist_ok=True)
    known_speakers_embeddings = {}
    
    for name, audio_file in known_speaker_files.items():
        embedding_file = os.path.join(embeddings_dir, f"{name}_embedding.npy")
        
        if os.path.exists(embedding_file):
            print(f"Embedding for {name} already exists. Loading from disk.")
            embedding = np.load(embedding_file)
        else:
            print(f"Computing embedding for {name}...")
            embedding = get_speaker_embedding(audio_file)
            np.save(embedding_file, embedding)
            print(f"Embedding for {name} saved to {embedding_file}")
        
        known_speakers_embeddings[name] = embedding
    
    return known_speakers_embeddings

# Function to identify speaker
def identify_speaker(embedding, known_speakers):
    max_similarity = -1
    identified_speaker = None
    for name, known_embedding in known_speakers.items():
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            identified_speaker = name
    return identified_speaker, max_similarity

# Function to extract audio segments
def extract_segment(audio_file, segment):
    audio = AudioSegment.from_file(audio_file)
    start_ms = int(segment.start * 1000)
    end_ms = int(segment.end * 1000)
    segment_audio = audio[start_ms:end_ms]
    return segment_audio

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
    # Create an empty list to hold the speaker-attributed transcript segments
    speaker_transcript = []

    # Iterate over each transcription segment
    for t_segment in transcription_result['segments']:
        t_start = t_segment['start']
        t_end = t_segment['end']
        t_text = t_segment['text']

        # Find the speaker for this transcription segment
        # We assume the speaker is the one who speaks during the majority of this segment
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

#  tokenize the merged text into sentences
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
    response = requests.get(enclosure_url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {save_path}")
        return True
    else:
        print(f"Failed to download {enclosure_url}")
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
        result = whisper_model.transcribe(audio_file, language='en', word_timestamps=False)
        
        # Save the transcript data to a file for future use
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"Transcript data saved to {transcript_path}")
    
    return result


# Step 2: Speaker Diarization using pyannote.audio
def identify_speakers(audio_file, output_dir):
    # RTTM file name
    rttm_filename = 'audio_with_names.rttm'
    rttm_path = os.path.join(output_dir, rttm_filename)
    
    # Check if RTTM file exists
    if os.path.exists(rttm_path):
        print(f"Diarization already exists for {audio_file}. Loading from {rttm_path}...")
        # Load the diarization from the RTTM file
        diarization_dict = load_rttm(rttm_path)
        
        # Since we have only one file, get the first Annotation
        diarization = next(iter(diarization_dict.values()))
        return diarization
    
    # Proceed with diarization if RTTM file does not exist
    # Load diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", 
        use_auth_token=hf_token
    ) 

    pipeline = pipeline.to(torch.device('cuda:0'))

    # Test to try to avoid tensor size mismatch issue
    audio = decode_audio(audio_file, 16000)
    audio_data = {
        'waveform': torch.from_numpy(audio[None, :]),
        'sample_rate': 16000
    }

    diarization = pipeline(audio_data)
    
    # **Existing code for speaker identification**
    # Define known speaker files
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
    known_speakers = compute_and_save_known_speaker_embeddings(known_speaker_files)
    
    # Extract embeddings for diarized segments
    speaker_embeddings = {}
    for segment, _, speaker_label in diarization.itertracks(yield_label=True):
        # Extract segment audio
        segment_audio = extract_segment(audio_file, segment)
        segment_file = f'temp_{speaker_label}.wav'
        segment_audio.export(segment_file, format='wav')
        embedding = get_speaker_embedding(segment_file)
        os.remove(segment_file)
        
        # Check if embedding is valid
        if embedding is not None:
            if speaker_label not in speaker_embeddings:
                speaker_embeddings[speaker_label] = []
            speaker_embeddings[speaker_label].append(embedding)
        else:
            print(f"Could not get embedding for segment {segment} and speaker {speaker_label}")
    
    # Identify speakers
    speaker_names = {}
    for speaker_label, embeddings in speaker_embeddings.items():
        print(f"Speaker {speaker_label} has {len(embeddings)} embeddings")
        shapes = [e.shape for e in embeddings]
        print(f"Embedding shapes: {shapes}")
        embeddings_array = np.array(embeddings)
        
        try:
            avg_embedding = np.mean(embeddings_array, axis=0)
            name, similarity = identify_speaker(avg_embedding, known_speakers)
            print(f"Speaker {speaker_label} identified as {name} with similarity {similarity}")
            speaker_names[speaker_label] = name if similarity > 0.7 else 'Unknown'
        except Exception as e:
            print(f"Error computing average embedding for speaker {speaker_label}: {e}")
            speaker_names[speaker_label] = 'Unknown'
    
    # Relabel diarization
    new_diarization = relabel_diarization(diarization, speaker_names)
    
    # Output speaker turns with original names
    for turn, _, speaker in new_diarization.itertracks(yield_label=True):
        # Replace underscores with spaces for printing
        display_speaker = speaker.replace('_', ' ')
        print(f"Speaker {display_speaker} from {turn.start:.1f}s to {turn.end:.1f}s")
    
    # Save the new diarization
    with open(rttm_path, "w") as rttm:
        new_diarization.write_rttm(rttm)
    print(f"Diarization saved to {rttm_path}")
    
    return new_diarization


# Taxonomy steps remain commented out while you perfect the previous functions]

# Main execution
if __name__ == "__main__":
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
        
        # Extract the first two episodes
        episodes = feed.entries[:config.NUMBER_OF_EPS_TO_ANALYSE]
        print("\nFinished parsing the RSS feed successfully.")
        
    except HTTPError as http_err:
        print(f"HTTP error occurred while fetching the RSS feed: {http_err}")
    except RequestException as req_err:
        print(f"Network-related error occurred while fetching the RSS feed: {req_err}")
    except ValueError as val_err:
        print(f"Data parsing error: {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    # Record episode metadata
    print(f"\nRecording episode metadata")
    episode_metadata = []
    for entry in episodes:
        title = entry.title
        link = entry.link
        enclosure_url = entry.enclosures[0].href
        episode_metadata.append({
            'title': title,
            'link': link,
            'enclosure_url': enclosure_url
        })
    print(f"\nFinished recording episode metadata")
    
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
            print(f"\nProcessing {episode['title']}")
            
            # Paths for transcripts and diarization
            transcript_filename = 'transcript.json'
            transcript_path = os.path.join(episode_dir, transcript_filename)
            
            # Check if the merged speaker-attributed transcript already exists
            merged_speaker_transcript_filename = 'merged_speaker_transcript.json'
            merged_speaker_transcript_path = os.path.join(episode_dir, merged_speaker_transcript_filename)
            if os.path.exists(merged_speaker_transcript_path):
                print(f"Merged speaker-attributed transcript already exists for {episode['title']}. Skipping processing.")
                continue  # Skip to next episode
            
            # Step 1: Transcribe
            transcription_result = transcribe_audio(original_mp3_path, transcript_path)
            
            # Step 2: Speaker Identification
            diarization = identify_speakers(original_mp3_path, episode_dir)
            
            # Step 3: Align Diarization with Transcript
            speaker_transcript = align_diarization_with_transcript(diarization, transcription_result)
            
            # **New**: Merge speaker segments
            merged_speaker_transcript = merge_speaker_segments(speaker_transcript)
            
            # Step 4: Save the Speaker-Attributed Transcripts
            save_speaker_transcript_json(speaker_transcript, merged_speaker_transcript, episode_dir)

            # Clear GPU cache
            torch.cuda.empty_cache()
        else:
            print(f"Original MP3 not available for {episode['title']}")
    
    # Save episode metadata
    with open('episode_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(episode_metadata, f, ensure_ascii=False, indent=4)
    print("Episode metadata saved to episode_metadata.json")

