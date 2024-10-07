# podcasts-double-plus

Podcasts-Double-Plus came about because I wanted to find more Economist podcast episodes with Zanny as a guest speaker. Currently it's not possible to do this on economist.com. 

For each episode, Podcasts-Double-Plus creates transcriptions, identifies speakers and generates taxonomies, so that you can find more episodes based on a given speaker (e.g. John Prideaux) or a theme (e.g. "AI"). 

Instructions:
1. Install dependencies (requirements.txt).
2. Install ffmpeg and ensure /bin is in system path.
3. Install Microsoft Visual C++ 14.0 or greater. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
4. Add your Hugging Face API key to your OS env variables (`HUGGING_FACE_TOKEN`).
5. Add your RSS feed of podcast episodes to your OS env variables (`PODCAST_RSS_FEED`).
6. Run `generate-transcripts-speakers.py` to create transcriptions and speaker diarizations of episodes.
7. Run `generate-taxonomies.py` to generate metadata and taxonomies based on the transcriptions.
8. Run `recommendations.py` to find more episodes based on speaker, taxonomy, or Surprise Me with something new.

To do:
* Switch from CPU to CUDA device to improve processing. Current stuck with CPU.
* Once switched to CUDA, transcribe entire episodes. ATM `generate-taxonomies.py` only transcribes and diarizes x3 random 60sec samples of each episode, for the first 20 episodes in the feed. This is to reduce processing time while we are stuck using CPU processing.
* Switch from LDA topic processing to something better (BERTopic? Tried with this but got poor results).
* Create more speaker samples. Currently I have to create these manually, by snipping a segment of a speaker and saving that mp3 to .assets/samples. So far we have ~20 speaker samples. Can these be created more efficiently?
* Provide some example recommendations, dynamically (e.g. "John Prideaux")
* Add to config: no. of eps to download, entire eps or just x samples
* Move generated json files to a folder, instead of root
* Direct recommendations to an economist.com episode page, instead of an acast playback page.
* Upgrade Whisper model from "tiny" to one that's more accurate (using tiny right now for speed during development)
