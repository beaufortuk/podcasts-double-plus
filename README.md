# podcasts-double-plus

Podcasts-Double-Plus came about because I wanted to find more podcast episodes with Zanny as a guest speaker. Currently it's not possible to do this on the web and app. 

For each episode, Podcasts-Double-Plus generates transcriptions (Whisper w/ CUDA), identifies speakers (PyTorch w/ CUDA) and generates taxonomies (LDA), so that you can find more episodes based on a given speaker (e.g. John Prideaux) or a theme (e.g. "AI"). 

For whatever you enter it uses fuzzy logic and synonyms, to do the thinking for you ("Zanny/Zany").

A web frontend (Flask) allows you to search for something to listen to!

Instructions:
1. Install dependencies (requirements.txt).
2. Install ffmpeg and ensure /bin is in system path.
3. Install Microsoft Visual C++ 14.0 or greater. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
4. Add your Hugging Face API key to your env variables (`HUGGING_FACE_TOKEN`).
5. Add your RSS feed of podcast episodes to your env variables (`PODCAST_RSS_FEED`).
6. To create transcriptions and speaker diarizations of episodes, run `generate-transcripts-speakers-gpu.py` for CUDA-equipped machines or `generate-transcripts-speakers-cpu.py` for CPU-equipped only.
7. Run `generate-taxonomies.py` to generate metadata and taxonomies based on the transcriptions.
8. Run `recommendations.py` to find more episodes based on speaker, taxonomy, or Surprise Me with something new.
9. Deploy Flask for web interface.

To do:
* Switch from LDA topic processing to something better (BERTopic? Tried with this but got poor results).
* Create more speaker samples. Currently I have to create these manually, by snipping a segment of a speaker and saving that mp3 to .assets/samples. So far we have ~20 speaker samples. Can these be created more efficiently?
* Provide some example recommendations, dynamically (e.g. "John Prideaux")
* Move generated json files to a folder, instead of root
