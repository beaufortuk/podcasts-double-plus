# app.py

from flask import Flask, render_template, request
import os
import recommendations  # Import your recommendation module

app = Flask(__name__)

# Load preprocessed data once when the app starts
try:
    episodes, inverted_index, speakers = recommendations.load_preprocessed_data()
except Exception as e:
    print(f"Failed to load preprocessed data: {e}")
    episodes = {}
    inverted_index = {}
    speakers = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('keyword', '').strip()
    if keyword:
        recommended_episodes = recommendations.recommend_by_keyword(episodes, inverted_index, keyword)
        return render_template(
            'results.html',
            episodes=recommended_episodes,
            query=keyword,
            type='Search Results'
        )
    else:
        return render_template('error.html', message="Please enter a keyword.")

@app.route('/speaker', methods=['POST'])
def speaker():
    speaker_name = request.form.get('speaker_name', '').strip()
    if speaker_name:
        recommended_episodes = recommendations.recommend_by_speaker(episodes, speakers, speaker_name)
        return render_template(
            'results.html',
            episodes=recommended_episodes,
            query=speaker_name,
            type='Speaker Recommendations (sorted by airtime)'
        )
    else:
        return render_template('error.html', message="Please enter a speaker's name.")

@app.route('/explore', methods=['POST'])
def explore():
    user_keywords = request.form.get('keywords', '').split(',')
    user_keywords = [kw.strip() for kw in user_keywords if kw.strip()]
    if user_keywords:
        recommended_episodes = recommendations.recommend_explore_more(episodes, user_keywords)
        return render_template(
            'results.html',
            episodes=recommended_episodes,
            query=', '.join(user_keywords),
            type='Explore More'
        )
    else:
        return render_template('error.html', message="Please enter at least one keyword to explore.")

@app.route('/surprise')
def surprise():
    recommended_episodes = recommendations.recommend_surprise_me(episodes)
    return render_template(
        'results.html',
        episodes=recommended_episodes,
        type='Surprise Me'
    )

@app.route('/understand_more')
def understand_more():
    recommended_episodes = recommendations.recommend_understand_more(episodes)
    return render_template(
        'results.html',
        episodes=recommended_episodes,
        type='Understand More'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default to 8080 if PORT isn't set
    app.run(host='0.0.0.0', port=port)
