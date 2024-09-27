import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import Counter
from nltk.stem import WordNetLemmatizer
from gensim.models import CoherenceModel

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Download necessary NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_text(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatizer.lemmatize(token))
    return result

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def get_topic_label(topic_words, top_n=5):
    words = [word for word, _ in topic_words[:top_n]]
    pos_words = pos_tag(words)
    
    # Prioritize nouns and adjectives
    priority_words = [word for word, pos in pos_words if pos.startswith('N') or pos.startswith('J')]
    
    if priority_words:
        # Use the most common priority word
        return Counter(priority_words).most_common(1)[0][0]
    else:
        # Fallback to the most common word overall
        return Counter(words).most_common(1)[0][0]

def train_and_print_topics(num_topics):
    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                         update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)
    
    # Evaluate topic coherence
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_articles, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Coherence Score: {coherence_lda}")
    
    # Print topics and labels
    print(f"\nTopics for model with {num_topics} topics:")
    topic_labels = {}
    for idx in range(num_topics):
        topic_words = lda_model.get_topic_terms(idx, topn=10)
        words_only = [(dictionary[id], value) for id, value in topic_words]
        label = get_topic_label(words_only)
        topic_labels[idx] = label
        print(f"Topic {idx} ('{label}'): \n{words_only}\n")
    
    return lda_model, topic_labels

def suggest_alternative_labels(topic_words, n=3):
    words = [word for word, _ in topic_words[:10]]
    pos_words = pos_tag(words)
    nouns = [word for word, pos in pos_words if pos.startswith('N')]
    return Counter(nouns).most_common(n)

# Main module guard required for multiprocessing on Windows
if __name__ == "__main__":

    # Expanded corpus of articles
    articles = [
        "The global economy shows signs of recovery as countries adapt to post-pandemic realities.",
        "Climate change continues to be a pressing issue, with rising sea levels threatening coastal cities.",
        "Advancements in artificial intelligence are reshaping various industries, from healthcare to finance.",
        "Political tensions between major powers are increasing, raising concerns about global stability.",
        "The stock market reached new highs today, driven by strong performances in the tech sector.",
        "Renewable energy sources are becoming increasingly competitive with fossil fuels.",
        "A new study reveals the impact of social media on mental health and societal polarization.",
        "Cybersecurity threats are on the rise as more businesses shift to remote work environments.",
        "The COVID-19 pandemic has accelerated the adoption of digital technologies across all sectors.",
        "Central banks worldwide are considering the implementation of digital currencies.",
        "The growing wealth gap is fueling discussions about universal basic income and wealth taxes.",
        "Advances in biotechnology are opening new frontiers in medicine and agriculture.",
        "The space industry is seeing a resurgence with private companies leading the charge.",
        "Global supply chain disruptions are forcing companies to rethink their logistics strategies.",
        "The rise of e-commerce is reshaping retail landscapes in cities around the world.",
        "Governments are grappling with the regulation of big tech companies and data privacy.",
        "The future of work is being redefined with the growth of remote and hybrid work models.",
        "Renewable energy investments are reaching record levels as countries aim to meet climate goals.",
        "Artificial intelligence is playing an increasing role in scientific research and discovery.",
        "The global semiconductor shortage is affecting industries from automotive to consumer electronics.",
        "Demographic shifts in developed countries are putting pressure on pension systems and healthcare.",
        "The rise of cryptocurrencies is challenging traditional notions of money and financial systems.",
        "Urbanization trends are shaping city planning and infrastructure development worldwide.",
        "The sharing economy continues to disrupt traditional business models across various sectors.",
        "Advances in quantum computing are promising to revolutionize fields like cryptography and drug discovery.",
        "The growing importance of ESG (Environmental, Social, and Governance) factors in investment decisions.",
        "The impact of automation and robotics on employment and the future job market.",
        "Water scarcity is becoming a critical issue in many regions, affecting agriculture and urban planning.",
        "The role of social media in shaping public opinion and political discourse.",
        "The development of autonomous vehicles and their potential impact on transportation and urban design."
    ]

    processed_articles = [preprocess_text(doc) for doc in articles]
    dictionary = corpora.Dictionary(processed_articles)
    corpus = [dictionary.doc2bow(text) for text in processed_articles]

    # Train model with 5 topics and evaluate coherence
    lda_model, topic_labels = train_and_print_topics(5)

    # Classify a new article
    new_article = "The blackwell chip from Nvidia, shovel-maker for the artificial-intelligence (AI) gold rush, contains 208bn transistors spread over two dies..."
    bow_vector = dictionary.doc2bow(preprocess_text(new_article))

    print("\nTopic distribution for the new article:")
    for topic_id, probability in lda_model.get_document_topics(bow_vector):
        print(f"Topic {topic_id} ('{topic_labels[topic_id]}'): {probability:.4f}")
        topic_words = lda_model.get_topic_terms(topic_id, topn=5)
        words_only = [(dictionary[id], value) for id, value in topic_words]
        print(f"Top words: {words_only}")
        print()

    # Function to suggest alternative labels
    print("\nSuggested alternative labels for each topic:")
    for idx in range(lda_model.num_topics):
        topic_words = lda_model.get_topic_terms(idx, topn=10)
        words_only = [(dictionary[id], value) for id, value in topic_words]
        alternatives = suggest_alternative_labels(words_only)
        print(f"Topic {idx} (current: '{topic_labels[idx]}')")
        print(f"Alternatives: {alternatives}\n")
