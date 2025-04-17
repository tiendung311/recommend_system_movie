from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import numpy as np
import os
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
import ast

GENRE_MAP = {
    "0": "Action", "1": "Adventure", "2": "Animation", "3": "Children",
    "4": "Comedy", "5": "Crime", "6": "Documentary", "7": "Drama",
    "8": "Fantasy", "9": "Film-Noir", "10": "Horror", "11": "Musical",
    "12": "Mystery", "13": "Romance", "14": "Sci-Fi", "15": "Thriller",
    "16": "War", "17": "Western", "18": "IMAX", "19": "Noir"
}

TMDB_API_KEY = '0527fb6d99e09b2225d6b39f89bd6334'
app = Flask(__name__)
app.secret_key = 'None'  

# ----------------------------
# Load data
# ----------------------------
def decode_genres(genre_str):
    ids = str(genre_str).split(',')
    names = [GENRE_MAP.get(id.strip(), f"ID {id.strip()}") for id in ids]
    return ', '.join(names)

def genre_score_boost(genre_str):
    user_fav_genres = {"Comedy", "Animation", "Sci-Fi"}  # 💡 customize later
    genres = decode_genres(genre_str).split(', ')
    if not genres:
        return 1.0
    boost = 1.0
    if genres[0] in user_fav_genres:
        boost += 0.2
    if any(g in user_fav_genres for g in genres[1:]):
        boost += 0.1
    return boost

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    movies_metadata = pd.read_csv('recsys_export_bundle/movies_metadata.csv')
    item_factors = pd.read_csv('recsys_export_bundle/item_factors.csv')

    with open('recsys_export_bundle/item2idx.json', 'r') as f:
        item2idx = json.load(f)
    with open('recsys_export_bundle/idx2item.json', 'r') as f:
        idx2item = json.load(f)

    ncf_user_embeddings = np.load('recsys_export_bundle/ncf_user_embeddings.npy')
    ncf_item_embeddings = np.load('recsys_export_bundle/ncf_item_embeddings.npy')

    return movies_metadata, item_factors, item2idx, idx2item, ncf_user_embeddings, ncf_item_embeddings

movies_metadata, item_factors, item2idx, idx2item, ncf_user_embeddings, ncf_item_embeddings = load_data()
print("✅ All data loaded successfully")

item_embeddings = np.vstack(item_factors['features'].apply(ast.literal_eval).values)
item_similarity = cosine_similarity(item_embeddings)

def fetch_poster(title):
    query = title.split('(')[0]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
    response = requests.get(url)
    data = response.json()
    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/300x450?text=No+Image"

def get_user_recommendations(user_id, top_n=10):
    try:
        user_idx = int(user_id)
        user_vector = ncf_user_embeddings[user_idx]
    except Exception as e:
        print(f"❌ Invalid user: {e}")
        return []

    scores = np.dot(user_vector, ncf_item_embeddings.T)
    scored_items = list(enumerate(scores))

    # Apply genre-based score boost
    for i, (idx, base_score) in enumerate(scored_items):
        try:
            movie_id = int(idx2item[str(idx)])
            genre_raw = movies_metadata[movies_metadata['movie_id'] == movie_id]['movie_genres'].iloc[0]
            scored_items[i] = (idx, base_score * genre_score_boost(genre_raw))
        except:
            continue

    top_indices = [idx for idx, _ in sorted(scored_items, key=lambda x: x[1], reverse=True)[:top_n]]

    results = []
    for idx in top_indices:
        try:
            movie_id = int(idx2item[str(idx)])
            movie = movies_metadata[movies_metadata['movie_id'] == movie_id].iloc[0]
            results.append({
                'title': movie['movie_title'],
                'id': movie_id,
                'genres': decode_genres(movie['movie_genres']),
                'poster': fetch_poster(movie['movie_title'])
            })
        except Exception as e:
            print(f"❌ Error at idx {idx}: {e}")
            continue

    return results

def get_similar_movies(movie_id, n=10):
    try:
        if str(movie_id) not in item2idx:
            return []

        idx = item2idx[str(movie_id)]
        sim_scores = item_similarity[idx]
        similar_indices = sim_scores.argsort()[::-1][1:n+1]

        results = []
        for idx in similar_indices:
            movie_id = idx2item[str(idx)]
            movie = movies_metadata[movies_metadata['movie_id'] == int(movie_id)].iloc[0]
            results.append({
                'title': movie['movie_title'],
                'id': int(movie_id),
                'genres': decode_genres(movie['movie_genres']),
                'poster': fetch_poster(movie['movie_title'])
            })
        return results
    except Exception as e:
        print(f"Error getting similar movies: {e}")
        return []

@app.route('/')
def home():
    return render_template('index_ai.html')

@app.route('/recommend_user')
def recommend_user():
    try:
        user_id = int(request.args.get('user_id'))
    except:
        return jsonify({'error': 'Invalid or missing user_id'}), 400

    recommendations = get_user_recommendations(user_id)
    if not recommendations:
        return jsonify({'error': 'No recommendations found'}), 404
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

@app.route('/similar/<int:movie_id>')
def similar(movie_id):
    movies = get_similar_movies(movie_id)
    if not movies:
        return jsonify({'error': 'No similar movies found'}), 404
    return jsonify({'movie_id': movie_id, 'recommendations': movies})

@app.route('/search_movies')
def search_movies():
    query = request.args.get('query', '').lower()
    if not query:
        return jsonify([])

    matching_movies = movies_metadata[
        movies_metadata['movie_title'].str.lower().str.contains(query)
    ].head(10)

    results = []
    for _, movie in matching_movies.iterrows():
        results.append({
            'id': int(movie['movie_id']),
            'title': movie['movie_title'],
            'genres': decode_genres(movie['movie_genres']),
            'poster': fetch_poster(movie['movie_title'])
        })
    return jsonify(results)

@app.route('/login', methods=['POST'])
def login():
    user_id = request.json.get('user_id')
    try:
        user_id = int(user_id)
        if 0 <= user_id < len(ncf_user_embeddings):
            session['user_id'] = user_id
            return jsonify({'success': True, 'user_id': user_id})
        else:
            return jsonify({'success': False, 'error': 'Invalid user ID'}), 400
    except:
        return jsonify({'success': False, 'error': 'Invalid user ID format'}), 400

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True})

@app.route('/get_current_user')
def get_current_user():
    return jsonify({'user_id': session.get('user_id')})

if __name__ == '__main__':
    app.run(debug=True)