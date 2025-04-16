from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

TMDB_API_KEY = '0527fb6d99e09b2225d6b39f89bd6334'
app = Flask(__name__)

# ----------------------------
# Load dữ liệu
# ----------------------------
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies = pd.read_csv(os.path.join(base_dir, 'ml-1m', 'movies.dat'), sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='latin-1')
    ratings = pd.read_csv(os.path.join(base_dir, 'ml-1m', 'ratings.dat'), sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'])
    return movies, ratings

movies_df, ratings_df = load_data()

# ----------------------------
# Content-based Filtering
# ----------------------------
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower())

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

def get_recommendations(title, top_n=10):
    title = title.lower()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]

    results = []
    for i in movie_indices:
        try:
            movie_title = movies_df['title'].iloc[i]
            if movie_title.lower() == title:
                continue
            movie_id = int(movies_df['movieId'].iloc[i])
            genres_raw = movies_df['genres'].iloc[i]
            genres = genres_raw.replace('|', ', ') 
            poster_url = fetch_poster(movie_title)
            results.append({
                'title': movie_title,
                'id': movie_id,
                'genres': genres,
                'poster': poster_url
            })
        except Exception as e:
            print(f"❌ Lỗi khi xử lý phim tại index {i}: {e}")
            continue
    return results


# ----------------------------
# Mô hình Neural Collaborative Filtering
# ----------------------------
class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=50):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.fc1 = nn.Linear(emb_size * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, user, item):
        u = self.user_emb(user)
        i = self.item_emb(item)
        x = torch.cat([u, i], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

# ----------------------------
# Load mô hình NCF đã huấn luyện
# ----------------------------
import joblib

user_encoder = joblib.load('user_encoder.pkl')
item_encoder = joblib.load('item_encoder.pkl')

n_users = len(user_encoder.classes_)
n_items = len(item_encoder.classes_)

ncf_model = NCF(n_users, n_items)
try:
    state_dict = torch.load('ncf_model.pt', map_location='cpu')
    ncf_model.load_state_dict(state_dict)
    ncf_model.eval()

    print("✅ Mô hình NCF đã sẵn sàng")
except Exception as e:
    print("❌ Lỗi khi load mô hình NCF:", e)

def get_user_recommendations(user_id, top_n=10):
    try:
        user_idx = user_encoder.transform([user_id])[0]  # encode user_id
    except ValueError:
        return []

    # Tạo tensor user lặp lại cho tất cả items
    user_tensor = torch.tensor([user_idx] * n_items)
    item_tensor = torch.tensor(list(range(n_items)))

    with torch.no_grad():
        scores = ncf_model(user_tensor, item_tensor).squeeze()

    # Lấy top N item index
    _, top_indices = torch.topk(scores, top_n)

    results = []
    for idx in top_indices:
        try:
            item_idx = idx.item()
            movie_id = int(item_encoder.classes_[item_idx])  # lấy movieId gốc từ chỉ số item

            movie_row = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            title = movie_row['title']
            genres = movie_row['genres'].replace('|', ', ')
            poster_url = fetch_poster(title)

            results.append({
                'title': title,
                'id': movie_id,
                'genres': genres,
                'poster': poster_url
            })
        except Exception as e:
            print(f"❌ Lỗi khi xử lý phim có index {item_idx}: {e}")
            continue

    return results


from collections import defaultdict

def analyze_user_preferences(user_id):
    user_ratings = ratings_df[ratings_df['userId'] == user_id]
    if user_ratings.empty:
        return {}

    merged = user_ratings.merge(movies_df, on='movieId')
    genre_scores = defaultdict(list)  # lưu danh sách các rating cho từng thể loại

    # Gán rating cho từng thể loại (không chia đều)
    for _, row in merged.iterrows():
        genres = row['genres'].split('|')
        rating = row['rating']
        for g in genres:
            genre_scores[g].append(rating)

    # Tính điểm trung bình mỗi thể loại
    genre_avg = {
        genre: round(sum(scores) / len(scores), 2)
        for genre, scores in genre_scores.items()
        if scores  # tránh chia cho 0
    }

    # Sắp xếp theo điểm trung bình giảm dần
    genre_avg = dict(sorted(genre_avg.items(), key=lambda x: x[1], reverse=True))
    return genre_avg

# ----------------------------
# API Routes
# ----------------------------
@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    if not title:
        return jsonify({'error': 'Missing "title" parameter'}), 400
    recommendations = get_recommendations(title)
    if not recommendations:
        return jsonify({'error': f'No movie found for title: {title}'}), 404
    return jsonify({'input': title, 'recommendations': recommendations})

@app.route('/recommend_user', methods=['GET'])
def recommend_user():
    try:
        user_id = int(request.args.get('user_id'))
    except:
        return jsonify({'error': 'Invalid or missing "user_id" parameter'}), 400
    recommendations = get_user_recommendations(user_id)
    if not recommendations:
        return jsonify({'error': f'No recommendations for user_id: {user_id}'}), 404
    return jsonify({'user_id': user_id, 'recommendations': recommendations})

@app.route('/user_profile', methods=['GET'])
def user_genres():
    try:
        user_id = int(request.args.get('user_id'))
    except:
        return jsonify({'error': 'Invalid or missing "user_id" parameter'}), 400

    genre_pref = analyze_user_preferences(user_id)
    if not genre_pref:
        return jsonify({'error': f'No genre data for user_id: {user_id}'}), 404

    return jsonify({'user_id': user_id, 'genre_preferences': genre_pref})

@app.route('/titles')
def titles():
    return jsonify(movies_df['title'].tolist())

@app.route('/')
def home():
    return render_template('index_ai.html')

if __name__ == '__main__':
    app.run(debug=True)
