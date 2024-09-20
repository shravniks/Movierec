import pandas as pd
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, url_for, redirect

app = Flask(__name__)

# Load and preprocess data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'popularity']]
movies.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 4:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert2)

def convert3(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i.get('job') == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(convert3)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_movies = movies[['movie_id', 'title', 'tags', 'popularity']]
new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x))
new_movies['tags'] = new_movies['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_movies['tags'] = new_movies['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_movies['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_movies.iloc[i[0]].title for i in movies_list]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/page2')
def page2():
    return render_template('page2.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    movie_name = request.form['movie_name']
    try:
        recommendations = recommend(movie_name)
        return render_template('page2.html', recommendations=recommendations, movie_name=movie_name)
    except:
        return render_template('page2.html', error="Movie not found. Please try another movie.")

if __name__ == '__main__':
    app.run(debug=True)