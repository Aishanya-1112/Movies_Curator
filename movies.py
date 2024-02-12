import streamlit as st
import base64
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# CSS for center-aligning the header and styling the line
header_style = """
    <style>
       body {
background-image: url("https://www.google.com/url?sa=i&url=https%3A%2F%2Fpixabay.com%2Fimages%2Fsearch%2Fnature%2F&psig=AOvVaw1oTsq5dnOUtKU-XYobt42B&ust=1707849694958000&source=images&cd=vfe&opi=89978449&ved=0CBMQjRxqFwoTCKj3zIW6poQDFQAAAAAdAAAAABAE");
background-size: cover;
}
        .header {
            color: #fff;
            text-align: center;
            padding: 0;
        }
        .caption {
            color: #fff;
            text-align: center;
            margin-top: 0;
            padding-top: 100px:
        }
        .line {
           
            border-bottom: 2px dashed #f85a40 #ccc;
            margin-bottom: 20px;
            padding-bottom: 80px;
        }
       
    </style>
"""




# Adding the CSS to the Streamlit app
st.markdown(header_style, unsafe_allow_html=True)

# Header with center alignment and line separator
st.markdown("<h1 class='header'>The Curator<span style='color: #f85a40;'>.</span></h1>", unsafe_allow_html=True)


st.markdown("<h3 class='caption'>Movie Recommendation Tool</h3>", unsafe_allow_html=True)
st.markdown("<div class='line'></div>", unsafe_allow_html=True)

# Load movie data
movies = pd.read_csv(r"top10K-TMDB-movies.csv")

# Convert the genre column to a list of genres, handling NaN values
movies['genre'] = movies['genre'].apply(lambda x: x.split(',') if pd.notna(x) else [])

def get_movie_recommendations(movie_title, previous_movies, genres):
    df_excluded = movies[(~movies['title'].isin(previous_movies)) & (movies['title'] != movie_title)]
    
    # Filter by genre if selected
    if genres:
        df_excluded = df_excluded[df_excluded['genre'].apply(lambda x: any(g in x for g in genres))]

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_excluded['overview'].fillna(''))
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_vectorizer.transform([movies[movies['title'] == movie_title]['overview'].iloc[0]]))
    similar_movies_indices = cosine_similarities.flatten().argsort()[::-1]
    similar_movies = df_excluded.iloc[similar_movies_indices]['title'].head(5).tolist()
    posters = [fetch_poster(movie) for movie in similar_movies]
    return similar_movies, posters

def fetch_poster(movie_title):
    movie_id = movies[movies['title'] == movie_title]['id'].iloc[0]
    url = "https://api.themoviedb.org/3/movie/{}?api_key=dec4731f59413ae816d86ea96c1b1677&language=en-US".format(
        movie_id)
    data = requests.get(url).json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def get_movie_details(movie_title):
    movie_id = movies[movies['title'] == movie_title]['id'].iloc[0]
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=dec4731f59413ae816d86ea96c1b1677&language=en-US"
    data = requests.get(url).json()
    return data

# Streamlit app
def app():
    # st.header("Movie Recommender")
    # st.markdown(selectbox_style, unsafe_allow_html=True)
    # Select a movie the user previously liked
    selected_value = st.selectbox("Select the Movie you previously liked", movies['title'].values, key='movie_select', help='### Select the movie you previously liked')

    # Show Recommendations button
    if st.button("Show Recommendations"):
        recommendations, posters = get_movie_recommendations(selected_value, [], [])
        
        # Display recommendations and details
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, (recommendation, poster) in enumerate(zip(recommendations, posters)):
            with locals()[f"col{i + 1}"]:
                # Use HTML styling to highlight the movie name with a black background and capitalize it
                st.markdown(f'<div style="background-color: black; padding: 10px; text-align: center; text-transform: uppercase;">{recommendation}</div>', unsafe_allow_html=True)
                st.image(poster)
                
                # Display overview details without an expander
                st.subheader(f"Overview for {recommendation}")
                movie_details = get_movie_details(recommendation)
                st.write("**Overview:**", movie_details['overview'])
                st.write("**Popularity:**", movie_details['popularity'])
                st.write("**Release Date:**", movie_details['release_date'])
                st.write("**Rating:**", movie_details['vote_average'])

    # Advanced Search options
    # st.subheader("Advanced Search Options:")
    advanced_search_expander = st.expander("Advanced Search Options")

    # Checkbox widgets for individual genres
    selected_genres = set()
    for genre in movies['genre'].explode().unique():
        selected = advanced_search_expander.checkbox(str(genre))
        if selected:
            selected_genres.add(genre)

    # Recommendations button for advanced search
    advanced_recommendations_pressed = st.button("Show Advanced Recommendations")
    if advanced_recommendations_pressed:
        recommendations, posters = get_movie_recommendations(selected_value, [], list(selected_genres))
        
        # Display recommendations and details for the advanced search
        col1, col2, col3, col4, col5 = st.columns(5)
        for i, (recommendation, poster) in enumerate(zip(recommendations, posters)):
            with locals()[f"col{i + 1}"]:
                # Use HTML styling to highlight the movie name with a black background and capitalize it
                st.markdown(f'<div style="background-color: black; padding: 10px; text-align: center; text-transform: uppercase;">{recommendation}</div>', unsafe_allow_html=True)
                st.image(poster)
                
                # Display overview details without an expander
                st.subheader(f"Overview for {recommendation}")
                movie_details = get_movie_details(recommendation)
                st.write("**Overview:**", movie_details['overview'])
                st.write("**Popularity:**", movie_details['popularity'])
                st.write("**Release Date:**", movie_details['release_date'])
                st.write("**Rating:**", movie_details['vote_average'])

# Run the Streamlit app
if __name__ == "__main__":
    app()
