import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
import nltk 
from zipfile import ZipFile
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from streamlit_lottie import st_lottie
from kaggle.api.kaggle_api_extended import KaggleApi
# --- DOWNLOAD NLTK DATA ---
# This ensures the necessary packages are available in the Streamlit environment
nltk.download('stopwords')
nltk.download('punkt')

# --- PAGE CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="Zomato Recommender Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a modern look and feel
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #F0F2F6;
    }
    /* Card style for recommendations */
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }
    .card:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transform: translateY(-5px);
    }
    .card h3 {
        color: #E23744; /* Zomato Red */
        margin-bottom: 5px;
    }
    .card p {
        color: #4F4F4F;
        font-size: 1rem;
        line-height: 1.4;
    }
    /* Style for the sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# --- LOTTIE ANIMATION LOADER ---
def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL."""
    import requests
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- DATA LOADING AND PREPROCESSING ---
@st.cache_data
def load_and_preprocess_data():
    """
    Downloads the dataset from Kaggle using API credentials from st.secrets,
    unzips it, preprocesses it, and returns a clean DataFrame.
    """
    # --- 1. Kaggle API Setup ---
    # Create the .kaggle directory and write the secrets to kaggle.json
    # This is done in memory and is secure
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    kaggle_credentials = f'{{"username":"{st.secrets.kaggle.username}","key":"{st.secrets.kaggle.key}"}}'
    
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(kaggle_credentials)

    # Set correct permissions for the kaggle.json file
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

    # --- 2. Download and Unzip the Dataset ---
    dataset_slug = "rajeshrampure/zomato-dataset"
    download_path = "./temp_data"
    os.makedirs(download_path, exist_ok=True)
    
    api = KaggleApi()
    api.authenticate()

    #st.info("Downloading dataset from Kaggle... this may take a moment.")
    api.dataset_download_files(dataset_slug, path=download_path, quiet=True) # Set quiet=True for cleaner logs

    # Unzip the downloaded file
    with ZipFile(os.path.join(download_path, f"{dataset_slug.split('/')[1]}.zip"), 'r') as zip_ref:
        zip_ref.extractall(download_path)
    
    # --- 3. Load and Preprocess Data with Pandas ---
    #st.info("Preprocessing data...")
    df = pd.read_csv(os.path.join(download_path, "zomato.csv"))
    
    df1 = df.head(10000) # Using a sample for performance

    # Basic cleaning
    df1 = df1.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(city)': 'city'})
    df1.dropna(subset=['name', 'cost', 'rate', 'reviews_list', 'cuisines', 'city'], inplace=True)
    df1['cost'] = df1['cost'].astype(str).apply(lambda x: x.replace(',', '')).astype(float)
    df1 = df1[df1.rate.apply(lambda x: isinstance(x, str) and '/5' in x)].copy()
    df1['rate'] = df1['rate'].apply(lambda x: float(x.strip().replace('/5', ''))).astype(float)
    df1['name'] = df1['name'].str.title()

    # Calculate Mean Rating for better recommendations
    mean_ratings = df1.groupby('name')['rate'].mean().round(2).reset_index()
    mean_ratings.rename(columns={'rate': 'mean_rating'}, inplace=True)
    df1 = pd.merge(df1, mean_ratings, on='name', how='left')

    # Text preprocessing for the review-based model
    stop_words = set(stopwords.words('english'))
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'rated\s*\d+\.?\d*', '', text)
        text = re.sub(r'ratedn', '', text)
        text = re.compile(r'https?://\S+|www\.\S+').sub('', text)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    df1['reviews_list'] = df1['reviews_list'].apply(clean_text)

    # Final selection of columns and dropping duplicates for the main app dataframe
    df_processed = df1[['name', 'city', 'cuisines', 'cost', 'mean_rating', 'reviews_list']].drop_duplicates(subset=['name'])
    
    #st.success("Data loaded and processed successfully!")
    return df_processed.reset_index(drop=True)

# --- SIMILARITY MATRIX ---
@st.cache_resource
def create_similarity_matrix(_df):
    """Creates TF-IDF and Cosine Similarity matrices."""
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, stop_words='english', max_features=7000)
    tfidf_matrix = tfidf.fit_transform(_df['reviews_list'])
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_similarities, _df

# --- UI & APP LOGIC ---

# Load data and build models
df = load_and_preprocess_data()
cosine_sim, df_sim = create_similarity_matrix(df)

# Sidebar for Filters
with st.sidebar:
    st.header("🔍 Find Your Next Meal")
    animation_url = "https://lottie.host/86c67531-5975-48b0-a6ff-8e244b25691c/F5kIe9z7mW.json"
    lottie_anim = load_lottieurl(animation_url)
    if lottie_anim:
        st_lottie(lottie_anim, height=200, key="sidebar_animation")

    st.subheader("Discover by Preference")
    selected_city = st.selectbox("City", options=sorted(df['city'].unique()))

    # Dynamic cuisine multiselect based on city
    available_cuisines = df[df['city'] == selected_city]['cuisines'].str.split(', ').explode().str.strip().unique()
    selected_cuisines = st.multiselect("Cuisines", options=sorted(available_cuisines))

    # Dynamic cost slider based on city
    city_df = df[df['city'] == selected_city]
    min_cost, max_cost = int(city_df['cost'].min()), int(city_df['cost'].max())
    cost_range = st.slider("Cost for Two (₹)", min_cost, max_cost, (min_cost, max_cost))

# Main Page
st.title("Zomato Recommender Pro")
st.markdown("Discover new restaurants by filtering your preferences or find places similar to your favorites!")

tab1, tab2 = st.tabs(["**✨ Discover Restaurants**", "**👯 Find Similar Restaurants**"])

# --- TAB 1: DISCOVER BY PREFERENCE ---
with tab1:
    st.header("Tell us what you're craving!")
    if st.button("Find Restaurants", type="primary"):
        with st.spinner("Searching for the best spots..."):
            # Filtering logic
            filtered_df = df[
                (df['city'] == selected_city) &
                (df['cost'] >= cost_range[0]) &
                (df['cost'] <= cost_range[1])
            ]
            if selected_cuisines:
                filtered_df = filtered_df[filtered_df['cuisines'].apply(lambda x: all(c in x for c in selected_cuisines))]

            if filtered_df.empty:
                st.warning("No restaurants match your criteria. Try being a bit more flexible with your filters!")
            else:
                results = filtered_df.sort_values('mean_rating', ascending=False).head(10)
                st.success(f"Found {len(results)} great options for you!")
                for _, row in results.iterrows():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"""<div class="card"><h3>{row['name']}</h3><p><strong>Cuisines:</strong> {row['cuisines']}</p><p><strong>Location:</strong> {row['city']}</p></div>""", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""<div class="card"><p style="text-align:center; font-size: 1.2rem;"><strong>⭐ {row['mean_rating']}/5</strong></p><p style="text-align:center;"><strong>₹{int(row['cost'])}</strong> for two</p></div>""", unsafe_allow_html=True)

# --- TAB 2: FIND SIMILAR RESTAURANTS (REVIEW-BASED) ---
with tab2:
    st.header("Find places similar to a restaurant you love!")
    restaurant_names = sorted(df_sim['name'].unique())
    selected_restaurant = st.selectbox("Select a restaurant:", options=restaurant_names, key="similar_select")

    if st.button("Find Similar", type="primary", key="similar_button"):
        with st.spinner(f"Finding recommendations like '{selected_restaurant}'..."):
            try:
                indices = pd.Series(df_sim.index, index=df_sim['name'])
                idx = indices[selected_restaurant]
                
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:11]  # Top 10, excluding itself
                restaurant_indices = [i[0] for i in sim_scores]
                results = df_sim.iloc[restaurant_indices]

                st.success(f"Top 10 restaurants with reviews similar to '{selected_restaurant}':")
                for _, row in results.iterrows():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                         st.markdown(f"""<div class="card"><h3>{row['name']}</h3><p><strong>Cuisines:</strong> {row['cuisines']}</p><p><strong>Location:</strong> {row['city']}</div>""", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""<div class="card"><p style="text-align:center; font-size: 1.2rem;"><strong>⭐ {row['mean_rating']}/5</strong></p><p style="text-align:center;"><strong>₹{int(row['cost'])}</strong> for two</p></div>""", unsafe_allow_html=True)
            except Exception:
                st.error(f"Could not find recommendations for '{selected_restaurant}'. Please try another.")
