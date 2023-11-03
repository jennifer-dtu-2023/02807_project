#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from itertools import combinations
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from zipfile import ZipFile
import warnings
import os


# used csv files, linking it with their source zip files
csv_to_zip_source = {
    "Netflix_Dataset_Movie.csv": "02807_project/zip_sources/Netflix_Dataset_Movie.csv.zip",
    "Netflix_Dataset_Rating.csv": "02807_project/zip_sources/Netflix_Dataset_Rating.csv.zip",
    "tmdb_5000_credits.csv": "02807_project/zip_sources/tmdb.zip",
    "tmdb_5000_movies.csv": "02807_project/zip_sources/tmdb.zip"
}

csv_file_names = list(csv_to_zip_source.keys())

# common data directory path
data_dir = "./data"

# if directory does not exist, create it
if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

# check if all .csv can be found in data directory, if not we extract it from corresponding zip source
for csv_file_name in csv_file_names:
    if not os.path.exists(f"{data_dir}/{csv_file_name}"):
        # if it is not found
        zip_source_file_name = csv_to_zip_source[csv_file_name]
        print(f"❗'{csv_file_name}' does not exist in the '{data_dir}' directory, extracting it from zip file '{zip_source_file_name}'...")
        with ZipFile(zip_source_file_name, 'r') as zip:
            zip.extract(csv_file_name, path=data_dir)
            print(f"Done extracting {csv_file_name} from {zip_source_file_name}")
    else:
        # csv exists and found, let user know
        print(f"'{data_dir}/{csv_file_name}' exists ✅")
    print("---")

movies = pd.read_csv(f"{data_dir}/Netflix_Dataset_Movie.csv")
ratings = pd.read_csv(f"{data_dir}/Netflix_Dataset_Rating.csv")
credits = pd.read_csv(f"{data_dir}/tmdb_5000_credits.csv")
tmdb_movies = pd.read_csv(f"{data_dir}/tmdb_5000_movies.csv")
movies


ratings


credits


tmdb_movies


df_movies = movies[movies.Name.isin(credits.title)]
df_movies


df_ratings = ratings[ratings.Movie_ID.isin(df_movies.Movie_ID)]
df_ratings


# ### Initial EDA analysis as a foundation for the methods in W5 (Apripro/FP-Growth for frequent item-set mining)

# Basic statistics: mean, median, and standard deviation
# 
# For df_rating: Finding out these values for the Rating column.
# 
# For df_movies: Finding out thhe distribution of movies across years. This will to contexualise the ratings to help understand viewer taste and perhaps how rating behaviour changes over time/movie release time.


# For df_ratings
mean_ratings = df_ratings['Rating'].mean()
median_rating = df_ratings['Rating'].median()
std_rating = df_ratings['Rating'].std()

# For df_movies
mean_year = df_movies['Year'].mean()
median_year = df_movies['Year'].median()
std_year = df_movies['Year'].std()


# Visualisations:
# - Using a histogram for the Rating column of df_ratings to see the frequency of each rating.
# - Using a histogram on the year column of df_movies to see the number of movies released year year, only including the movies that have been rated.
# - Using scatter plots to visualise the relationships between release year and average rating, as an example.

# Merge the dataframes on 'Movie_ID'
df_merged = pd.merge(df_movies, df_ratings, on='Movie_ID')
# Find unique years with ratings
unique_years_with_ratings = df_merged['Year'].unique()
# Filter df_movies using the unique years
df_movies_filtered = df_movies[df_movies['Year'].isin(unique_years_with_ratings)]

# Histogram for Ratings using df_ratings_filtered
plt.hist(df_ratings['Rating'], bins=5, alpha=0.5, color='g', zorder=2)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(True, zorder=1)
plt.show()

# Histogram for Year using df_movies_filtered
plt.figure(figsize=(10, 6))
plt.hist(df_movies_filtered['Year'], bins=70, alpha=0.7, color='b', zorder=2)
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.title('Distribution of Movies Across Years')
plt.grid(True, zorder=1)
plt.show()

# Calculate Average Ratings for Scatter Plot
df_avg_ratings = df_merged.groupby(['Movie_ID', 'Year'])['Rating'].mean().reset_index()

# Scatter Plot for Average Rating by Release Year
plt.figure(figsize=(10, 6))
plt.scatter(df_avg_ratings['Year'], df_avg_ratings['Rating'], alpha=0.5)
plt.title('Average Rating by Release Year')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()


# ### Initial data transformation and graph analysis as a foundation for methods in W7 (Mining Social-Network Graphs/Betweeness Centrality)

# Add weighted edge based on the number of common raters
# Sample a smaller fraction for faster processing
df_ratings_sample = df_ratings.sample(frac=0.01)

user_movies_dict = df_ratings_sample.groupby('User_ID')['Movie_ID'].apply(set).to_dict()

edges_to_add = Counter()
for movies in user_movies_dict.values():
    edges_to_add.update(combinations(movies, 2))

# Create an empty graph
G = nx.Graph()

threshold = 5
G.add_edges_from((movie1, movie2, {'weight': weight}) for (movie1, movie2), weight in edges_to_add.items() if weight >= threshold)

# Compute average ratings for each movie
avg_ratings = df_ratings_sample.groupby('Movie_ID')['Rating'].mean().to_dict()

# Map node sizes based on ratings, with a scaling factor for visibility
node_sizes = [avg_ratings.get(movie, 0) * 100 for movie in G.nodes()]

# Movie titles, assuming df_movies DataFrame exists
movie_titles = df_movies.set_index('Movie_ID')['Name'].to_dict()

# Identify the top 20 movies by degree (connectivity)
top_20_degree_movies = sorted(dict(G.degree()).items(), key=lambda x: x[1], reverse=True)[:20]
top_20_movie_ids = [movie[0] for movie in top_20_degree_movies]

# Use a faster layout with fewer iterations
layout = nx.kamada_kawai_layout(G)

# Node colors based on average ratings
node_colors = [avg_ratings.get(movie, 0) for movie in G.nodes()]
cmap = plt.cm.coolwarm

weights = [G[u][v]['weight'] for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(12, 12))
nx.draw_networkx_nodes(G, layout, node_size=node_sizes, node_color=node_colors, cmap=cmap, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, layout, width=0.5, edge_color='grey', alpha=0.5, ax=ax)

ax.set_title("Movie Similarity Graph with Average Ratings")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), label='Average Rating', orientation='horizontal', ax=ax)
ax.axis('off')
plt.tight_layout()
plt.show()

# Create and display a table showcasing top 20 movies by degree
top_20_movie_names = [movie_titles[movie] for movie in top_20_movie_ids]
degree_values = [degree for _, degree in top_20_degree_movies]
data = {"Movie Names": top_20_movie_names, "Degree": degree_values}
df_top_20 = pd.DataFrame(data)

print(df_top_20)

# 2023-11-03 jennifer methods from W5.
# Preparing the data to be a managable size for the Apriori algorithm:
# Sampling based on users with the highest levels of activity, rather than the popularity of the movies.

# 1. Identify Active Users
# Considering the top 200 top users
top_users = df_ratings['User_ID'].value_counts().head(200).index 

# 2. Sample Ratings of Active Users
sampled_df = df_ratings[df_ratings['User_ID'].isin(top_users)]

# 3. Transform Data
# Convert the data to a user-item matrix format
movie_matrix = sampled_df.pivot_table(index='User_ID', columns='Movie_ID', values='Rating', aggfunc='size', fill_value=0)

# Reduce the dimensions by filtering out movies rated by few users (let's assume threshold = 10 as an example)
threshold = 10
movie_matrix = movie_matrix.loc[:, (movie_matrix.sum(axis=0) > threshold)]

# Convert into a binary matrix: watched or not
movie_matrix = (movie_matrix > 0).astype(bool)

# 4. Apply Apriori

# Find frequent itemsets using the binary matrix
frequent_itemsets = apriori(movie_matrix, min_support=0.1, max_len=2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Considering rules with high 'lift' and 'confidence', for simplicity.
strong_rules = rules[(rules['lift'] > 1.2) & (rules['confidence'] > 0.5)]

# Create the associations dictionary
# Extract the associations from the rules
associations = {}
for index, rule in strong_rules.iterrows():
    genre = list(rule['antecedents'])[0]
    associated_genre = list(rule['consequents'])[0]
    if genre not in associations:
        associations[genre] = [associated_genre]
    else:
        associations[genre].append(associated_genre)

# print(associations)
# each genre was associated with several other genres.
# suggests strong patterns of co-watching. 

# Understand viewer preferences to guide content acquisition or production
# Integrate data by merging tmbd datasets.
merged_df = pd.merge(tmdb_movies, credits, left_on='id', right_on='movie_id')
# checking the merged dataframe
print(merged_df.head()) 

# Extract genre names into a list of dictionaries
merged_df['genres'] = merged_df['genres'].apply(lambda x: [i['name'] for i in eval(x)])
# checking the genre transformation
print(merged_df['genres'].head())

# Extract top 3 cast members
merged_df['cast'] = merged_df['cast'].apply(lambda x: [i['name'] for i in eval(x)][:3])
# checking the cast transformation
print(merged_df['cast'].head())

# Extract production company names
merged_df['production_companies'] = merged_df['production_companies'].apply(lambda x: [i['name'] for i in eval(x)])
# checking the production company transformation
print(merged_df['production_companies'].head())

# Creating a 'Star Power' metrix to quantify the influence or popularity of an actor.
# Later use this as a feature in the dataset.
# Calculate actor frequencies
actor_frequency = merged_df['cast'].explode().value_counts().to_dict()

# Calculate average revenue and rating for each actor
actor_avg_revenue = merged_df.explode('cast').groupby('cast')['revenue'].mean().to_dict()
actor_avg_rating = merged_df.explode('cast').groupby('cast')['vote_average'].mean().to_dict()

# Normalize the metrics
def normalize_metric(metric_dict):
    max_val = max(metric_dict.values())
    min_val = min(metric_dict.values())
    return {actor: (value-min_val)/(max_val-min_val) for actor, value in metric_dict.items()}

normalized_frequency = normalize_metric(actor_frequency)
normalized_avg_revenue = normalize_metric(actor_avg_revenue)
normalized_avg_rating = normalize_metric(actor_avg_rating)

# Calculate Star Power for each actor
star_power = {}
for actor in normalized_frequency.keys():
    star_power[actor] = (0.5 * normalized_frequency[actor]) + (0.3 * normalized_avg_revenue.get(actor, 0)) + (0.2 * normalized_avg_rating.get(actor, 0))

# Integrate Star Power into main dataframe
merged_df['movie_star_power'] = merged_df['cast'].apply(lambda x: sum([star_power.get(actor, 0) for actor in x]))

# Making a cool recommendation system for users based on star power
# Genre need to be one-hot encoded.
merged_df['genres'] = merged_df['genres'].apply(lambda x: [i['name'] for i in eval(str(x))] if isinstance(x, str) else x)
vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
genre_ohe = vectorizer.fit_transform(merged_df['genres']).toarray()

# Convert the one-hot encoded array into a DataFrame, concatenate it with merged_df:
genre_df = pd.DataFrame(genre_ohe, columns=vectorizer.get_feature_names_out().tolist())
merged_df = pd.concat([merged_df, genre_df.add_prefix('genre_')], axis=1)

# Let's combine 'Star Power' and genre features for similarity computation
features = merged_df[['movie_star_power'] + list(merged_df.columns[merged_df.columns.str.contains('genre_')])]

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Compute cosine similarity
similarity_matrix = cosine_similarity(features_scaled)

## Using the movie_star_power similarity matrix to define the recommendation function.
# takes a movie title as input and returns a list of movies most similar to it
def recommend_movie(title, num_recommendations=5):
    # Get the index of the movie from its title
    idx = merged_df[merged_df['title_x'] == title].index[0]

    # Get the pairwise similarity scores
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort the movies based on similarity scores
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top n most similar movies (excluding itself)
    top_movies_idx = [i[0] for i in scores_sorted[1:num_recommendations+1]]

    # Return the top n most similar movie titles
    return merged_df['title_x'].iloc[top_movies_idx]

recommendations = recommend_movie("Inception")
print("1. Recommendations for 'Inception' using basic similarity:")
print(recommendations)

# More transpatent use of the star_power feature
# Where user can decide how much the stat_power should influence the recommendation.
def recommend_movie_weighted(title, star_power_weight, num_recommendations=5):
    idx = merged_df[merged_df['title_x'] == title].index[0]

    # Modifying the similarity scores based on star_power_weight
    weighted_scores = [(i, score * (1 + merged_df['movie_star_power'].iloc[i] * star_power_weight)) for i, score in enumerate(similarity_matrix[idx])]

    # Sort the movies based on the weighted similarity scores
    scores_sorted = sorted(weighted_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top n most similar movies (excluding itself)
    top_movies_idx = [i[0] for i in scores_sorted[1:num_recommendations+1]]

    # Return the top n most similar movie titles
    return merged_df['title_x'].iloc[top_movies_idx]

# Allow user to influence the recommendation by star_power
star_power_weight = float(input("Enter weight for star power influence (e.g., 0.5 for 50% more influence): "))
weighted_recommendations = recommend_movie_weighted("Inception", star_power_weight)
print("2. Weighted recommendations for 'Inception':")
print(weighted_recommendations)

# Enhanced recommentation with apriori integrated
# helper function to extract movie name
def extract_genres(genre_list):
    if not isinstance(genre_list, list):
        return []
    return [genre_dict["name"] for genre_dict in genre_list if isinstance(genre_dict, dict) and "name" in genre_dict]

def create_genre_id_name_mapping():
    """Create a dictionary to map genre IDs to names using the 'genres' column."""
    genre_id_name_mapping = {}
    for genre_list in merged_df['genres']:
        for genre in genre_list:
            if isinstance(genre, dict) and 'id' in genre and 'name' in genre:
                genre_id_name_mapping[genre['id']] = genre['name']
    return genre_id_name_mapping

def apriori_recommendations(movie_genre):
    """Recommend movies based on Apriori genre associations."""
    associated_genres = associations.get(movie_genre, [])
    print(f"Associated genres for {movie_genre}: {associated_genres}")  # Diagnostic print
    
    # Added diagnostic print to see what genres are in associated_genres
    print(f"Genres to be matched: {associated_genres}")
    
    def genre_matches(genres_of_movie):
        associated_genres = associations.get(movie_genre, [])
        return any(genre in genres_of_movie for genre in associated_genres)

    recommended_movies = merged_df[merged_df['genres'].apply(genre_matches)]
    return recommended_movies['title_x']

# example use case:
print("3. Apriori genre-based recommendations for 'Action':")
print(apriori_recommendations("Action"))

def enhanced_recommendation(title, star_power_weight, num_recommendations=5):
    star_power_recommendations = recommend_movie_weighted(title, star_power_weight, num_recommendations)
    
    movie_genre = merged_df[merged_df['title_x'] == title]['genres'].iloc[0]
    if movie_genre:
        apriori_recs = apriori_recommendations(movie_genre[0])  # Using the primary genre for simplicity
        # Concatenate and return combined recommendations, while ensuring no duplicates
        combined_recommendations = pd.concat([star_power_recommendations, apriori_recs]).drop_duplicates().head(num_recommendations)
        return combined_recommendations
    else:
        return star_power_recommendations

# example use case:
print("4. Enhanced recommendations for 'Die Hard' with star power influence of 0.7:")
print(enhanced_recommendation("Die Hard", 0.7, 10))
