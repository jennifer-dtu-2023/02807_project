import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from itertools import combinations
from collections import Counter
from zipfile import ZipFile
import os
#import requests
#import re


# used csv files, linking it with their source zip files
csv_to_zip_source = {
    "Netflix_Dataset_Movie.csv": "zip_sources/Netflix_Dataset_Movie.csv.zip",
    "Netflix_Dataset_Rating.csv": "zip_sources/Netflix_Dataset_Rating.csv.zip",
    "tmdb_5000_credits.csv": "zip_sources/tmdb.zip",
    "tmdb_5000_movies.csv": "zip_sources/tmdb.zip"
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

 [markdown]
# ### Initial EDA analysis as a foundation for the methods in W5 (Apripro/FP-Growth for frequent item-set mining)

 [markdown]
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

 [markdown]
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

 [markdown]
# ### Initial data transformation and graph analysis as a foundation for methods in W7 (Mining Social-Network Graphs/Betweeness Centrality)

 [markdown]
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


