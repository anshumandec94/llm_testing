"""
This script takes the ml-32m ratings dataset's movies.csv data and for every movie, queries the TMBB API
to get the movie's overview. The output is a csv file with two columns: movieId and overview. For any API error, the
stored overview is an empty string. The output csv file is stored in the same directory as this script and is named movie_overviews.csv.
"""
import tmdbsimple as tmdb
import os
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from tqdm import tqdm
def load_movies_links_df():
    links_df = pd.read_csv("data/ml-32m/links.csv")
    return links_df

def get_movie_overview_for_tmdb_id(tmdb_id):
    try:
        movie = tmdb.Movies(tmdb_id)
        response = movie.info()
        overview = response.get("overview", "")
        return overview
    except Exception as e:
        print(f"Error fetching overview for TMDB ID {tmdb_id}: {e}")
        return ""

def get_movie_overviews():
    tqdm.pandas()
    tmdb.API_KEY = os.getenv("TMDB_API_KEY")
    links_df = load_movies_links_df()

    
    links_df['overview'] = links_df['tmdbId'].progress_apply(lambda x: get_movie_overview_for_tmdb_id(x) if pd.notnull(x) else "")
    
    overviews_df = links_df[['movieId', 'overview']]
    overviews_df.to_csv("data/ml-32m/movie_overviews.csv", index=False)

if __name__ == "__main__":
    get_movie_overviews()