import pandas as pd
import pickle

df = pd.read_csv('top5000.csv')

df = df.fillna('')

def get_weighted_tags_string(tags_str: str, max_repetitions: int = 5, weight_multiplier: float = 1.0) -> str:

    if not isinstance(tags_str, str) or not tags_str:
        return ""

    tags = [tag.strip() for tag in tags_str.split('|') if tag.strip()]
    n_tags = len(tags)

    if n_tags == 0:
        return ""
    
    if n_tags == 1:
        return ' '.join([tags[0]] * max(1, round(weight_multiplier * max_repetitions)))

    weighted_tag_list = []
    for i, tag in enumerate(tags):
        # Exponential scale from 1.0 (100%) down to 0.2 (20%)
        weight = (0.2 ** (i / (n_tags - 1))) * weight_multiplier
        
        # Calculate repetitions, ensuring at least 1
        repetitions = max(1, round(weight * max_repetitions))
        weighted_tag_list.extend([tag] * int(repetitions))
        
    return ' '.join(weighted_tag_list)

def get_weighted_genres_string(genres_str: str, max_repetitions: int = 5) -> str:
    if not isinstance(genres_str, str) or not genres_str:
        return ""

    genres = [g.strip() for g in genres_str.split('|') if g.strip()]
    
    weighted_genre_list = []
    for genre in genres:
        # Give a 0.9 weighting penalty to action and drama
        weight = 0.9 if genre.lower() in ['action', 'drama'] else 1.0
        repetitions = max(1, round(weight * max_repetitions))
        weighted_genre_list.extend([genre] * int(repetitions))
        
    return ' '.join(weighted_genre_list)

df['weighted_tags'] = df['tags'].astype(str).apply(lambda x: get_weighted_tags_string(x, max_repetitions=10, weight_multiplier=1.2))
df['weighted_sp_tags'] = df['sp_tags'].astype(str).apply(lambda x: get_weighted_tags_string(x,max_repetitions=10, weight_multiplier=0.8))
df['weighted_genres'] = df['genres'].astype(str).apply(get_weighted_genres_string)

df['description'] = (
    df['title_english'].astype(str) + ' ' + 
    df['weighted_genres'] + ' ' + 
    df['weighted_tags'] + ' ' + 
    df['weighted_sp_tags'] + ' ' + 
    df['format'].astype(str) + ' episodes: ' + 
    df['episodes'].astype(str)
)

df['description_no_spoilers'] = (
    df['title_english'].astype(str) + ' ' + 
    df['weighted_genres'] + ' ' + 
    df['weighted_tags'] + ' ' + 
    df['format'].astype(str) + ' episodes: ' + 
    df['episodes'].astype(str)
)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

anime_embeddings = model.encode(df['description'].tolist())
anime_embeddings_no_spoilers = model.encode(df['description_no_spoilers'].tolist())

print("Saving embeddings and model...")

with open('anime_embeddings.pkl', 'wb') as f:
    pickle.dump(anime_embeddings, f)
with open('anime_embeddings_no_spoilers.pkl', 'wb') as f:
    pickle.dump(anime_embeddings_no_spoilers, f)
df.to_pickle('anime_df.pkl')

model.save('./local_anime_model')
print("Saved successfully!")