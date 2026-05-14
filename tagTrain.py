import pandas as pd
import pickle

df = pd.read_csv('top5000.csv')

df = df.dropna()

def get_weighted_tags_string(tags_str: str, max_repetitions: int = 5) -> str:
    """
    Processes a pipe-separated string of tags, weighting them by order.
    The first tag gets 100% weight, the last gets 20%, scaled linearly.
    Weight is implemented by repeating the tag to influence the NLP model.
    """
    if not isinstance(tags_str, str) or not tags_str:
        return ""

    tags = [tag.strip() for tag in tags_str.split('|') if tag.strip()]
    n_tags = len(tags)

    if n_tags == 0:
        return ""
    
    # If only one tag, give it max weight
    if n_tags == 1:
        return ' '.join([tags[0]] * max_repetitions)

    weighted_tag_list = []
    for i, tag in enumerate(tags):
        # Linear scale from 1.0 (100%) down to 0.2 (20%)
        weight = 1.0 - (0.8 * i / (n_tags - 1))
        
        # Calculate repetitions, ensuring at least 1
        repetitions = max(1, round(weight * max_repetitions))
        weighted_tag_list.extend([tag] * int(repetitions))
        
    return ' '.join(weighted_tag_list)

# Create a new column with weighted tags
df['weighted_tags'] = df['tags'].astype(str).apply(get_weighted_tags_string)

# Convert each column to a string during concatenation to avoid TypeErrors
# Replacing '|' with a space helps the NLP model read them as distinct words
df['description'] = (
    df['title_english'].astype(str) + ' ' + 
    df['genres'].astype(str).str.replace('|', ' ') + ' ' + 
    df['weighted_tags'] + ' ' + 
    df['format'].astype(str) + ' episodes: ' + 
    df['episodes'].astype(str)
)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

anime_embeddings = model.encode(df['description'].tolist())

print("Saving embeddings and model...")
# Save the generated embeddings and dataframe
with open('anime_embeddings.pkl', 'wb') as f:
    pickle.dump(anime_embeddings, f)
df.to_pickle('anime_df.pkl')

# Save the transformer model locally
model.save('./local_anime_model')
print("Saved successfully!")