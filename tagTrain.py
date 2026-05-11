import pandas as pd
import pickle

df = pd.read_csv('top5000.csv')

df = df.dropna()

# Convert each column to a string during concatenation to avoid TypeErrors
# Replacing '|' with a space helps the NLP model read them as distinct words
df['description'] = (
    df['title_english'].astype(str) + ' ' + 
    df['genres'].astype(str).str.replace('|', ' ') + ' ' + 
    df['tags'].astype(str).str.replace('|', ' ') + ' ' + 
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