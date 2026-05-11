import pandas as pd
import pickle

df = pd.read_csv('top5000.csv')

df = df.dropna()

# Convert each column to a string during concatenation to avoid TypeErrors
df['description'] = (
    df['title_english'].astype(str) + ' ' + 
    df['genres'].astype(str) + ' ' + 
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

from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(query, embeddings, df, top_n=5):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices]