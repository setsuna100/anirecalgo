import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading recommendation system (this takes just a second)...")

# 1. Load the DataFrame
df = pd.read_pickle('anime_df.pkl')

# 2. Load the pre-computed embeddings
with open('anime_embeddings.pkl', 'rb') as f:
    anime_embeddings = pickle.load(f)

# 3. Load the local NLP Model
model = SentenceTransformer('./local_anime_model')

def get_recommendations(query, top_n=5):
    # Convert user query to a semantic vector
    query_embedding = model.encode([query])
    # Compare the query vector to all anime vectors using cosine similarity
    similarities = cosine_similarity(query_embedding, anime_embeddings)
    # Sort and get the indices of the top matches
    top_indices = similarities[0].argsort()[-top_n:][::-1]
    
    return df.iloc[top_indices]

print("System Ready!\n" + "-" * 50)

if __name__ == "__main__":
    while True:
        user_query = input("\nWhat kind of anime would you like to watch? (type 'exit' to quit):\n> ")
        if user_query.lower() in ['exit', 'quit']:
            break
        
        recs = get_recommendations(user_query)
        for i, row in enumerate(recs.itertuples(), 1):
            title = row.title_english if row.title_english else row.title_romaji
            genres = str(row.genres).replace("|", ", ")
            print(f"{i}. {title} (Score: {row.average_score}) - {genres}")
        print("-" * 50)