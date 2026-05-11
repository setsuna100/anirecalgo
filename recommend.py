import pandas as pd
import pickle
import difflib
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
    # Sort and get the indices of all matches
    sorted_indices = similarities[0].argsort()[::-1]
    
    selected_indices = []
    seen_titles = []
    query_lower = query.lower()

    for idx in sorted_indices:
        row = df.iloc[idx]
        t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
        t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
        
        # Skip if the show is directly mentioned in the user's prompt
        if (len(t_eng) > 3 and t_eng.lower() in query_lower) or (len(t_rom) > 3 and t_rom.lower() in query_lower):
            continue
            
        # Check for sequels or very similar names to already selected recommendations
        is_duplicate = False
        for seen in seen_titles:
            if t_eng and (t_eng.lower() in seen.lower() or seen.lower() in t_eng.lower() or difflib.SequenceMatcher(None, t_eng.lower(), seen.lower()).ratio() > 0.75):
                is_duplicate = True
                break
            if t_rom and (t_rom.lower() in seen.lower() or seen.lower() in t_rom.lower() or difflib.SequenceMatcher(None, t_rom.lower(), seen.lower()).ratio() > 0.75):
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
            
        selected_indices.append(idx)
        if t_eng: seen_titles.append(t_eng)
        if t_rom: seen_titles.append(t_rom)
        
        if len(selected_indices) == top_n:
            break

    return df.iloc[selected_indices]

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