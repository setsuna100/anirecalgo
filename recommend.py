import pandas as pd
import pickle
import difflib
import re
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

# 4. Extract all unique genres and tags from the dataframe for keyword detection
all_genres_tags = set()
if 'genres' in df.columns:
    for g_val in df['genres'].dropna():
        for g in str(g_val).split('|'):
            if g.strip(): all_genres_tags.add(g.strip().lower())
if 'tags' in df.columns:
    for t_val in df['tags'].dropna():
        for t in str(t_val).split('|'):
            if t.strip(): all_genres_tags.add(t.strip().lower())

# Sort keywords by length descending so longer phrases match before shorter ones
valid_keywords = sorted(list(all_genres_tags), key=len, reverse=True)

def get_recommendations(query, top_n=5):
    original_query_lower = query.lower()
    search_query = query
    
    found_titles = []
    for i in range(len(df)):
        row = df.iloc[i]
        t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
        t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
        
        if len(t_eng) > 3 and t_eng.lower() in original_query_lower:
            found_titles.append((t_eng, i))
        if len(t_rom) > 3 and t_rom.lower() in original_query_lower:
            found_titles.append((t_rom, i))
            
    # Sort by length descending to replace longer titles first (e.g., "Attack on Titan" before "Titan")
    found_titles.sort(key=lambda x: len(x[0]), reverse=True)
    
    replaced_titles = []
    for title, idx in found_titles:
        if any(title.lower() in r.lower() for r in replaced_titles):
            continue
        match_row = df.iloc[idx]
        genres_str = str(match_row['genres']).replace('|', ' ') if pd.notna(match_row['genres']) else ""
        tags_str = str(match_row['tags']).replace('|', ' ') if pd.notna(match_row['tags']) else ""
        
        search_query = re.sub(re.escape(title), f"{genres_str} {tags_str}", search_query, flags=re.IGNORECASE)
        replaced_titles.append(title)
        
    if replaced_titles:
        titles_str = ", ".join(replaced_titles)
        print(f"\n  [Detected anime in prompt: {titles_str}. Searching by tags instead...]")
        
    # Detect explicit genres or tags in the prompt
    mentioned_keywords = []
    temp_query = original_query_lower
    for kw in valid_keywords:
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, temp_query):
            mentioned_keywords.append(kw)
            temp_query = re.sub(pattern, ' ', temp_query)
            
    if mentioned_keywords:
        display_kws = ", ".join([kw.title() for kw in mentioned_keywords])
        print(f"\n  [Detected genres/tags in prompt: {display_kws}]")

    # Convert user query to a semantic vector
    query_embedding = model.encode([search_query])
    # Compare the query vector to all anime vectors using cosine similarity
    similarities = cosine_similarity(query_embedding, anime_embeddings)[0]
    
    # Normalize mean scores to be between 0 and 1, defaulting to 50 if missing
    mean_scores = pd.to_numeric(df['mean_score'], errors='coerce').fillna(50)
    normalized_scores = mean_scores / 100.0
    
    # Soft score filter: gives a 20% score weight boost to shows with a mean score of 75 and above
    score_boost = mean_scores.apply(lambda x: 1.2 if x >= 75 else 1.0)
    
    # Format filter: gives priority to TV series and Movies, penalizing OVAs, ONAs, and Specials
    format_boost = df['format'].apply(lambda x: 1.0 if str(x).upper() in ['TV', 'MOVIE'] else 0.8)
    
    # Calculate genre/tag boost based on explicit mentions
    keyword_boost = []
    for i in range(len(df)):
        row = df.iloc[i]
        item_keywords = set()
        if 'genres' in df.columns and pd.notna(row['genres']):
            item_keywords.update([g.strip().lower() for g in str(row['genres']).split('|')])
        if 'tags' in df.columns and pd.notna(row['tags']):
            item_keywords.update([t.strip().lower() for t in str(row['tags']).split('|')])
        
        matches = sum(1 for kw in mentioned_keywords if kw in item_keywords)
        keyword_boost.append(1.0 + (0.3 * matches)) # 30% score boost per matching keyword
        
    # Create a combined score and apply format weight + keyword boost
    combined_scores = ((similarities * 0.8) + (normalized_scores.values * score_boost.values * 0.2)) * format_boost.values * pd.Series(keyword_boost).values
    
    # Sort and get the indices of all matches based on the combined score
    sorted_indices = combined_scores.argsort()[::-1]
    
    selected_indices = []
    seen_titles = []
    query_lower = original_query_lower

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
            print(f"{i}. {title} ({row.format}, Score: {row.mean_score}) - {genres}")
        print("-" * 50)