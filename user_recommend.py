import pandas as pd
import numpy as np
import pickle
import os
import requests
import difflib
from sklearn.metrics.pairwise import cosine_similarity

# ── GraphQL Query ──────────────────────────────────────────────────────────────
ANILIST_API = "https://graphql.anilist.co"
USER_LIST_QUERY = """
query ($userName: String) {
  MediaListCollection(userName: $userName, type: ANIME) {
    lists {
      name
      entries {
        status
        score
        media {
          id
          title {
            romaji
            english
          }
        }
      }
    }
  }
}
"""

def fetch_user_list_api(username: str):
    """Fetches the user's anime list from the AniList API."""
    print(f"Fetching anime list for user: {username}...")
    variables = {"userName": username}
    
    response = requests.post(
        ANILIST_API,
        json={"query": USER_LIST_QUERY, "variables": variables},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=15
    )
    
    if response.status_code == 404:
        raise ValueError("User not found.")
    
    response.raise_for_status()
    result = response.json()
    
    if "errors" in result:
        raise ValueError(f"AniList API error: {result['errors'][0]['message']}")
        
    lists = result.get("data", {}).get("MediaListCollection", {}).get("lists", [])
    
    user_anime = []
    for lst in lists:
        for entry in lst.get("entries", []):
            media = entry.get("media", {})
            user_anime.append({
                "media_id": media.get("id"),
                "user_score": entry.get("score", 0),
                "user_status": entry.get("status", "")
            })
            
    return user_anime

def load_user_list_csv(csv_path: str):
    """Loads user anime list from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")
    
    df_user = pd.read_csv(csv_path)
    user_anime = []
    for _, row in df_user.iterrows():
        user_anime.append({
            "media_id": row.get("media_id"),
            "user_score": row.get("user_score", 0),
            "user_status": row.get("user_status", "")
        })
    return user_anime

def get_recommendations_for_user(user_anime, top_n=10, allow_ecchi=False):
    print("Loading recommendation system (this takes just a second)...")
    
    df = pd.read_pickle('anime_df.pkl')
    with open('anime_embeddings.pkl', 'rb') as f:
        anime_embeddings = pickle.load(f)

    # Identify what the user has watched to exclude it
    watched_ids = set([a['media_id'] for a in user_anime if pd.notna(a['media_id'])])
    
    # Build user profile embedding based on highly rated or completed shows
    liked_ids = []
    liked_weights = {}
    for a in user_anime:
        try:
            score = float(a['user_score']) if pd.notna(a['user_score']) else 0
        except ValueError:
            score = 0
            
        # Consider a show "liked" if scored >= 7 on a 10 scale, >= 70 on a 100 scale, or if they completed it
        if score >= 7 or score >= 70 or a['user_status'] == 'COMPLETED':
            liked_ids.append(a['media_id'])
            
            # Give double weighting for user_score > 80 (or > 8 on a 10-point scale)
            if score > 80 or (8 < score <= 10):
                liked_weights[a['media_id']] = 2.0
            else:
                liked_weights[a['media_id']] = 1.0
             
    if not liked_ids:
        # Fallback if no scores/completed shows found
        liked_ids = list(watched_ids)
        liked_weights = {mid: 1.0 for mid in liked_ids}

    # Get the positional indices of the liked anime in our local dataset
    liked_indices = np.where(df['id'].isin(liked_ids))[0].tolist()
    
    if not liked_indices:
        print("Not enough matching anime in the database to build a user profile.")
        return []
        
    # Create the user profile by taking a weighted average of embeddings of their liked shows
    user_embs = anime_embeddings[liked_indices]
    weights = np.array([liked_weights.get(df.iloc[idx]['id'], 1.0) for idx in liked_indices])
    profile_embedding = np.average(user_embs, axis=0, weights=weights).reshape(1, -1)
    
    # Calculate similarities against ALL anime
    similarities = cosine_similarity(profile_embedding, anime_embeddings)[0]
    
    # Apply boosts
    mean_scores = pd.to_numeric(df['mean_score'], errors='coerce').fillna(50)
    normalized_scores = mean_scores / 100.0
    score_boost = mean_scores.apply(lambda x: 1.2 if x >= 75 else 1.0)
    format_boost = df['format'].apply(lambda x: 1.0 if str(x).upper() in ['TV', 'MOVIE'] else 0.8)
    
    combined_scores = ((similarities * 0.8) + (normalized_scores.values * score_boost.values * 0.2)) * format_boost.values
    
    # Sort by combined score
    sorted_indices = combined_scores.argsort()[::-1]
    
    print("\n" + "=" * 60)
    print(f" Top {top_n} Recommendations For You ")
    print("=" * 60)
    
    count = 0
    seen_titles = []
    recommendations = []
    for idx in sorted_indices:
        row = df.iloc[idx]
        
        # Skip if already watched
        if row['id'] in watched_ids:
            continue
            
        # Filter out Ecchi/Hentai if not allowed
        if not allow_ecchi:
            g_str = str(row['genres']).lower() if pd.notna(row['genres']) else ""
            t_str = str(row['tags']).lower() if pd.notna(row['tags']) else ""
            if 'ecchi' in g_str or 'hentai' in g_str or 'ecchi' in t_str or 'hentai' in t_str:
                continue

        t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
        t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
        title = t_eng if t_eng else t_rom
        
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
            
        seen_titles.extend([t for t in (t_eng, t_rom) if t])
        genres = str(row['genres']).replace("|", ", ") if pd.notna(row['genres']) else "Unknown"
        fmt = row['format'] if pd.notna(row['format']) else "Unknown"
        score = row['mean_score'] if pd.notna(row['mean_score']) else "N/A"
        
        print(f"{count+1}. {title}")
        print(f"   Format: {fmt} | Score: {score}")
        print(f"   Genres: {genres}")
        print("-" * 60)
        
        recommendations.append({
            "title": title,
            "format": fmt,
            "score": score,
            "genres": genres
        })
        
        count += 1
        if count >= top_n:
            break
            
    return recommendations