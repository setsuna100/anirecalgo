import pandas as pd
import numpy as np
import pickle
import os
import requests
import difflib
from sklearn.metrics.pairwise import cosine_similarity

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

    def clean_id(val):
        s = str(val).strip()
        return s[:-2] if s.endswith('.0') else s

    id_to_idx = {clean_id(df.iloc[i]['id']): i for i in range(len(df))}
    franchise_graph = {aid: set() for aid in id_to_idx}

    if 'prequel_id' in df.columns and 'sequel_id' in df.columns:
        for aid, idx in id_to_idx.items():
            row = df.iloc[idx]
            if pd.notna(row.get('prequel_id')) and str(row['prequel_id']).strip():
                franchise_graph[aid].update(clean_id(x) for x in str(row['prequel_id']).split('|') if x.strip())
            if pd.notna(row.get('sequel_id')) and str(row['sequel_id']).strip():
                franchise_graph[aid].update(clean_id(x) for x in str(row['sequel_id']).split('|') if x.strip())

    for node, neighbors in list(franchise_graph.items()):
        for neighbor in neighbors:
            if neighbor not in franchise_graph:
                franchise_graph[neighbor] = set()
            franchise_graph[neighbor].add(node)

    visited = set()
    idx_to_earliest_idx = {}
    franchise_components = []

    for node in franchise_graph:
        if node not in visited:
            component = set()
            stack = [node]
            while stack:
                curr = stack.pop()
                if curr not in visited:
                    visited.add(curr)
                    component.add(curr)
                    for neighbor in franchise_graph.get(curr, []):
                        if neighbor not in visited:
                            stack.append(neighbor)
            
            franchise_components.append(component)
            
            valid_indices = [id_to_idx[n] for n in component if n in id_to_idx]
            if valid_indices:
                def get_sort_key(i):
                    y = df.iloc[i].get('season_year')
                    try: return float(y)
                    except (ValueError, TypeError): return 9999.0
                
                earliest_idx = min(valid_indices, key=lambda i: (get_sort_key(i), i))
                for i in valid_indices:
                    idx_to_earliest_idx[i] = earliest_idx

    watched_ids = set([a['media_id'] for a in user_anime if pd.notna(a['media_id'])])
    watched_ids_clean = set(clean_id(x) for x in watched_ids)
    
    expanded_watched_ids = set(watched_ids_clean)
    for component in franchise_components:
        if not component.isdisjoint(watched_ids_clean):
            expanded_watched_ids.update(component)
    
    liked_ids = []
    liked_weights = {}
    for a in user_anime:
        try:
            score = float(a['user_score']) if pd.notna(a['user_score']) else 0
        except ValueError:
            score = 0
            
        if score >= 7 or score >= 70 or a['user_status'] == 'COMPLETED':
            liked_ids.append(a['media_id'])
            
            if score > 0:
                norm_score = score if score <= 10 else score / 10.0
                liked_weights[a['media_id']] = norm_score
            else:
                liked_weights[a['media_id']] = 7.0  # Baseline weight for unscored but completed shows
             
    if not liked_ids:
        liked_ids = list(watched_ids)
        liked_weights = {mid: 1.0 for mid in liked_ids}

    liked_indices = np.where(df['id'].isin(liked_ids))[0].tolist()
    
    if not liked_indices:
        print("Not enough matching anime in the database to build a user profile.")
        return []
        
    user_embs = anime_embeddings[liked_indices]
    weights = np.array([liked_weights.get(df.iloc[idx]['id'], 1.0) for idx in liked_indices])
    profile_embedding = np.average(user_embs, axis=0, weights=weights).reshape(1, -1)
    
    similarities = cosine_similarity(profile_embedding, anime_embeddings)[0]
    
    mean_scores = pd.to_numeric(df['mean_score'], errors='coerce').fillna(50)
    normalized_scores = mean_scores / 100.0
    score_boost = mean_scores.apply(lambda x: 1.5 if x >= 80 else (1.2 if x >= 70 else 1.0))
    format_boost = df['format'].apply(lambda x: 1.0 if str(x).upper() in ['TV', 'MOVIE'] else 0.8)
    
    combined_scores = ((similarities * 0.7) + (normalized_scores.values * score_boost.values * 0.3)) * format_boost.values
    
    sorted_indices = combined_scores.argsort()[::-1]
    
    print("\n" + "=" * 60)
    print(f" Top {top_n} Recommendations For You ")
    print("=" * 60)
    
    count = 0
    seen_titles = []
    recommendations = []
    selected_indices = []
    for original_idx in sorted_indices:
        # Map to the first season/earliest entry of the franchise
        idx = idx_to_earliest_idx.get(int(original_idx), int(original_idx))
        
        if idx in selected_indices:
            continue
            
        row = df.iloc[idx]
        
        # Exclude if the user has watched this show OR any prequel/sequel in the franchise
        if row['id'] in watched_ids or clean_id(row['id']) in expanded_watched_ids:
            continue
            
        if not allow_ecchi:
            g_str = str(row['genres']).lower() if pd.notna(row['genres']) else ""
            t_str = str(row['tags']).lower() if pd.notna(row['tags']) else ""
            if 'ecchi' in g_str or 'hentai' in g_str or 'ecchi' in t_str or 'hentai' in t_str:
                continue

        t_eng = str(row['title_english']) if pd.notna(row['title_english']) and row['title_english'] else ""
        t_rom = str(row['title_romaji']) if pd.notna(row['title_romaji']) and row['title_romaji'] else ""
        title = t_eng if t_eng else t_rom
        
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
        
        episodes = ""
        if pd.notna(row['season_year']) and str(row['season_year']).strip():
            try:
                year = int(float(row['season_year']))
            except ValueError:
                year = row['season_year']
                
        match_pct = f"{similarities[idx] * 100:.1f}%"
                
        recommendations.append({
            "title": title,
            "format": fmt,
            "episodes": episodes,
            "score": score,
            "genres": genres,
            "year": year,
            "match_score": match_pct
        })
        
        selected_indices.append(idx)
        count += 1
        if count >= top_n:
            break
            
    return recommendations